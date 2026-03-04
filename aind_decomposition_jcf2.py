"""
aIND via Joint Coupling Flow v2  (JCF2)

Fixes three weaknesses of jcf.py, combining the best of jcf.py and aind_decomposition.py:

  (1) Product Gaussian NLL drove A → 0 (no informativeness signal).
      Fix: Replace with BIVARIATE Gaussian NLL + learnable scalar rho.
           Cross-term  -rho*z_phi*z_psi  provides gradient to push rho ≠ 0.
           log(1-rho^2) regularises rho away from ±1.

  (2) L_perp = Cov(r, z_psi)^2 is algebraically 0 by OLS → no signal.
      LATENT HSIC(r, z_psi) only catches independence in the Gaussianized space,
      which may not translate to physical-space independence if flows are imperfect.
      Fix: Physical-space HSIC_RBF(Phi_R_decoded, Phi_I_decoded), matching the
           I(Phi_R; Phi_I) criterion used by aind_decomposition.py.

  (3) No energy constraint — nothing prevents Phi_I -> 0.
      Fix: lambda3 * L_Smooth = ||Phi - Phi_I||^2

aIND Loss  (L_total = L_Flow + lambda1*L_Func + lambda2*L_Indep + lambda3*L_Smooth)
------------------------------------------------------------------------------
    L_Flow   : Bivariate NLL  [ -log p_Z(Z) - log|det J| ]
    L_Func   : ||Z_Psi - Z_I||^2   [ functional Psi = F(Phi_I) in latent ]
    L_Indep  : HSIC(Phi_I, Phi_R) or Corr(Z_I, Z_R)^2
    L_Smooth : ||Phi_std - Phi_I||^2  [ residual / anti-collapse ]

    rho  = tanh(rho_raw)   [learnable]
    A    = Cov(z_phi, z_psi) / Var(z_psi)   [closed-form OLS]

Comparison table
----------------
    aind_decomposition.py : I(Phi_R; Phi_I) [physical] + gamma*MSE    [physical space ✓]
                            Architecture (DSF bijection) = informativeness [architectural ✓]
                            No Gaussianization — raw-space MI only         [latent ✗]

    jcf.py (product NLL)  : L_flow → A = 0 (trivial), L_perp ≡ 0 (no signal)   [❌]

    jcf2.py (this file)   : L_biv  → rho ≠ 0 (informativeness)       [latent ✓]
                            L_white → conditional variance             [latent ✓]
                            HSIC(Phi_I, Phi_R) physical               [physical ✓]
                            gamma*MSE physical                         [physical ✓]

Training decode step
--------------------
Physical-space terms require decoding during training:
    z_phi_I = A * z_psi
    Phi_I_std = pi_phi( f^{-1}(z_phi_I, z_psi) )   [coupling inverse, same pass]
    Phi_R_std = Phi_std - Phi_I_std

Coupling layer inverses are analytical (same cost as forward pass), so
this adds only one extra network evaluation per training step.

Architecture and Decomposition
-------------------------------
    K alternating JointCouplingLayers → full joint inverse → Phi_I
    A       = Cov(z_phi, z_psi) / Var(z_psi)   [OLS, differentiable]
    z_phi^I = A * z_psi  =  projection of z_phi onto z_psi  [source-derived]
    Phi^I   = pi_phi( f^{-1}(z_phi^I, z_psi) )
    Phi^R   = Phi - Phi^I

Anti-cheating safeguards
------------------------
  (1) Strict latent partitioning: z_phi^I is the projection of z_phi onto z_psi,
      i.e. the component of the SOURCE latent that aligns with the target direction.
  (2) Inference test: run decomposition with Phi-only (Psi=dummy). If corr(Phi_I, Psi)
      drops sharply vs (Phi,Psi) case, the model was shortcutting through Psi.
  (3) lambda3 (L_Smooth = ||Phi - Phi_I||^2): set high enough to anchor Phi_I in Phi.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error

from toy_settings import (
    phi_func,
    psi_plus_func,
    f_func,
    g_func,
    F_analytical,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_decomposition(Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true=None):
    m = {}
    m["mutual_information"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Phi_I, random_state=42)[0]
    m["residual_energy"] = mean_squared_error(Phi, Phi_I)
    m["mi_phiI_psi"] = mutual_info_regression(
        Phi_I.reshape(-1, 1), Psi_plus, random_state=42)[0]
    m["mi_phiR_psi"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Psi_plus, random_state=42)[0]
    m["gt_error"] = (
        mean_squared_error(Phi_I_true, Phi_I)
        if Phi_I_true is not None else float("nan")
    )
    m["total_loss"] = m["mutual_information"] + m["residual_energy"]
    return m


# ---------------------------------------------------------------------------
# Coupling network helper (scalar → (s, t))
# ---------------------------------------------------------------------------

def _make_st_net(hidden_dim: int, n_hidden: int) -> nn.Sequential:
    layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, 2))
    net = nn.Sequential(*layers)
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


# ---------------------------------------------------------------------------
# Joint Coupling Layer  (identical to jcf.py)
# ---------------------------------------------------------------------------

class JointCouplingLayer(nn.Module):
    """
    One coupling block on scalar pairs (phi_i, psi_i).

    reverse=False  (even layers):
        psi' = psi * exp(s_psi(phi))  + t_psi(phi)
        phi' = phi * exp(s_phi(psi')) + t_phi(psi')

    reverse=True   (odd layers):
        phi' = phi * exp(s_phi(psi))  + t_phi(psi)
        psi' = psi * exp(s_psi(phi')) + t_psi(phi')

    s is tanh-squashed to (-2, 2) → exp(s) ∈ (e^{-2}, e^{2}).
    """

    def __init__(self, hidden_dim: int = 32, n_hidden: int = 2, reverse: bool = False):
        super().__init__()
        self.reverse = reverse
        self.net_A = _make_st_net(hidden_dim, n_hidden)
        self.net_B = _make_st_net(hidden_dim, n_hidden)

    def _st(self, net, x):
        out = net(x.unsqueeze(-1))         # (N, 2)
        s = 2.0 * torch.tanh(out[:, 0])   # in (-2, 2)
        t = out[:, 1]
        return s, t

    def forward(self, phi, psi):
        if not self.reverse:
            s1, t1 = self._st(self.net_A, phi)
            psi_new = psi * s1.exp() + t1
            s2, t2 = self._st(self.net_B, psi_new)
            phi_new = phi * s2.exp() + t2
        else:
            s1, t1 = self._st(self.net_A, psi)
            phi_new = phi * s1.exp() + t1
            s2, t2 = self._st(self.net_B, phi_new)
            psi_new = psi * s2.exp() + t2
        return phi_new, psi_new, s1 + s2

    def inverse(self, phi_new, psi_new):
        if not self.reverse:
            s2, t2 = self._st(self.net_B, psi_new)
            phi = (phi_new - t2) * (-s2).exp()
            s1, t1 = self._st(self.net_A, phi)
            psi = (psi_new - t1) * (-s1).exp()
        else:
            s2, t2 = self._st(self.net_B, phi_new)
            psi = (psi_new - t2) * (-s2).exp()
            s1, t1 = self._st(self.net_A, psi)
            phi = (phi_new - t1) * (-s1).exp()
        return phi, psi


# ---------------------------------------------------------------------------
# Joint Coupling Flow v2  (adds learnable rho)
# ---------------------------------------------------------------------------

class JointCouplingFlow2(nn.Module):
    """
    K alternating JointCouplingLayers + output affine normalisation
    + learnable scalar rho (used only in the bivariate NLL loss).

    rho = tanh(rho_raw)  ensures rho ∈ (-1, 1).
    At convergence rho ≈ A = Cov(z_phi, z_psi) / Var(z_psi).
    """

    def __init__(self, n_layers: int = 6, hidden_dim: int = 32, n_hidden: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            JointCouplingLayer(hidden_dim, n_hidden, reverse=(i % 2 == 1))
            for i in range(n_layers)
        ])
        self.log_scale_phi = nn.Parameter(torch.tensor(0.0))
        self.log_scale_psi = nn.Parameter(torch.tensor(0.0))
        self.shift_phi     = nn.Parameter(torch.tensor(0.0))
        self.shift_psi     = nn.Parameter(torch.tensor(0.0))
        # Learnable correlation parameter (used in bivariate NLL)
        self.rho_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def rho(self) -> torch.Tensor:
        return torch.tanh(self.rho_raw)

    def forward(self, phi, psi):
        log_det = phi.new_zeros(phi.shape[0])
        for layer in self.layers:
            phi, psi, ld = layer(phi, psi)
            log_det = log_det + ld
        eps     = 1e-8
        z_phi   = (phi - self.shift_phi) / (self.log_scale_phi.exp() + eps)
        z_psi   = (psi - self.shift_psi) / (self.log_scale_psi.exp() + eps)
        log_det = log_det - self.log_scale_phi - self.log_scale_psi
        return z_phi, z_psi, log_det

    def inverse(self, z_phi, z_psi):
        eps = 1e-8
        phi = z_phi * (self.log_scale_phi.exp() + eps) + self.shift_phi
        psi = z_psi * (self.log_scale_psi.exp() + eps) + self.shift_psi
        for layer in reversed(self.layers):
            phi, psi = layer.inverse(phi, psi)
        return phi, psi


# ---------------------------------------------------------------------------
# Closed-form OLS  (differentiable)
# ---------------------------------------------------------------------------

def compute_A(z_phi: torch.Tensor, z_psi: torch.Tensor) -> torch.Tensor:
    z_phi_c = z_phi - z_phi.mean()
    z_psi_c = z_psi - z_psi.mean()
    cov = (z_phi_c * z_psi_c).mean()
    var = (z_psi_c ** 2).mean().clamp(min=1e-8)
    return cov / var


# ---------------------------------------------------------------------------
# HSIC with RBF kernel  (nonlinear independence test)
# ---------------------------------------------------------------------------

def hsic_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    max_n: int = 2000,
) -> torch.Tensor:
    """
    Biased HSIC estimator with RBF kernels and median-heuristic bandwidth.

    X, Y are 1-D tensors of length N.
    Subsamples to max_n points to control O(N^2) cost.
    Returns a scalar >= 0;  HSIC = 0  iff X ⊥ Y (for universal kernels).
    """
    N = X.shape[0]
    if N > max_n:
        idx = torch.randperm(N, device=X.device)[:max_n]
        X = X[idx]
        Y = Y[idx]
    n = X.shape[0]

    # Pairwise squared distances
    dX = (X.unsqueeze(0) - X.unsqueeze(1)).pow(2)   # (n, n)
    dY = (Y.unsqueeze(0) - Y.unsqueeze(1)).pow(2)

    # Median bandwidth (no grad — only affects scale, not direction)
    with torch.no_grad():
        sigma_X = dX.median().sqrt().clamp(min=1e-4)
        sigma_Y = dY.median().sqrt().clamp(min=1e-4)

    KX = torch.exp(-dX / (2.0 * sigma_X ** 2))
    KY = torch.exp(-dY / (2.0 * sigma_Y ** 2))

    # Centre kernels:  H K H  where  H = I - (1/n) 11^T
    KX = KX - KX.mean(0, keepdim=True) - KX.mean(1, keepdim=True) + KX.mean()
    KY = KY - KY.mean(0, keepdim=True) - KY.mean(1, keepdim=True) + KY.mean()

    return (KX * KY).sum() / (n - 1) ** 2


# ---------------------------------------------------------------------------
# Combined loss  (aIND specification)
# ---------------------------------------------------------------------------
#
# L_total = L_Flow + lambda1*L_Func + lambda2*L_Indep + lambda3*L_Smooth
#
# A. L_Flow   : Flow NLL (change-of-variables)
# B. L_Func   : ||Z_Psi - Z_I||^2  (functional: Psi = F(Phi_I) in latent)
# C. L_Indep  : I(Phi_I; Phi_R) via HSIC or Corr(Z_I, Z_R)^2
# D. L_Smooth : ||Phi - Phi_I||^2  (residual energy; anchors Phi_I in Phi.
#              Set lambda3 high enough to force informative component to stay
#              grounded in the physical reality of the source.)
# L_recon    : ||Phi - (Phi_I + Phi_R)||^2  (sanity: should be ~0 by construction)
# ---------------------------------------------------------------------------


def _correlation_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Correlation(Z_I, Z_R)^2 for latent-space independence (differentiable)."""
    xc = x - x.mean()
    yc = y - y.mean()
    cov = (xc * yc).mean()
    vx = (xc ** 2).mean().clamp(min=1e-10)
    vy = (yc ** 2).mean().clamp(min=1e-10)
    corr = cov / (vx * vy).sqrt()
    return corr ** 2


def jcf2_loss(
    z_phi:      torch.Tensor,
    z_psi:      torch.Tensor,
    z_phi_I:    torch.Tensor,
    z_phi_R:    torch.Tensor,
    log_det:    torch.Tensor,
    rho:        torch.Tensor,
    Phi_I_std:  torch.Tensor,
    Phi_R_std:  torch.Tensor,
    phi_t:      torch.Tensor,
    lambda1:    float = 1.0,
    lambda2:    float = 1.0,
    lambda3:    float = 1.0,
    indep_mode: str   = "hsic",
    max_hsic_n: int   = 2000,
):
    """
    aIND loss: L_Flow + lambda1*L_Func + lambda2*L_Indep + lambda3*L_Smooth

    L_Flow   : Bivariate Gaussian NLL  [ -log p_Z(Z) - log|det J| ]
    L_Func   : ||Z_Psi - Z_I||^2       [ functional Psi = F(Phi_I) ]
    L_Indep  : HSIC(Phi_I, Phi_R) or Corr(Z_I, Z_R)^2
    L_Smooth : ||Phi_std - Phi_I||^2  [ residual / anti-collapse ]
    """
    # --- A. L_Flow ---
    det  = (1.0 - rho ** 2).clamp(min=1e-6)
    quad = (z_phi ** 2 + z_psi ** 2 - 2.0 * rho * z_phi * z_psi) / det
    L_Flow = (0.5 * quad + 0.5 * torch.log(det) - log_det).mean()

    # --- B. L_Func: ||Z_Psi - Z_I||^2 ---
    L_Func = ((z_psi - z_phi_I) ** 2).mean()

    # --- C. L_Indep ---
    if indep_mode == "corr":
        L_Indep = _correlation_sq(z_phi_I, z_phi_R)
    else:
        L_Indep = hsic_rbf(Phi_I_std, Phi_R_std, max_n=max_hsic_n)

    # --- D. L_Smooth (residual energy, anchors Phi_I in Phi) ---
    L_Smooth = ((phi_t - Phi_I_std) ** 2).mean()

    # L_recon = ||Phi - (Phi_I + Phi_R)||^2  (should be ~0 by construction)
    L_recon = ((phi_t - (Phi_I_std + Phi_R_std)) ** 2).mean()

    total = L_Flow + lambda1 * L_Func + lambda2 * L_Indep + lambda3 * L_Smooth

    A = compute_A(z_phi, z_psi)
    mi_rho = float(-0.5 * np.log(max(1.0 - rho.item() ** 2, 1e-8)))
    mi_A   = float(-0.5 * np.log(max(1.0 - A.item() ** 2, 1e-8)))

    bd = {
        "total":  total.item(),
        "flow":   L_Flow.item(),
        "func":   L_Func.item(),
        "indep":  L_Indep.item(),
        "smooth": L_Smooth.item(),
        "recon":  L_recon.item(),
        "A":      A.item(),
        "rho":    rho.item(),
        "mi_rho": mi_rho,
        "mi_A":   mi_A,
    }
    return total, bd


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_jcf2(
    model:      JointCouplingFlow2,
    phi_t:      torch.Tensor,
    psi_t:      torch.Tensor,
    num_epochs: int   = 1000,
    lr:         float = 1e-3,
    lambda1:    float = 1.0,
    lambda2:    float = 1.0,
    lambda3:    float = 1.0,
    indep_mode: str   = "hsic",
    max_hsic_n: int   = 2000,
    grad_clip:  float = 5.0,
    verbose:    bool  = True,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    log_every = max(1, num_epochs // 10)
    history   = {k: [] for k in ("total", "flow", "func", "indep", "smooth", "recon", "A", "rho", "mi_rho", "mi_A")}

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward: (Phi_std, Psi_std) → latent
        z_phi, z_psi, log_det = model(phi_t, psi_t)

        # OLS projection in latent → informative latent code
        A       = compute_A(z_phi, z_psi)
        z_phi_I = A * z_psi
        z_phi_R = z_phi - z_phi_I

        # Decode to physical space — same coupling nets, one extra pass.
        # Gradients flow through the inverse because it reuses the same net weights.
        Phi_I_std, _ = model.inverse(z_phi_I, z_psi)
        Phi_R_std    = phi_t - Phi_I_std   # additive residual in standardised space

        # Combined loss (JCF bivariate NLL + aind.py physical-space terms)
        loss, bd = jcf2_loss(
            z_phi, z_psi, z_phi_I, z_phi_R, log_det, model.rho,
            Phi_I_std, Phi_R_std, phi_t,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            indep_mode=indep_mode,
            max_hsic_n=max_hsic_n,
        )

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == num_epochs:
            for k in history:
                history[k].append(bd[k])
            if verbose:
                print(
                    f"  [Epoch {epoch:5d}]  "
                    f"flow={bd['flow']:+.4f}  "
                    f"func={bd['func']:.4f}  "
                    f"indep={bd['indep']:.3e}  "
                    f"smooth={bd['smooth']:.4f}  "
                    f"rho={bd['rho']:+.4f}  A={bd['A']:+.4f}  "
                    f"I~={bd['mi_A']:.4f} nats"
                )

    model.eval()
    return history


# ---------------------------------------------------------------------------
# Inference test  (Phi-only: detect shortcutting through Psi)
# ---------------------------------------------------------------------------


def run_inference_test(
    model: torch.nn.Module,
    phi_t: torch.Tensor,
    psi_t: torch.Tensor,
    phi_std: float,
    phi_mean: float,
    Phi_I_with_psi: np.ndarray,
    Psi_flat: np.ndarray,
) -> dict:
    """
    Test whether the model shortcuts through Psi.

    Run decomposition with (Phi, Psi) and with (Phi, Psi_dummy=0).
    If corr(Phi_I, Psi) drops sharply in the Phi-only case, the model
    was relying on the target latent during decoding.
    """
    model.eval()
    with torch.no_grad():
        # Normal path: (Phi, Psi)
        z_phi, z_psi, _ = model(phi_t, psi_t)
        A = float(
            ((z_phi - z_phi.mean()) * (z_psi - z_psi.mean())).mean()
            / (z_psi - z_psi.mean()).pow(2).mean().clamp(min=1e-8)
        )
        z_phi_I = A * z_psi
        Phi_I_std, _ = model.inverse(z_phi_I, z_psi)
        Phi_I_normal = (Phi_I_std.cpu().numpy() * phi_std + phi_mean).ravel()

        # Phi-only path: (Phi, Psi_dummy=0)
        psi_dummy = torch.zeros_like(psi_t, device=psi_t.device)
        z_phi_d, z_psi_d, _ = model(phi_t, psi_dummy)
        A_d = float(
            ((z_phi_d - z_phi_d.mean()) * (z_psi_d - z_psi_d.mean())).mean()
            / (z_psi_d - z_psi_d.mean()).pow(2).mean().clamp(min=1e-8)
        )
        z_phi_I_d = A_d * z_psi_d
        Phi_I_std_d, _ = model.inverse(z_phi_I_d, z_psi_d)
        Phi_I_phi_only = (Phi_I_std_d.cpu().numpy() * phi_std + phi_mean).ravel()

    r_normal = float(np.corrcoef(Phi_I_normal, Psi_flat)[0, 1])
    r_phi_only = float(np.corrcoef(Phi_I_phi_only, Psi_flat)[0, 1])
    r_with_psi = float(np.corrcoef(Phi_I_with_psi, Psi_flat)[0, 1])

    return {
        "corr_PhiI_Psi_normal": r_normal,
        "corr_PhiI_Psi_phi_only": r_phi_only,
        "corr_PhiI_Psi_from_results": r_with_psi,
        "inference_test_pass": r_phi_only > 0.5 * r_normal,  # heuristic
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def aind_decomposition_jcf2(
    Phi,
    Psi_plus,
    n_layers:    int   = 6,
    hidden_dim:  int   = 32,
    n_hidden:    int   = 2,
    num_epochs:  int   = 1000,
    lr:          float = 1e-3,
    lambda1:     float = 1.0,
    lambda2:     float = 1.0,
    lambda3:     float = 1.0,
    indep_mode:  str   = "hsic",
    max_hsic_n:  int   = 2000,
    verbose:     bool  = True,
    seed:        int   = 42,
    device=None,
):
    """
    aIND via Joint Coupling Flow v2 (bivariate NLL + HSIC).

    Steps
    -----
    1.  Pre-standardise (Phi, Psi+).
    2.  Train JointCouplingFlow2 with L_biv + L_white + L_hsic.
    3.  Compute A = Cov(z_phi, z_psi) / Var(z_psi).
    4.  z_phi^I = A*z_psi,  z_phi^R = z_phi - z_phi^I.
    5.  Decode: Phi^I = pi_phi(f^{-1}(z_phi^I, z_psi)).
    6.  Phi^R = Phi - Phi^I.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device: {device}")

    phi_flat = Phi.ravel()
    psi_flat = Psi_plus.ravel()

    phi_mean, phi_std = phi_flat.mean(), phi_flat.std() + 1e-8
    psi_mean, psi_std = psi_flat.mean(), psi_flat.std() + 1e-8
    Phi_std = (phi_flat - phi_mean) / phi_std
    Psi_std = (psi_flat - psi_mean) / psi_std

    phi_t = torch.tensor(Phi_std, dtype=torch.float32, device=device)
    psi_t = torch.tensor(Psi_std, dtype=torch.float32, device=device)

    model = JointCouplingFlow2(
        n_layers=n_layers, hidden_dim=hidden_dim, n_hidden=n_hidden
    ).to(device)

    if verbose:
        n_p = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_p} params  ({n_layers} coupling layers, hidden_dim={hidden_dim})")
        print(f"Training: {num_epochs} epochs  lr={lr}  "
              f"lambda1={lambda1}  lambda2={lambda2}  lambda3={lambda3}  indep={indep_mode}")
        print("=" * 60)

    history = train_jcf2(
        model, phi_t, psi_t,
        num_epochs=num_epochs, lr=lr,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        indep_mode=indep_mode,
        max_hsic_n=max_hsic_n,
        verbose=verbose,
    )

    # --- Decompose ---
    model.eval()
    with torch.no_grad():
        z_phi, z_psi, _ = model(phi_t, psi_t)

        z_phi_c = z_phi - z_phi.mean()
        z_psi_c = z_psi - z_psi.mean()
        A_val   = float((z_phi_c * z_psi_c).mean() /
                        (z_psi_c ** 2).mean().clamp(min=1e-8))

        z_phi_I = A_val * z_psi
        z_phi_R = z_phi - z_phi_I

        # Full joint inverse
        Phi_I_std_flat, _ = model.inverse(z_phi_I, z_psi)
        Phi_I_std_flat    = Phi_I_std_flat.cpu().numpy()

    Phi_I_flat  = Phi_I_std_flat * phi_std + phi_mean
    Phi_R_flat  = phi_flat - Phi_I_flat
    rho_val     = float(model.rho.item())
    mi_estimate = float(-0.5 * np.log(max(1.0 - A_val ** 2, 1e-8)))

    if verbose:
        r_R  = float(np.corrcoef(z_phi_R.cpu().numpy(), z_psi.cpu().numpy())[0, 1])
        r_var = float((z_phi_R.cpu().numpy() ** 2).mean())
        print(f"\nLearned rho = {rho_val:+.4f}")
        print(f"Closed-form A = {A_val:+.4f}")
        print(f"I(Phi; Psi) ~= {mi_estimate:.4f} nats")
        print(f"corr(z_phi^R, z_psi) = {r_R:.4f}  (should be ~0)")
        print(f"Var(z_phi^R) = {r_var:.4f}  (target: 1-A^2 = {1-A_val**2:.4f})")

    # Inference test: Phi-only vs (Phi, Psi) — detect shortcutting through Psi
    infer_results = run_inference_test(
        model, phi_t, psi_t, phi_std, phi_mean,
        Phi_I_flat, psi_flat,
    )
    if verbose:
        print(f"\n--- Inference test (Phi-only vs full) ---")
        print(f"  corr(Phi_I, Psi) with (Phi,Psi) : {infer_results['corr_PhiI_Psi_from_results']:.4f}")
        print(f"  corr(Phi_I, Psi) with Phi-only   : {infer_results['corr_PhiI_Psi_phi_only']:.4f}")
        if infer_results["corr_PhiI_Psi_phi_only"] < 0.5 * abs(infer_results["corr_PhiI_Psi_from_results"]):
            print("  WARNING: Phi-only corr drops sharply -> possible shortcut through Psi")
        else:
            print("  OK: Phi_I remains informative when Psi is withheld")

    return {
        "Phi_I":         Phi_I_flat,
        "Phi_R":         Phi_R_flat,
        "z_phi":         z_phi.cpu().numpy(),
        "z_psi":         z_psi.cpu().numpy(),
        "z_phi_I":       z_phi_I.cpu().numpy(),
        "z_phi_R":       z_phi_R.cpu().numpy(),
        "A":             A_val,
        "rho":           rho_val,
        "mi_estimate":   mi_estimate,
        "loss_history":  history,
        "model":         model,
        "inference_test": infer_results,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_latent_space(z_phi, z_psi, z_phi_I, z_phi_R, A, rho,
                           output_dir="results_jcf2", prefix="aind_jcf2"):
    os.makedirs(output_dir, exist_ok=True)
    N   = z_phi.size
    idx = np.random.choice(N, min(3000, N), replace=False)
    zp  = z_phi[idx];   zpsi = z_psi[idx]
    zI  = z_phi_I[idx]; zR   = z_phi_R[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    data = [
        (zp,  zpsi, f"Joint latent (z\u03a6, z\u03a8)\nrho={rho:+.4f}  A={A:+.4f}", "steelblue"),
        (zI,  zpsi, f"Informative z\u03a6\u1d35 = A\u00b7z\u03a8", "darkorange"),
        (zR,  zpsi, "Residual z\u03a6\u1d3f = z\u03a6 \u2212 A\u00b7z\u03a8\nTarget: cloud", "firebrick"),
    ]
    for ax, (xd, yd, title, col) in zip(axes, data):
        r = float(np.corrcoef(xd, yd)[0, 1])
        ax.scatter(xd, yd, s=4, alpha=0.35, color=col)
        ax.set_title(f"{title}\nPearson r = {r:.4f}", fontsize=9)
        ax.set_xlabel("z\u03a6 component"); ax.set_ylabel("z\u03a8"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_latent.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Latent space saved to: {path}")


def visualize_training(history, output_dir="results_jcf2", prefix="aind_jcf2"):
    os.makedirs(output_dir, exist_ok=True)
    n  = len(history["total"])
    xs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 6, figsize=(30, 4))

    axes[0].plot(xs, history["flow"], color="steelblue", lw=2)
    axes[0].set_title("L_Flow  (bivariate NLL)\n-change-of-variables")
    axes[0].set_xlabel("Checkpoint"); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(xs, [max(v, 1e-12) for v in history["func"]], color="darkorange", lw=2)
    axes[1].set_title("L_Func  ||Z_Psi - Z_I||^2\nfunctional Psi = F(Phi_I)")
    axes[1].set_xlabel("Checkpoint"); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(xs, [max(v, 1e-12) for v in history["indep"]], color="firebrick", lw=2)
    axes[2].set_title("L_Indep  I(Phi_I; Phi_R)\nHSIC or Corr^2")
    axes[2].set_xlabel("Checkpoint"); axes[2].grid(True, alpha=0.3)

    axes[3].semilogy(xs, [max(v, 1e-12) for v in history["smooth"]], color="teal", lw=2)
    axes[3].set_title("L_Smooth  ||Phi - Phi_I||^2\nresidual / anti-collapse")
    axes[3].set_xlabel("Checkpoint"); axes[3].grid(True, alpha=0.3)

    axes[4].plot(xs, history["rho"], color="purple", lw=2, label="\u03c1 (learned)")
    axes[4].plot(xs, history["A"],   color="green",  lw=2, ls="--", label="A (OLS)")
    axes[4].axhline(0, color="k", lw=0.5, ls=":")
    axes[4].set_title("\u03c1 vs A  (should converge together)")
    axes[4].set_xlabel("Checkpoint"); axes[4].grid(True, alpha=0.3); axes[4].legend(fontsize=8)

    axes[5].plot(xs, history["mi_A"],   color="green",  lw=2, label="I\u2248\u22120.5log(1\u2212A\u00b2)")
    axes[5].plot(xs, history["mi_rho"], color="purple", lw=1.5, ls="--", label="I\u2248\u22120.5log(1\u2212\u03c1\u00b2)")
    axes[5].set_title("MI estimates [nats]"); axes[5].set_xlabel("Checkpoint")
    axes[5].grid(True, alpha=0.3); axes[5].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_training.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Training dynamics saved to: {path}")


def visualize_decomposition(Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true, Phi_R_true,
                            output_dir="results_jcf2", prefix="aind_jcf2"):
    os.makedirs(output_dir, exist_ok=True)

    if Phi_I.ndim == 1:
        Phi_I = Phi_I.reshape(Phi.shape)
        Phi_R = Phi_R.reshape(Phi.shape)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = [
        "\u03a6 (Source)\n\u03a6 = f + g",
        "\u03a6\u1d35_true = f\n(True Informative)",
        "\u03a6\u1d35 (Reconstructed)\n[JCF2: biv NLL + HSIC]",
        "\u03a8\u207a (Target)\n\u03a8\u207a = 0.5f\u00b2 \u2212 0.2f + \u03b5",
        "\u03a6\u1d3f_true = g\n(True Residual)",
        "\u03a6\u1d3f (Reconstructed Residual)",
    ]
    fields = [Phi, Phi_I_true, Phi_I, Psi_plus, Phi_R_true, Phi_R]

    # Shared color ranges: Phi^I true & Phi^I recon | Phi^R true & Phi^R recon
    vmin_I, vmax_I = min(Phi_I_true.min(), Phi_I.min()), max(Phi_I_true.max(), Phi_I.max())
    vmin_R, vmax_R = min(Phi_R_true.min(), Phi_R.min()), max(Phi_R_true.max(), Phi_R.max())

    vmin_list = [None, vmin_I, vmin_I, None, vmin_R, vmin_R]
    vmax_list = [None, vmax_I, vmax_I, None, vmax_R, vmax_R]

    for ax, field, title, vmin, vmax in zip(axes.flat, fields, titles, vmin_list, vmax_list):
        kwargs = {"origin": "lower", "cmap": "RdBu", "extent": [0, 1, 0, 2], "aspect": 0.5}
        if vmin is not None:
            kwargs["vmin"], kwargs["vmax"] = vmin, vmax
        im = ax.imshow(field, **kwargs)
        ax.set_title(title, fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fp = os.path.join(output_dir, f"{prefix}_fields.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Field visualization saved to: {fp}")

    # Scatter diagnostics (same layout as aind_decomposition.py)
    Phi_flat = Phi.ravel()
    Psi_flat = Psi_plus.ravel()
    Phi_I_flat = Phi_I.ravel()
    Phi_R_flat = Phi_R.ravel()
    f_flat = Phi_I_true.ravel()

    n_plot = min(5000, len(Phi_flat))
    idx = np.random.choice(len(Phi_flat), n_plot, replace=False)

    plt.figure(figsize=(15, 5))

    # Subplot 1: Mapping Phi^I -> Psi+
    plt.subplot(1, 3, 1)
    plt.scatter(Phi_I_flat[idx], Psi_flat[idx], s=6, alpha=0.5, label="Data")
    phi_I_range = np.linspace(Phi_I_flat.min(), Phi_I_flat.max(), 200)
    psi_analytical = F_analytical(phi_I_range)
    plt.plot(phi_I_range, psi_analytical, "r--", linewidth=2, label="F(\u03a6\u1d35) = 0.5\u03a6\u1d35\u00b2 - 0.2\u03a6\u1d35")
    plt.xlabel("\u03a6\u1d35 (Informative Component)")
    plt.ylabel("\u03a8\u207a (Target Field)")
    plt.title("Mapping: \u03a8\u207a \u2248 F(\u03a6\u1d35)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Independence Phi^I vs Phi^R
    plt.subplot(1, 3, 2)
    plt.scatter(Phi_I_flat[idx], Phi_R_flat[idx], s=6, alpha=0.5, color="red")
    plt.xlabel("\u03a6\u1d35 (Informative Component)")
    plt.ylabel("\u03a6\u1d3f (Residual Component)")
    plt.title("Independence: I(\u03a6\u1d35; \u03a6\u1d3f) should be minimized")
    plt.grid(True, alpha=0.3)

    # Subplot 3: Comparison with true f
    plt.subplot(1, 3, 3)
    plt.scatter(f_flat[idx], Phi_I_flat[idx], s=6, alpha=0.5, color="green")
    f_range = np.linspace(f_flat.min(), f_flat.max(), 100)
    plt.plot(f_range, f_range, "k--", linewidth=2, label="Perfect: \u03a6\u1d35 = f")
    plt.xlabel("f (True Informative Component)")
    plt.ylabel("\u03a6\u1d35 (Reconstructed Informative)")
    plt.title("Reconstruction Quality")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    sp = os.path.join(output_dir, f"{prefix}_scatter.png")
    plt.savefig(sp, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Scatter plots saved to: {sp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    nx:           int   = 100,
    ny:           int   = 100,
    t:            float = 0.0,
    # Architecture
    n_layers:     int   = 6,
    hidden_dim:   int   = 32,
    n_hidden:     int   = 2,
    # Training
    num_epochs:   int   = 1000,
    lr:           float = 5e-3,
    lambda1:     float = 1.0,
    lambda2:     float = 5.0,
    lambda3:     float = 1.0,
    max_hsic_n:   int   = 2000,
    # Output
    output_dir:   str   = "results_jcf2",
    seed:         int   = 1,
    verbose:      bool  = True,
):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    Phi        = phi_func(X, Y, t)
    Psi_plus   = psi_plus_func(X, Y, t, deltaT=0.0, noise_std=1e-8)
    Phi_I_true = f_func(X, Y, t)
    Phi_R_true = g_func(X, Y, t)

    if verbose:
        print("=" * 60)
        print("aIND  --  Joint Coupling Flow v2  (JCF2)")
        print("Fixes: bivariate NLL (rho!=0 signal) + HSIC (nonlinear independence)")
        print("=" * 60)
        print(f"Toy v1: Phi = f + g,  Psi+ = 0.5*f^2 - 0.2*f + eps")
        print(f"Grid: {nx}x{ny}   t={t}")
        print(f"Flow: {n_layers} coupling layers  hidden_dim={hidden_dim}")
        print(f"Epochs={num_epochs}  lr={lr}  "
              f"lambda1={lambda1}  lambda2={lambda2}  lambda3={lambda3}")
        mi_gt = mutual_info_regression(
            Phi_I_true.ravel().reshape(-1, 1), Phi_R_true.ravel(), random_state=42)[0]
        print(f"\nGround truth I(f; g) = {mi_gt:.6f}  (target: ~0)")
        print()

    results = aind_decomposition_jcf2(
        Phi, Psi_plus,
        n_layers=n_layers, hidden_dim=hidden_dim, n_hidden=n_hidden,
        num_epochs=num_epochs, lr=lr,
        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
        max_hsic_n=max_hsic_n,
        verbose=verbose, seed=seed,
    )

    Phi_I = results["Phi_I"].reshape(Phi.shape)
    Phi_R = results["Phi_R"].reshape(Phi.shape)

    metrics = evaluate_decomposition(
        Phi.ravel(), Psi_plus.ravel(),
        results["Phi_I"], results["Phi_R"],
        Phi_I_true=Phi_I_true.ravel(),
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Final Evaluation Metrics  (JCF2)")
        print("=" * 60)
        print(f"  Learned rho                        : {results['rho']:+.6f}")
        print(f"  Closed-form A                      : {results['A']:+.6f}")
        print(f"  I(Phi;Psi) ~= -0.5*log(1-A^2)     : {results['mi_estimate']:.6f} nats")
        print(f"  I(Phi_I ; Phi_R)                   : {metrics['mutual_information']:.6f}")
        print(f"  Residual energy ||Phi - Phi_I||^2  : {metrics['residual_energy']:.6f}")
        print(f"  MI(Phi_I , Psi+)                   : {metrics['mi_phiI_psi']:.6f}")
        print(f"  MI(Phi_R , Psi+)                   : {metrics['mi_phiR_psi']:.6f}")
        print(f"  GT error ||Phi_I - Phi_I_true||^2  : {metrics['gt_error']:.6f}")
        print("=" * 60)

    path = os.path.join(output_dir, "aind_jcf2_metrics.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("aIND Metrics  (JCF2: bivariate NLL + HSIC)\n")
        fh.write(f"n_layers={n_layers} | hidden_dim={hidden_dim} | epochs={num_epochs}\n")
        fh.write(f"rho={results['rho']:+.6f}  A={results['A']:+.6f}  "
                 f"I~={results['mi_estimate']:.6f} nats\n")
        fh.write("=" * 60 + "\n")
        for name, key in [
            ("I(Phi_I ; Phi_R)",              "mutual_information"),
            ("Residual energy",               "residual_energy"),
            ("MI(Phi_I, Psi+)",               "mi_phiI_psi"),
            ("MI(Phi_R, Psi+)",               "mi_phiR_psi"),
            ("GT error ||Phi_I-Phi_I_true||", "gt_error"),
        ]:
            fh.write(f"{name:<35}: {metrics[key]:.6f}\n")
    print(f"Metrics saved to: {path}")

    visualize_latent_space(
        results["z_phi"], results["z_psi"],
        results["z_phi_I"], results["z_phi_R"],
        A=results["A"], rho=results["rho"],
        output_dir=output_dir, prefix="aind_jcf2",
    )
    visualize_training(results["loss_history"], output_dir=output_dir, prefix="aind_jcf2")
    visualize_decomposition(
        Phi, Psi_plus, Phi_I, Phi_R,
        Phi_I_true, Phi_R_true,
        output_dir=output_dir, prefix="aind_jcf2",
    )

    return results, metrics


if __name__ == "__main__":
    main(
        nx=100,
        ny=100,
        t=0.0,
        n_layers=6,
        hidden_dim=32,
        n_hidden=2,
        num_epochs=2000,
        lr=3e-3,
        lambda1=5.0,
        lambda2=150.0,
        lambda3=0.50,
        max_hsic_n=2000,
        output_dir="results_jcf2",
        seed=1,
        verbose=True,
    )
