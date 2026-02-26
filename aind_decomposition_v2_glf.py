"""
aIND in Gaussian Latent Space  (Toy v2)

Architecture
------------

  Physical space              Latent space (all ~ N(0,1))         Physical space
  Phi  --[f_phi]--> z_phi --+-----------------------------+--> z_phi^I = g_theta(z_psi)
                             |                             |     Phi^I = f_phi^{-1}(z_phi^I)
  Psi+ --[f_psi]--> z_psi --+-> g_theta(z_psi) = z_phi^I |
                                 z_phi^R = z_phi - z_phi^I +-->  Phi^R = Phi - Phi^I

"""
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from toy_settings_v2 import (
#     phi_func,
#     phi_informative_true,
#     phi_residual_true,
#     psi_plus_func,
# )
from toy_settings import (
    phi_func,
    psi_plus_func,
    f_func,
    g_func,
    F_analytical,
)

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error


def evaluate_decomposition(Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true=None):
    """
    Evaluate aIND decompositionF
    -------
    mutual_information : I(Phi_I; Phi_R) via sklearn  (lower = more independent)
    residual_energy    : MSE(Phi, Phi_I)              (lower = smaller residual)
    mi_phiI_psi        : MI(Phi_I, Psi+)              (higher = informative)
    mi_phiR_psi        : MI(Phi_R, Psi+)              (lower  = non-informative)
    gt_error           : MSE(Phi_I, Phi_I_true)       (lower = closer to ground truth)
    """
    metrics = {}
    metrics["mutual_information"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Phi_I, random_state=42
    )[0]
    metrics["residual_energy"] = mean_squared_error(Phi, Phi_I)
    metrics["mi_phiI_psi"] = mutual_info_regression(
        Phi_I.reshape(-1, 1), Psi_plus, random_state=42
    )[0]
    metrics["mi_phiR_psi"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Psi_plus, random_state=42
    )[0]
    metrics["gt_error"] = (
        mean_squared_error(Phi_I_true, Phi_I) if Phi_I_true is not None else float("nan")
    )
    metrics["total_loss"] = metrics["mutual_information"] + metrics["residual_energy"]
    return metrics


# ---------------------------------------------------------------------------
# Monotone Normalizing Flow  (1D element-wise, reused from v2_nf design)
# ---------------------------------------------------------------------------

class MonotoneFlow(nn.Module):
    """
    Learnable strictly-monotone 1D normalizing flow.

    Forward:  x  -->  z = (raw(x) - shift) / exp(log_scale)
    raw(x)  = softplus(slope) * x
            + sum_k softplus(w_k) * tanh(softplus(a_k) * x + b_k)

    Every term in d(raw)/dx is strictly positive -> raw is monotone.
    Log-Jacobian:  log|dz/dx| = log|d(raw)/dx| - log_scale.
    Inverse via 60-step bisection (error < 20 / 2^60 ~ 2e-17).
    """

    def __init__(self, n_terms: int = 32):
        super().__init__()
        self.log_slope = nn.Parameter(torch.tensor(0.0))
        self.log_w     = nn.Parameter(torch.zeros(n_terms) - 1.0)
        self.log_a     = nn.Parameter(torch.zeros(n_terms))
        self.b         = nn.Parameter(torch.linspace(-3.0, 3.0, n_terms))
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.shift     = nn.Parameter(torch.tensor(0.0))

    def _raw(self, x: torch.Tensor) -> torch.Tensor:
        slope = F.softplus(self.log_slope)
        w = F.softplus(self.log_w)
        a = F.softplus(self.log_a)
        h = torch.tanh(a[None, :] * x[:, None] + self.b[None, :])
        return slope * x + (w[None, :] * h).sum(-1)

    def _log_d_raw(self, x: torch.Tensor) -> torch.Tensor:
        slope = F.softplus(self.log_slope)
        w = F.softplus(self.log_w)
        a = F.softplus(self.log_a)
        h     = torch.tanh(a[None, :] * x[:, None] + self.b[None, :])
        dtanh = (1.0 - h ** 2) * a[None, :]
        df    = slope + (w[None, :] * dtanh).sum(-1)
        return torch.log(df.clamp(min=1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self._raw(x) - self.shift) / (self.log_scale.exp() + 1e-8)

    def log_abs_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return self._log_d_raw(x) - self.log_scale

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, n_iter: int = 60) -> torch.Tensor:
        z_raw = z * (self.log_scale.exp() + 1e-8) + self.shift
        lo = torch.full_like(z_raw, -30.0)
        hi = torch.full_like(z_raw, +30.0)
        for _ in range(n_iter):
            mid   = 0.5 * (lo + hi)
            f_mid = self._raw(mid.ravel()).reshape(mid.shape)
            lo = torch.where(f_mid <  z_raw, mid, lo)
            hi = torch.where(f_mid >= z_raw, mid, hi)
        return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Informative extraction module  g_theta : R -> R   (scalar latent, 1D case)
# ---------------------------------------------------------------------------

class InformativeExtractor(nn.Module):
    """
    g_theta: z_psi -> z_phi^I

    A shallow MLP that predicts the informative latent component z_phi^I
    from the target latent code z_psi.  In the scalar case (m=1) this is
    a nonlinear function  R -> R.

    Optional gating (Section 1.2 of the design note):
        z_phi^I = alpha(z_psi) * g(z_psi),  alpha in [0,1]
    controlled by `use_gating=True`.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers:   int = 3,
        use_gating: bool = False,
    ):
        super().__init__()
        self.use_gating = use_gating

        def _mlp() -> nn.Sequential:
            layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.net = _mlp()
        if use_gating:
            self.gate = _mlp()   # outputs logit; sigmoid gives alpha in [0,1]

    def forward(self, z_psi: torch.Tensor) -> torch.Tensor:
        """z_psi: (N,)  ->  z_phi_I: (N,)"""
        x = z_psi.unsqueeze(-1)          # (N,1)
        out = self.net(x).squeeze(-1)    # (N,)
        if self.use_gating:
            alpha = torch.sigmoid(self.gate(x).squeeze(-1))
            out   = alpha * out
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GaussianLatentAIND(nn.Module):
    """
    End-to-end aIND model in Gaussian latent space.

    Components
    ----------
    f_phi   : MonotoneFlow  Phi  -> z_phi  ~ N(0,1)
    f_psi   : MonotoneFlow  Psi+ -> z_psi  ~ N(0,1)
    g_theta : InformativeExtractor  z_psi  -> z_phi^I

    forward() returns the latent tensors and log-Jacobians needed for the loss.
    """

    def __init__(
        self,
        n_flow_terms: int  = 32,
        hidden_dim:   int  = 64,
        n_layers:     int  = 3,
        use_gating:   bool = False,
    ):
        super().__init__()
        self.f_phi   = MonotoneFlow(n_terms=n_flow_terms)
        self.f_psi   = MonotoneFlow(n_terms=n_flow_terms)
        self.g_theta = InformativeExtractor(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            use_gating=use_gating,
        )

    def forward(self, phi_flat: torch.Tensor, psi_flat: torch.Tensor):
        z_phi = self.f_phi(phi_flat)
        z_psi = self.f_psi(psi_flat)

        z_phi_I = self.g_theta(z_psi)
        z_phi_R = z_phi - z_phi_I

        ldj_phi = self.f_phi.log_abs_jacobian(phi_flat)
        ldj_psi = self.f_psi.log_abs_jacobian(psi_flat)

        return z_phi, z_psi, z_phi_I, z_phi_R, ldj_phi, ldj_psi


# ---------------------------------------------------------------------------
# HSIC independence criterion  (kernel-based, differentiable)
# ---------------------------------------------------------------------------

def _hsic_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x: float,
    sigma_y: float,
) -> torch.Tensor:
    """
    Hilbert-Schmidt Independence Criterion with RBF kernels.

        HSIC(X, Y) = (1/(n-1)^2) * tr(K_X_c  K_Y_c)

    where K_c = H K H,  H = I - (1/n) 11^T is the centering matrix.
    HSIC = 0 iff X and Y are independent (under characteristic kernels).

    X, Y: 1D tensors (N,).  O(N^2) computation -- use subsampling for large N.
    """
    n = X.shape[0]
    Xv = X.view(-1, 1)
    Yv = Y.view(-1, 1)

    K = torch.exp(-torch.cdist(Xv, Xv) ** 2 / (2.0 * sigma_x ** 2))
    L = torch.exp(-torch.cdist(Yv, Yv) ** 2 / (2.0 * sigma_y ** 2))

    # Efficient centering: K_c = K - row_mean - col_mean + grand_mean
    K_row   = K.mean(dim=1, keepdim=True)
    K_col   = K.mean(dim=0, keepdim=True)
    K_grand = K.mean()
    Kc = K - K_row - K_col + K_grand

    L_row   = L.mean(dim=1, keepdim=True)
    L_col   = L.mean(dim=0, keepdim=True)
    L_grand = L.mean()
    Lc = L - L_row - L_col + L_grand

    return torch.trace(Kc @ Lc) / float((n - 1) ** 2)


def hsic_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x: float        = 1.0,
    sigma_y: float        = 1.0,
    adaptive_sigma: bool  = True,
    max_samples: int      = 2000,
) -> torch.Tensor:
    """
    HSIC with optional median-heuristic bandwidths and mini-batch subsampling.

    For N > max_samples, draws a random subset of size max_samples to keep
    the O(N^2) kernel computation tractable.
    """
    N = X.shape[0]
    if N > max_samples:
        idx = torch.randperm(N, device=X.device)[:max_samples]
        X, Y = X[idx], Y[idx]

    if adaptive_sigma:
        with torch.no_grad():
            sigma_x = float(torch.cdist(X.view(-1, 1), X.view(-1, 1)).median()) + 1e-8
            sigma_y = float(torch.cdist(Y.view(-1, 1), Y.view(-1, 1)).median()) + 1e-8

    return _hsic_rbf(X, Y, sigma_x, sigma_y)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class GaussianLatentLoss:
    """
    Total loss (Section 3 of the design note):

        L = L_flow(f_phi) + L_flow(f_psi)
          + lambda_pred * E[||z_phi - g_theta(z_psi)||^2]
          + lambda_ind  * HSIC(z_phi^R, z_psi)
          + lambda_min  * E[||g_theta(z_psi)||_1]

    Each flow loss is the NF negative log-likelihood under N(0,1):
        L_flow(f) = 0.5 * E[z^2] - E[log|dz/dx|]
    """

    def __init__(
        self,
        lambda_pred:    float = 1.0,
        lambda_ind:     float = 1.0,
        lambda_min:     float = 0.0,
        hsic_sigma:     float = 1.0,
        adaptive_sigma: bool  = True,
        hsic_max_n:     int   = 2000,
    ):
        self.lambda_pred    = lambda_pred
        self.lambda_ind     = lambda_ind
        self.lambda_min     = lambda_min
        self.hsic_sigma     = hsic_sigma
        self.adaptive_sigma = adaptive_sigma
        self.hsic_max_n     = hsic_max_n

    def __call__(
        self,
        z_phi, z_psi, z_phi_I, z_phi_R,
        ldj_phi, ldj_psi,
    ):
        # --- Flow NLL losses ---
        L_flow_phi = 0.5 * (z_phi ** 2).mean() - ldj_phi.mean()
        L_flow_psi = 0.5 * (z_psi ** 2).mean() - ldj_psi.mean()
        L_flow     = L_flow_phi + L_flow_psi

        # --- Predictive loss: z_phi^I should track z_phi ---
        L_pred = ((z_phi - z_phi_I) ** 2).mean()

        # --- HSIC independence loss: z_phi^R _|_ z_psi ---
        L_ind = hsic_rbf(
            z_phi_R, z_psi,
            sigma_x=self.hsic_sigma,
            sigma_y=self.hsic_sigma,
            adaptive_sigma=self.adaptive_sigma,
            max_samples=self.hsic_max_n,
        )

        # --- Minimality: L1 sparsity on z_phi^I ---
        L_min = z_phi_I.abs().mean()

        total = (
            L_flow
            + self.lambda_pred * L_pred
            + self.lambda_ind  * L_ind
            + self.lambda_min  * L_min
        )

        breakdown = {
            "total":    total.item(),
            "flow":     L_flow.item(),
            "flow_phi": L_flow_phi.item(),
            "flow_psi": L_flow_psi.item(),
            "pred":     L_pred.item(),
            "ind":      L_ind.item(),
            "min":      L_min.item(),
        }
        return total, breakdown


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(
    model:          GaussianLatentAIND,
    phi_t:          torch.Tensor,
    psi_t:          torch.Tensor,
    criterion:      GaussianLatentLoss,
    num_epochs:     int   = 2000,
    lr:             float = 1e-3,
    grad_clip:      float = 5.0,
    verbose:        bool  = True,
    device=None,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    log_every = max(1, num_epochs // 10)
    history   = {"total": [], "flow": [], "pred": [], "ind": [], "min": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        z_phi, z_psi, z_phi_I, z_phi_R, ldj_phi, ldj_psi = model(phi_t, psi_t)
        loss, bd = criterion(z_phi, z_psi, z_phi_I, z_phi_R, ldj_phi, ldj_psi)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == num_epochs:
            history["total"].append(bd["total"])
            history["flow"].append(bd["flow"])
            history["pred"].append(bd["pred"])
            history["ind"].append(bd["ind"])
            history["min"].append(bd["min"])

            if verbose:
                r_I = float(np.corrcoef(
                    z_phi_I.detach().cpu().numpy(),
                    z_psi.detach().cpu().numpy(),
                )[0, 1])
                r_R = float(np.corrcoef(
                    z_phi_R.detach().cpu().numpy(),
                    z_psi.detach().cpu().numpy(),
                )[0, 1])
                print(
                    f"  [Epoch {epoch:5d}]  "
                    f"total={bd['total']:+.4f}  "
                    f"flow={bd['flow']:.4f}  "
                    f"pred={bd['pred']:.4f}  "
                    f"HSIC={bd['ind']:.6f}  "
                    f"r(z_I,z_psi)={r_I:+.3f}  "
                    f"r(z_R,z_psi)={r_R:+.3f}"
                )

    return history


# ---------------------------------------------------------------------------
# Full aIND pipeline
# ---------------------------------------------------------------------------

def aind_decomposition_glf(
    Phi,
    Psi_plus,
    # Flow architecture
    n_flow_terms:   int   = 32,
    hidden_dim:     int   = 64,
    n_layers:       int   = 3,
    use_gating:     bool  = False,
    # Optimisation
    num_epochs:     int   = 2000,
    lr:             float = 1e-3,
    # Loss weights
    lambda_pred:    float = 1.0,
    lambda_ind:     float = 1.0,
    lambda_min:     float = 0.0,
    # HSIC options
    hsic_sigma:     float = 1.0,
    adaptive_sigma: bool  = True,
    hsic_max_n:     int   = 2000,
    # Misc
    verbose:        bool  = True,
    seed:           int   = 42,
    device=None,
):
    """
    aIND decomposition via Gaussian latent flows.

    Steps
    -----
    1.  Pre-standardize Phi and Psi+ to zero mean / unit variance.
    2.  Train f_phi, f_psi, g_theta jointly end-to-end.
    3.  Extract  z_phi^I = g_theta(z_psi),  z_phi^R = z_phi - z_phi^I.
    4.  Decode:  Phi^I = f_phi^{-1}(z_phi^I)  (bisection inverse).
    5.  Compute  Phi^R = Phi - Phi^I.

    Returns
    -------
    dict with: Phi_I, Phi_R, z_phi, z_psi, z_phi_I, z_phi_R,
               loss_history, model
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    Phi_flat = Phi.ravel()
    Psi_flat = Psi_plus.ravel()

    # Pre-standardize to give flows a well-scaled input
    phi_mean, phi_std = Phi_flat.mean(), Phi_flat.std() + 1e-8
    psi_mean, psi_std = Psi_flat.mean(), Psi_flat.std() + 1e-8
    Phi_std = (Phi_flat - phi_mean) / phi_std
    Psi_std = (Psi_flat - psi_mean) / psi_std

    phi_t = torch.tensor(Phi_std, dtype=torch.float32, device=device)
    psi_t = torch.tensor(Psi_std, dtype=torch.float32, device=device)

    model = GaussianLatentAIND(
        n_flow_terms=n_flow_terms,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_gating=use_gating,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params} parameters")
        print(f"  f_phi / f_psi : MonotoneFlow ({n_flow_terms} terms each)")
        print(f"  g_theta       : {n_layers}-layer MLP (dim {hidden_dim})"
              + ("  + gating" if use_gating else ""))
        print(f"Training: epochs={num_epochs}  lr={lr}")
        print(f"  lambda_pred={lambda_pred}  lambda_ind={lambda_ind}  lambda_min={lambda_min}")
        print("=" * 60)

    criterion = GaussianLatentLoss(
        lambda_pred=lambda_pred,
        lambda_ind=lambda_ind,
        lambda_min=lambda_min,
        hsic_sigma=hsic_sigma,
        adaptive_sigma=adaptive_sigma,
        hsic_max_n=hsic_max_n,
    )

    loss_history = _train(
        model, phi_t, psi_t, criterion,
        num_epochs=num_epochs,
        lr=lr,
        verbose=verbose,
        device=device,
    )

    # --- Decode ---
    model.eval()
    with torch.no_grad():
        z_phi, z_psi, z_phi_I, z_phi_R, _, _ = model(phi_t, psi_t)

        # Invert flow: z_phi^I  ->  standardized Phi space
        Phi_I_std_flat = model.f_phi.inverse(z_phi_I).cpu().numpy()

    # Undo pre-standardization
    Phi_I_flat = Phi_I_std_flat * phi_std + phi_mean
    Phi_R_flat = Phi_flat - Phi_I_flat

    if verbose:
        r_I  = float(np.corrcoef(z_phi_I.cpu().numpy(), z_psi.cpu().numpy())[0, 1])
        r_R  = float(np.corrcoef(z_phi_R.cpu().numpy(), z_psi.cpu().numpy())[0, 1])
        print(f"\nFinal latent correlations:")
        print(f"  corr(z_phi^I, z_psi) = {r_I:.4f}   (higher = more informative)")
        print(f"  corr(z_phi^R, z_psi) = {r_R:.4f}   (lower  = more independent)")

    return {
        "Phi_I":        Phi_I_flat,
        "Phi_R":        Phi_R_flat,
        "z_phi":        z_phi.cpu().numpy(),
        "z_psi":        z_psi.cpu().numpy(),
        "z_phi_I":      z_phi_I.cpu().numpy(),
        "z_phi_R":      z_phi_R.cpu().numpy(),
        "loss_history": loss_history,
        "model":        model,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_latent_space(
    z_phi, z_psi, z_phi_I, z_phi_R,
    output_dir: str = "results_v2_glf",
    prefix:     str = "aind_v2_glf",
):
    """
    Three scatter plots showing the joint distribution at each stage:
        (z_phi, z_psi)   — both flows: should look like bivariate N(0,1)
        (z_phi^I, z_psi) — informative component: should be correlated
        (z_phi^R, z_psi) — residual: should be independent (scatter cloud)
    """
    os.makedirs(output_dir, exist_ok=True)

    N   = z_phi.size
    idx = np.random.choice(N, min(3000, N), replace=False)
    zp  = z_phi[idx];   zpsi = z_psi[idx]
    zI  = z_phi_I[idx]; zR   = z_phi_R[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    data = [
        (zp,  zpsi, "Both flows: (z\u03a6, z\u03a8)\nTarget: bivariate N(0,1)",           "steelblue"),
        (zI,  zpsi, "Informative: (z\u03a6\u1d35, z\u03a8)\nShould be correlated",        "darkorange"),
        (zR,  zpsi, "Residual: (z\u03a6\u1d3f, z\u03a8)\nTarget: independent",            "firebrick"),
    ]
    for ax, (xd, yd, title, col) in zip(axes, data):
        r = float(np.corrcoef(xd, yd)[0, 1])
        ax.scatter(xd, yd, s=4, alpha=0.35, color=col)
        ax.set_title(f"{title}\nPearson r = {r:.4f}", fontsize=9)
        ax.set_xlabel("z\u03a6 component")
        ax.set_ylabel("z\u03a8 = f\u03a8(\u03a8\u207a)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_latent.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Latent space visualization saved to: {path}")


def visualize_training(
    loss_history: dict,
    output_dir:   str = "results_v2_glf",
    prefix:       str = "aind_v2_glf",
):
    """Training dynamics: loss components and HSIC over log-checkpoints."""
    os.makedirs(output_dir, exist_ok=True)

    n  = len(loss_history["total"])
    xs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(xs, loss_history["total"], label="Total",      linewidth=2)
    axes[0].plot(xs, loss_history["flow"],  label="Flow NLL",   linestyle="--")
    axes[0].plot(xs, loss_history["pred"],  label="Pred (MSE)", linestyle="--")
    axes[0].set_xlabel("Log checkpoint")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(xs, [max(v, 1e-12) for v in loss_history["ind"]], color="firebrick")
    axes[1].set_xlabel("Log checkpoint")
    axes[1].set_ylabel("HSIC(z\u03a6\u1d3f, z\u03a8)")
    axes[1].set_title("Independence loss (HSIC)\n(should decrease)")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(xs, [max(v, 1e-12) for v in loss_history["pred"]], color="darkorange")
    axes[2].set_xlabel("Log checkpoint")
    axes[2].set_ylabel("||z\u03a6 \u2212 g\u03b8(z\u03a8)||^2")
    axes[2].set_title("Predictive loss\n(should decrease)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_training.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training dynamics saved to: {path}")


def visualize_decomposition(
    Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true, Phi_R_true,
    output_dir: str = "results_v2_glf",
    prefix:     str = "aind_v2_glf",
):
    """2 x 3 field comparison with individual colorbars per subplot, plus scatter diagnostics."""
    os.makedirs(output_dir, exist_ok=True)

    if Phi_I.ndim == 1:
        Phi_I = Phi_I.reshape(Phi.shape)
        Phi_R = Phi_R.reshape(Phi.shape)

    # --- Field comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = [
        "\u03a6 (Source Field)\n\u03a6 = c\u00b2+cf+cg+fg",
        "\u03a6\u1d35_true = f(c+g)\n(True Informative)",
        "\u03a6\u1d35 (Reconstructed Informative)",
        "\u03a8\u207a (Target Field)\n\u03a8\u207a = 0.5f\u00b2 \u2212 0.2f + \u03b5",
        "\u03a6\u1d3f_true = c(c+g)\n(True Residual)",
        "\u03a6\u1d3f (Reconstructed Residual)\n\u03a6\u1d3f = \u03a6 \u2212 \u03a6\u1d35",
    ]
    fields = [Phi, Phi_I_true, Phi_I, Psi_plus, Phi_R_true, Phi_R]

    for ax, field, title in zip(axes.flat, fields, titles):
        im = ax.imshow(field, origin="lower", cmap="RdBu",
                       extent=[0, 1, 0, 2], aspect=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fields_path = os.path.join(output_dir, f"{prefix}_fields.png")
    fig.savefig(fields_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Field visualization saved to: {fields_path}")

    # --- Scatter diagnostics ---
    Phi_flat        = Phi.ravel()
    Psi_flat        = Psi_plus.ravel()
    Phi_I_flat      = Phi_I.ravel()
    Phi_R_flat      = Phi_R.ravel()
    Phi_I_true_flat = Phi_I_true.ravel()

    n_plot = min(5000, len(Phi_flat))
    idx    = np.random.choice(len(Phi_flat), n_plot, replace=False)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(Phi_I_flat[idx], Psi_flat[idx], s=6, alpha=0.5, label="Data")
    plt.xlabel("\u03a6\u1d35 (Informative Component)")
    plt.ylabel("\u03a8\u207a (Target Field)")
    plt.title("Mapping: \u03a8\u207a \u2248 F(\u03a6\u1d35)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.scatter(Phi_I_flat[idx], Phi_R_flat[idx], s=6, alpha=0.5, color="red")
    plt.xlabel("\u03a6\u1d35 (Informative Component)")
    plt.ylabel("\u03a6\u1d3f (Residual Component)")
    plt.title("Independence: I(\u03a6\u1d35; \u03a6\u1d3f) should be minimized")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.scatter(Phi_I_true_flat[idx], Phi_I_flat[idx], s=6, alpha=0.5, color="green")
    rng = np.linspace(Phi_I_true_flat.min(), Phi_I_true_flat.max(), 100)
    plt.plot(rng, rng, "k--", linewidth=2,
             label="Perfect: \u03a6\u1d35 = \u03a6\u1d35_true")
    plt.xlabel("\u03a6\u1d35_true = f(c+g)")
    plt.ylabel("\u03a6\u1d35 (Reconstructed Informative)")
    plt.title("Reconstruction Quality")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"{prefix}_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plots saved to: {scatter_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    nx:             int   = 100,
    ny:             int   = 100,
    t:              float = 0.0,
    # Architecture
    n_flow_terms:   int   = 32,
    hidden_dim:     int   = 64,
    n_layers:       int   = 3,
    use_gating:     bool  = False,
    # Training
    num_epochs:     int   = 1000,
    lr:             float = 5e-3,
    # Loss weights
    lambda_pred:    float = 1.0,
    lambda_ind:     float = 5.0,
    lambda_min:     float = 0.0,
    # HSIC
    hsic_sigma:     float = 1.0,
    adaptive_sigma: bool  = True,
    hsic_max_n:     int   = 2000,
    # Output
    output_dir:     str   = "results_v2_glf",
    seed:           int   = 1,
    verbose:        bool  = True,
):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Build grid and fields
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    Phi        = phi_func(X, Y, t)
    Psi_plus   = psi_plus_func(X, Y, t, deltaT=0.0, noise_std=1e-8)
    # Phi_I_true = phi_informative_true(X, Y, t)
    # Phi_R_true = phi_residual_true(X, Y, t)
    Phi_I_true = f_func(X, Y, t)
    Phi_R_true = g_func(X, Y, t)

    if verbose:
        print("=" * 60)
        print("aIND in Gaussian Latent Space  --  Toy v2")
        print("=" * 60)
        print(f"Grid: {nx} x {ny}   t={t}")
        print(f"Flows: {n_flow_terms} terms | g_theta: {n_layers} x {hidden_dim}"
              + ("  + gating" if use_gating else ""))
        print(f"Epochs={num_epochs} | lr={lr} | seed={seed}")
        print(f"lambda_pred={lambda_pred} | lambda_ind={lambda_ind} | lambda_min={lambda_min}")
        print()

    results = aind_decomposition_glf(
        Phi, Psi_plus,
        n_flow_terms=n_flow_terms,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_gating=use_gating,
        num_epochs=num_epochs,
        lr=lr,
        lambda_pred=lambda_pred,
        lambda_ind=lambda_ind,
        lambda_min=lambda_min,
        hsic_sigma=hsic_sigma,
        adaptive_sigma=adaptive_sigma,
        hsic_max_n=hsic_max_n,
        verbose=verbose,
        seed=seed,
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
        print("Final Evaluation Metrics  (Gaussian Latent Flow aIND)")
        print("=" * 60)
        print(f"  I(Phi_I ; Phi_R)                   : {metrics['mutual_information']:.6f}  (lower = more independent)")
        print(f"  Residual energy ||Phi - Phi_I||^2  : {metrics['residual_energy']:.6f}")
        print(f"  MI(Phi_I , Psi+)                   : {metrics['mi_phiI_psi']:.6f}  (higher = informative)")
        print(f"  MI(Phi_R , Psi+)                   : {metrics['mi_phiR_psi']:.6f}  (lower = non-informative)")
        print(f"  GT error ||Phi_I - Phi_I_true||^2  : {metrics['gt_error']:.6f}")
        print(f"  Total loss                         : {metrics['total_loss']:.6f}")
        print("=" * 60)

    # Save metrics
    metrics_path = os.path.join(output_dir, "aind_v2_glf_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write("aIND Metrics  (Gaussian Latent Flow, Toy v2)\n")
        fh.write(f"Flows: {n_flow_terms} terms | g_theta: {n_layers} x {hidden_dim}"
                 + ("  + gating\n" if use_gating else "\n"))
        fh.write(f"lambda_pred={lambda_pred} | lambda_ind={lambda_ind} | lambda_min={lambda_min}\n")
        fh.write("=" * 60 + "\n")
        for name, key in [
            ("I(Phi_I ; Phi_R)",              "mutual_information"),
            ("Residual energy",               "residual_energy"),
            ("MI(Phi_I, Psi+)",               "mi_phiI_psi"),
            ("MI(Phi_R, Psi+)",               "mi_phiR_psi"),
            ("GT error ||Phi_I-Phi_I_true||", "gt_error"),
            ("Total loss",                    "total_loss"),
        ]:
            fh.write(f"{name:<35}: {metrics[key]:.6f}\n")
    print(f"Metrics saved to: {metrics_path}")

    visualize_latent_space(
        results["z_phi"], results["z_psi"],
        results["z_phi_I"], results["z_phi_R"],
        output_dir=output_dir, prefix="aind_v2_glf",
    )
    visualize_training(
        results["loss_history"],
        output_dir=output_dir, prefix="aind_v2_glf",
    )
    visualize_decomposition(
        Phi, Psi_plus, Phi_I, Phi_R,
        Phi_I_true, Phi_R_true,
        output_dir=output_dir, prefix="aind_v2_glf",
    )

    return results, metrics


if __name__ == "__main__":
    main(
        nx=100,
        ny=100,
        t=0.0,
        n_flow_terms=32,
        hidden_dim=64,
        n_layers=3,
        use_gating=False,
        num_epochs=2000,
        lr=7e-3,
        lambda_pred=1.0,
        lambda_ind=5,
        lambda_min=0.0,
        adaptive_sigma=True,
        hsic_max_n=2000,
        output_dir="results_v2_glf",
        seed=1,
        verbose=True,
    )
