"""
aIND via Joint Coupling Flow  (JCF)

Architecture overview
---------------------
Train ONE invertible joint flow on concatenated pairs (Phi_i, Psi_i):

    z = f_theta(Phi, Psi) = (z_phi, z_psi)  ~  N(0, I)      [joint NLL loss]

Implemented as K alternating affine coupling layers:
    Even layers:  psi'  = psi  * exp(s_psi(phi))  + t_psi(phi)
                  phi'  = phi  * exp(s_phi(psi'))  + t_phi(psi')
    Odd  layers:  phi'  = phi  * exp(s_phi(psi))   + t_phi(psi)
                  psi'  = psi  * exp(s_psi(phi'))  + t_psi(phi')

The t functions in each coupling layer learn to encode conditional structure:
    t_psi(phi)  ->  E[psi | phi]   (approximately)
    t_phi(psi') ->  E[phi | psi]   (approximately)

So even when the latent is perfectly decorrelated (A -> 0), the INVERSE
of the flow still recovers the informative component through the t functions.

Decomposition
-------------
Closed-form OLS in latent (differentiable, recomputed per batch):
    A       = Sigma_hat_phi_psi * Sigma_hat_psi_psi^{-1}

    z_phi^I = A * z_psi
    z_phi^R = z_phi - z_phi^I

Decode back to physical space using the FULL joint inverse:
    Phi^I = pi_phi( f_theta^{-1}(z_phi^I, z_psi) )
    Phi^R = Phi - Phi^I

Because the coupling layers store the conditional mean in their t-parameters,
the decode step f^{-1}(A*z_psi, z_psi) recovers E[Phi | Psi] even when A~=0.
This is the key advantage over the diagonal-flow approaches (jgf.py, cca.py).

Loss
----
    L = L_flow  +  lambda_perp  * L_perp  +  lambda_white * L_white

    L_flow  = 0.5*(z_phi^2 + z_psi^2) - log|det J|       [N(0,I) NLL]
    A       = Cov(z_phi, z_psi) / Var(z_psi)              [closed-form OLS]
    r       = z_phi - A * z_psi                           [residual]
    L_perp  = Cov(r, z_psi)^2                             [orthogonality; ~0 by OLS]
    L_white = (Var(r) - (1 - A^2))^2                      [conditional variance match]

L_perp  : algebraically zero when A is closed-form OLS; helps finite-batch numerics.
L_white : genuinely non-trivial — penalises deviation of residual variance from
          the Gaussian conditional variance  sigma^2 = 1 - A^2.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error

from toy_settings import (
    phi_func,
    psi_plus_func,
    f_func,
    g_func,
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
# Coupling network: R -> R^2  (shared helper)
# ---------------------------------------------------------------------------

def _make_st_net(hidden_dim: int, n_hidden: int) -> nn.Sequential:
    """
    Scalar -> (s, t): scale-translate network for affine coupling.
    Initialised to near-identity (s=0, t=0) so training starts stable.
    """
    layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, 2))
    net = nn.Sequential(*layers)
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


# ---------------------------------------------------------------------------
# Joint Coupling Layer
# ---------------------------------------------------------------------------

class JointCouplingLayer(nn.Module):
    """
    One coupling block for (phi_i, psi_i) scalar pairs.

    reverse=False  (even layers):
        psi' = psi * exp(s_psi(phi))  + t_psi(phi)    [transform psi | phi]
        phi' = phi * exp(s_phi(psi')) + t_phi(psi')   [transform phi | new psi]

    reverse=True   (odd layers):
        phi' = phi * exp(s_phi(psi))  + t_phi(psi)    [transform phi | psi]
        psi' = psi * exp(s_psi(phi')) + t_psi(phi')   [transform psi | new phi]

    Alternating directions gives full bidirectional expressiveness.
    Scale s is tanh-squashed to (-2, 2) → exp(s) in (e^{-2}, e^{2}) for stability.
    """

    def __init__(
        self,
        hidden_dim: int  = 32,
        n_hidden:   int  = 2,
        reverse:    bool = False,
    ):
        super().__init__()
        self.reverse  = reverse
        self.net_A    = _make_st_net(hidden_dim, n_hidden)  # conditioning var -> (s,t)
        self.net_B    = _make_st_net(hidden_dim, n_hidden)  # updated var -> (s,t)

    def _st(self, net: nn.Module, x: torch.Tensor):
        """Apply net to scalar x; squash s to (-2, 2)."""
        out = net(x.unsqueeze(-1))          # (N, 2)
        s   = 2.0 * torch.tanh(out[:, 0])  # in (-2, 2)
        t   = out[:, 1]
        return s, t

    def forward(self, phi: torch.Tensor, psi: torch.Tensor):
        if not self.reverse:
            # psi first, conditioned on phi
            s1, t1 = self._st(self.net_A, phi)
            psi_new = psi * s1.exp() + t1
            # then phi, conditioned on new psi
            s2, t2 = self._st(self.net_B, psi_new)
            phi_new = phi * s2.exp() + t2
        else:
            # phi first, conditioned on psi
            s1, t1 = self._st(self.net_A, psi)
            phi_new = phi * s1.exp() + t1
            # then psi, conditioned on new phi
            s2, t2 = self._st(self.net_B, phi_new)
            psi_new = psi * s2.exp() + t2

        log_det = s1 + s2   # (N,)
        return phi_new, psi_new, log_det

    def inverse(self, phi_new: torch.Tensor, psi_new: torch.Tensor):
        if not self.reverse:
            # undo phi step  (net_B conditioned on psi_new)
            s2, t2 = self._st(self.net_B, psi_new)
            phi    = (phi_new - t2) * (-s2).exp()
            # undo psi step  (net_A conditioned on phi)
            s1, t1 = self._st(self.net_A, phi)
            psi    = (psi_new - t1) * (-s1).exp()
        else:
            # undo psi step  (net_B conditioned on phi_new)
            s2, t2 = self._st(self.net_B, phi_new)
            psi    = (psi_new - t2) * (-s2).exp()
            # undo phi step  (net_A conditioned on psi)
            s1, t1 = self._st(self.net_A, psi)
            phi    = (phi_new - t1) * (-s1).exp()

        return phi, psi


# ---------------------------------------------------------------------------
# Joint Coupling Flow
# ---------------------------------------------------------------------------

class JointCouplingFlow(nn.Module):
    """
    K alternating JointCouplingLayers + output affine normalisation.

    The output affine layer ensures z_phi, z_psi have zero mean / unit variance
    at initialisation (before training changes things).

    log|det J| is accumulated across all layers plus the output scale.
    """

    def __init__(
        self,
        n_layers:   int = 6,
        hidden_dim: int = 32,
        n_hidden:   int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            JointCouplingLayer(hidden_dim, n_hidden, reverse=(i % 2 == 1))
            for i in range(n_layers)
        ])
        # Output normalisation (per variable)
        self.log_scale_phi = nn.Parameter(torch.tensor(0.0))
        self.log_scale_psi = nn.Parameter(torch.tensor(0.0))
        self.shift_phi     = nn.Parameter(torch.tensor(0.0))
        self.shift_psi     = nn.Parameter(torch.tensor(0.0))

    def forward(self, phi: torch.Tensor, psi: torch.Tensor):
        log_det = phi.new_zeros(phi.shape[0])
        for layer in self.layers:
            phi, psi, ld = layer(phi, psi)
            log_det = log_det + ld

        # Output affine normalisation
        eps      = 1e-8
        z_phi    = (phi - self.shift_phi) / (self.log_scale_phi.exp() + eps)
        z_psi    = (psi - self.shift_psi) / (self.log_scale_psi.exp() + eps)
        log_det  = log_det - self.log_scale_phi - self.log_scale_psi
        return z_phi, z_psi, log_det

    def inverse(self, z_phi: torch.Tensor, z_psi: torch.Tensor):
        eps  = 1e-8
        phi  = z_phi * (self.log_scale_phi.exp() + eps) + self.shift_phi
        psi  = z_psi * (self.log_scale_psi.exp() + eps) + self.shift_psi
        for layer in reversed(self.layers):
            phi, psi = layer.inverse(phi, psi)
        return phi, psi


# ---------------------------------------------------------------------------
# Closed-form OLS  (differentiable)
# ---------------------------------------------------------------------------

def compute_A(z_phi: torch.Tensor, z_psi: torch.Tensor) -> torch.Tensor:
    """
    A = Cov(z_phi, z_psi) / Var(z_psi)   [scalar OLS coefficient]

    Differentiable: gradients w.r.t. z_phi, z_psi flow back through the
    autograd graph.  Recomputed every batch so it tracks the current flow.
    """
    z_phi_c = z_phi - z_phi.mean()
    z_psi_c = z_psi - z_psi.mean()
    cov     = (z_phi_c * z_psi_c).mean()
    var     = (z_psi_c ** 2).mean().clamp(min=1e-8)
    return cov / var


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def jcf_loss(
    z_phi:         torch.Tensor,
    z_psi:         torch.Tensor,
    log_det:       torch.Tensor,
    lambda_perp:   float = 1.0,
    lambda_white:  float = 1.0,
):
    """
    L = L_flow + lambda_perp * L_perp + lambda_white * L_white

    L_flow  = 0.5*(z^2) - log|J|             NLL under N(0,I)
    A       = Cov(z_phi,z_psi) / Var(z_psi)  closed-form, differentiable
    r       = z_phi - A * z_psi              residual
    L_perp  = Cov(r, z_psi)^2               orthogonality (=0 by OLS; numerics aid)
    L_white = (Var(r) - (1-A^2))^2          residual variance = Gaussian cond. var
    """
    # Flow loss
    L_flow = (0.5 * (z_phi ** 2 + z_psi ** 2) - log_det).mean()

    # Closed-form A and residual
    A = compute_A(z_phi, z_psi)
    r = z_phi - A * z_psi

    # L_perp: Cov(r, z_psi)
    r_c       = r - r.mean()
    z_psi_c   = z_psi - z_psi.mean()
    L_perp    = ((r_c * z_psi_c).mean()) ** 2

    # L_white: Var(r) should equal 1 - A^2 (Gaussian conditional variance)
    target_var = (1.0 - A ** 2).clamp(min=0.01)
    L_white    = ((r ** 2).mean() - target_var) ** 2

    total = L_flow + lambda_perp * L_perp + lambda_white * L_white

    bd = {
        "total":  total.item(),
        "flow":   L_flow.item(),
        "perp":   L_perp.item(),
        "white":  L_white.item(),
        "A":      A.item(),
        "mi":     float(-0.5 * np.log(max(1.0 - A.item() ** 2, 1e-8))),
    }
    return total, bd


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_jcf(
    model:         JointCouplingFlow,
    phi_t:         torch.Tensor,
    psi_t:         torch.Tensor,
    num_epochs:    int   = 1000,
    lr:            float = 1e-3,
    lambda_perp:   float = 1.0,
    lambda_white:  float = 1.0,
    grad_clip:     float = 5.0,
    verbose:       bool  = True,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    log_every = max(1, num_epochs // 10)
    history   = {"total": [], "flow": [], "perp": [], "white": [], "A": [], "mi": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        z_phi, z_psi, log_det = model(phi_t, psi_t)
        loss, bd = jcf_loss(z_phi, z_psi, log_det,
                             lambda_perp=lambda_perp,
                             lambda_white=lambda_white)

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
                    f"NLL={bd['flow']:+.4f}  "
                    f"perp={bd['perp']:.2e}  "
                    f"white={bd['white']:.4f}  "
                    f"A={bd['A']:+.4f}  "
                    f"I\u2248{bd['mi']:.4f} nats"
                )

    model.eval()
    return history


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def aind_decomposition_jcf(
    Phi,
    Psi_plus,
    n_layers:      int   = 6,
    hidden_dim:    int   = 32,
    n_hidden:      int   = 2,
    num_epochs:    int   = 1000,
    lr:            float = 1e-3,
    lambda_perp:   float = 1.0,
    lambda_white:  float = 1.0,
    verbose:       bool  = True,
    seed:          int   = 42,
    device=None,
):
    """
    aIND via Joint Coupling Flow.

    Steps
    -----
    1.  Pre-standardize (Phi, Psi+) to zero mean / unit variance.
    2.  Train JointCouplingFlow with L_flow + L_perp + L_white.
    3.  Compute A = Cov(z_phi, z_psi) / Var(z_psi) [closed-form OLS].
    4.  z_phi^I = A * z_psi,   z_phi^R = z_phi - z_phi^I.
    5.  Decode: Phi^I = pi_phi( f^{-1}(z_phi^I, z_psi) ).
        [Full joint inverse uses both z_phi^I and z_psi — exploits coupling t functions.]
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

    model = JointCouplingFlow(
        n_layers=n_layers, hidden_dim=hidden_dim, n_hidden=n_hidden
    ).to(device)

    if verbose:
        n_p = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_p} params  "
              f"({n_layers} coupling layers, hidden_dim={hidden_dim})")
        print(f"Training: {num_epochs} epochs  lr={lr}  "
              f"lambda_perp={lambda_perp}  lambda_white={lambda_white}")
        print("=" * 60)

    history = train_jcf(
        model, phi_t, psi_t,
        num_epochs=num_epochs,
        lr=lr,
        lambda_perp=lambda_perp,
        lambda_white=lambda_white,
        verbose=verbose,
    )

    # --- Compute A and decompose ---
    model.eval()
    with torch.no_grad():
        z_phi, z_psi, _ = model(phi_t, psi_t)

        # Closed-form OLS (no grad needed here)
        z_phi_c = z_phi - z_phi.mean()
        z_psi_c = z_psi - z_psi.mean()
        A_val   = float((z_phi_c * z_psi_c).mean() /
                        (z_psi_c ** 2).mean().clamp(min=1e-8))

        z_phi_I = A_val * z_psi
        z_phi_R = z_phi - z_phi_I

        # Decode: full joint inverse with (z_phi^I, z_psi)
        # pi_phi extracts the phi-component of f^{-1}(z_phi^I, z_psi)
        Phi_I_std_flat, _ = model.inverse(z_phi_I, z_psi)
        Phi_I_std_flat    = Phi_I_std_flat.cpu().numpy()

    Phi_I_flat = Phi_I_std_flat * phi_std + phi_mean
    Phi_R_flat = phi_flat - Phi_I_flat
    mi_estimate = float(-0.5 * np.log(max(1.0 - A_val ** 2, 1e-8)))

    if verbose:
        r_R  = float(np.corrcoef(z_phi_R.cpu().numpy(), z_psi.cpu().numpy())[0, 1])
        r_var = float((z_phi_R.cpu().numpy() ** 2).mean())
        print(f"\nClosed-form A = {A_val:+.4f}")
        print(f"I(Phi; Psi) ~= {mi_estimate:.4f} nats")
        print(f"corr(z_phi^R, z_psi) = {r_R:.4f}  (should be ~0)")
        print(f"Var(z_phi^R) = {r_var:.4f}  (target: 1 - A^2 = {1 - A_val**2:.4f})")

    return {
        "Phi_I":        Phi_I_flat,
        "Phi_R":        Phi_R_flat,
        "z_phi":        z_phi.cpu().numpy(),
        "z_psi":        z_psi.cpu().numpy(),
        "z_phi_I":      z_phi_I.cpu().numpy(),
        "z_phi_R":      z_phi_R.cpu().numpy(),
        "A":            A_val,
        "mi_estimate":  mi_estimate,
        "loss_history": history,
        "model":        model,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_latent_space(
    z_phi, z_psi, z_phi_I, z_phi_R, A,
    output_dir: str = "results_jcf",
    prefix:     str = "aind_jcf",
):
    os.makedirs(output_dir, exist_ok=True)
    N   = z_phi.size
    idx = np.random.choice(N, min(3000, N), replace=False)
    zp  = z_phi[idx];   zpsi = z_psi[idx]
    zI  = z_phi_I[idx]; zR   = z_phi_R[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    data = [
        (zp,  zpsi, f"Joint latent (z\u03a6, z\u03a8)\nA = {A:+.4f}", "steelblue"),
        (zI,  zpsi, f"Informative z\u03a6\u1d35 = A\u00b7z\u03a8\nLine slope = A", "darkorange"),
        (zR,  zpsi, "Residual z\u03a6\u1d3f = z\u03a6 \u2212 A\u00b7z\u03a8\nTarget: cloud", "firebrick"),
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
    history:    dict,
    output_dir: str = "results_jcf",
    prefix:     str = "aind_jcf",
):
    os.makedirs(output_dir, exist_ok=True)
    n  = len(history["total"])
    xs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(xs, history["flow"], color="steelblue", lw=2)
    axes[0].set_title("NLL  L_flow"); axes[0].set_xlabel("Checkpoint"); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(xs, [max(v, 1e-12) for v in history["perp"]], color="firebrick", lw=2)
    axes[1].set_title("L_perp  (Cov(r,z\u03a8)\u00b2)\nTarget: ~0"); axes[1].set_xlabel("Checkpoint"); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(xs, [max(v, 1e-12) for v in history["white"]], color="darkorange", lw=2)
    axes[2].set_title("L_white  (Var(r) \u2212 1\u2212A\u00b2)\u00b2\nConditional var match"); axes[2].set_xlabel("Checkpoint"); axes[2].grid(True, alpha=0.3)

    axes[3].plot(xs, history["A"], color="purple", lw=2)
    axes[3].axhline(0, color="k", lw=0.5, ls="--")
    ax2 = axes[3].twinx()
    ax2.plot(xs, history["mi"], color="green", lw=1.5, ls="--", label="I~=\u22120.5log(1\u2212A\u00b2)")
    ax2.set_ylabel("I [nats]", color="green")
    axes[3].set_title("OLS coefficient A\n(and implied MI)"); axes[3].set_xlabel("Checkpoint"); axes[3].grid(True, alpha=0.3)
    axes[3].set_ylabel("A")

    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_training.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training dynamics saved to: {path}")


def visualize_decomposition(
    Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true, Phi_R_true,
    output_dir: str = "results_jcf",
    prefix:     str = "aind_jcf",
):
    os.makedirs(output_dir, exist_ok=True)

    if Phi_I.ndim == 1:
        Phi_I = Phi_I.reshape(Phi.shape)
        Phi_R = Phi_R.reshape(Phi.shape)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = [
        "\u03a6 (Source)\n\u03a6 = f + g",
        "\u03a6\u1d35_true = f\n(True Informative)",
        "\u03a6\u1d35 (Reconstructed)\n[Decoded via joint inverse]",
        "\u03a8\u207a (Target)\n\u03a8\u207a = 0.5f\u00b2 \u2212 0.2f + \u03b5",
        "\u03a6\u1d3f_true = g\n(True Residual)",
        "\u03a6\u1d3f (Reconstructed Residual)\n\u03a6\u1d3f = \u03a6 \u2212 \u03a6\u1d35",
    ]
    fields = [Phi, Phi_I_true, Phi_I, Psi_plus, Phi_R_true, Phi_R]
    for ax, field, title in zip(axes.flat, fields, titles):
        im = ax.imshow(field, origin="lower", cmap="RdBu",
                       extent=[0, 1, 0, 2], aspect=0.5)
        ax.set_title(title, fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fp = os.path.join(output_dir, f"{prefix}_fields.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Field visualization saved to: {fp}")

    # Scatter diagnostics
    pf  = Phi.ravel(); psif = Psi_plus.ravel()
    pIf = Phi_I.ravel(); pRf = Phi_R.ravel()
    pIt = Phi_I_true.ravel()
    idx = np.random.choice(len(pf), min(5000, len(pf)), replace=False)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(pIf[idx], psif[idx], s=6, alpha=0.5); plt.grid(True, alpha=0.3)
    plt.xlabel("\u03a6\u1d35"); plt.ylabel("\u03a8\u207a"); plt.title("\u03a8\u207a vs \u03a6\u1d35")

    plt.subplot(1, 3, 2)
    plt.scatter(pIf[idx], pRf[idx], s=6, alpha=0.5, color="red"); plt.grid(True, alpha=0.3)
    plt.xlabel("\u03a6\u1d35"); plt.ylabel("\u03a6\u1d3f"); plt.title("Independence check")

    plt.subplot(1, 3, 3)
    plt.scatter(pIt[idx], pIf[idx], s=6, alpha=0.5, color="green"); plt.grid(True, alpha=0.3)
    rng = np.linspace(pIt.min(), pIt.max(), 100)
    plt.plot(rng, rng, "k--", lw=2, label="y = x (perfect)")
    plt.xlabel("\u03a6\u1d35_true = f"); plt.ylabel("\u03a6\u1d35 (recon)"); plt.title("Reconstruction quality")
    plt.legend()

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
    # Flow architecture
    n_layers:     int   = 6,
    hidden_dim:   int   = 32,
    n_hidden:     int   = 2,
    # Training
    num_epochs:   int   = 1000,
    lr:           float = 1e-3,
    lambda_perp:  float = 1.0,
    lambda_white: float = 1.0,
    # Output
    output_dir:   str   = "results_jcf",
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
    Phi_I_true = f_func(X, Y, t)   # true informative = f
    Phi_R_true = g_func(X, Y, t)   # true residual    = g

    if verbose:
        print("=" * 60)
        print("aIND  --  Joint Coupling Flow  (JCF)")
        print("=" * 60)
        print(f"Toy v1: Phi = f + g,   Psi+ = 0.5*f^2 - 0.2*f + eps")
        print(f"Grid: {nx} x {ny}   t={t}")
        print(f"Flow: {n_layers} coupling layers  hidden_dim={hidden_dim}  n_hidden={n_hidden}")
        print(f"Epochs={num_epochs}  lr={lr}  lambda_perp={lambda_perp}  lambda_white={lambda_white}")
        # ground truth sanity check
        mi_gt = mutual_info_regression(
            Phi_I_true.ravel().reshape(-1, 1), Phi_R_true.ravel(), random_state=42)[0]
        print(f"\nGround truth I(f; g) = {mi_gt:.6f}  (target: ~0)")
        print()

    results = aind_decomposition_jcf(
        Phi, Psi_plus,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        num_epochs=num_epochs,
        lr=lr,
        lambda_perp=lambda_perp,
        lambda_white=lambda_white,
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
        print("Final Evaluation Metrics  (Joint Coupling Flow aIND)")
        print("=" * 60)
        print(f"  OLS coefficient A                  : {results['A']:+.6f}")
        print(f"  I(Phi;Psi) ~= -0.5*log(1-A^2)     : {results['mi_estimate']:.6f} nats")
        print(f"  I(Phi_I ; Phi_R)                   : {metrics['mutual_information']:.6f}")
        print(f"  Residual energy ||Phi - Phi_I||^2  : {metrics['residual_energy']:.6f}")
        print(f"  MI(Phi_I , Psi+)                   : {metrics['mi_phiI_psi']:.6f}")
        print(f"  MI(Phi_R , Psi+)                   : {metrics['mi_phiR_psi']:.6f}")
        print(f"  GT error ||Phi_I - Phi_I_true||^2  : {metrics['gt_error']:.6f}")
        print("=" * 60)

    path = os.path.join(output_dir, "aind_jcf_metrics.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("aIND Metrics  (Joint Coupling Flow)\n")
        fh.write(f"n_layers={n_layers} | hidden_dim={hidden_dim} | epochs={num_epochs}\n")
        fh.write(f"A = {results['A']:+.6f}  |  "
                 f"I(Phi;Psi) ~= {results['mi_estimate']:.6f} nats\n")
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
        A=results["A"],
        output_dir=output_dir, prefix="aind_jcf",
    )
    visualize_training(results["loss_history"], output_dir=output_dir, prefix="aind_jcf")
    visualize_decomposition(
        Phi, Psi_plus, Phi_I, Phi_R,
        Phi_I_true, Phi_R_true,
        output_dir=output_dir, prefix="aind_jcf",
    )

    return results, metrics


if __name__ == "__main__":
    main(
        nx=100,
        ny=100,
        t=0.0,
        n_layers=3,
        hidden_dim=32,
        n_hidden=2,
        num_epochs=1000,
        lr=2e-3,
        lambda_perp=1.0,
        lambda_white=1.0,
        output_dir="results_jcf",
        seed=1,
        verbose=True,
    )
