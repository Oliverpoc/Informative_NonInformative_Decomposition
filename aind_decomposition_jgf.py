"""
aIND via Joint Gaussian Flow  (JGF)

Key idea
--------
Train TWO marginal MonotoneFlows jointly under the BIVARIATE Gaussian NLL:

    (z_phi, z_psi) = (f_phi(Phi), f_psi(Psi+))

    Target: N(0, Sigma),  Sigma = [[1, rho], [rho, 1]],  rho LEARNABLE.

    L_joint = 0.5/(1-rho^2) * (z_phi^2 + z_psi^2 - 2*rho*z_phi*z_psi)
            + 0.5 * log(1 - rho^2)
            - log|df_phi/dPhi| - log|df_psi/dPsi|

The cross-term  -rho/(1-rho^2) * z_phi * z_psi  in the gradient couples the
training of f_phi and f_psi: each flow is trained not just to Gaussianize its
own marginal but to align its output with the other field's latent code.

OLS decomposition (exact for bivariate Gaussian)
-------------------------------------------------
    z_phi^I = rho * z_psi          E[z_phi | z_psi]  -- linear, exact
    z_phi^R = z_phi - z_phi^I      independent of z_psi by Gaussian property

    Mutual information:  I(Phi; Psi) ~= -0.5 * log(1 - rho^2)   [nats]

Decode
------
    Phi^I = f_phi^{-1}(z_phi^I)     (bisection inverse of monotone flow)
    Phi^R = Phi - Phi^I

Advantages over v2_glf.py (two flows + HSIC + g_theta MLP)
-----------------------------------------------------------
  * No independence penalty needed: Gaussian property gives z_phi^R ⊥ z_psi
    EXACTLY once (z_phi, z_psi) is jointly Gaussian.
  * No g_theta MLP to tune: single scalar rho captures all linear information.
  * Information is directly readable: I ~= -0.5 * log(1 - rho^2).
  * Training is simpler: one loss, no HSIC subsampling.

Built-in clean nonlinear toy
-----------------------------
    u = 2*sin(4*pi*x)*sin(4*pi*y)            informative latent
    v = (1/5)*sin(7*sqrt2*pi*x)*sin(8*sqrt3*pi*y)  residual latent

    Phi       = tanh(alpha*u) + sigma_v*v    additive-nonlinear mixing
    Phi^I_true = tanh(alpha*u)               depends only on u
    Phi^R_true = sigma_v*v                   depends only on v

    Psi+      = 0.5*u^2 - 0.2*u + eps       nonlinear target

Key property: I(Phi^I_true; Phi^R_true) = 0 EXACTLY (u ⊥ v by construction).
The algorithm must learn to invert tanh(alpha*u) out of Phi given Psi+.
This fixes the structural flaw of toy_v2 (shared (c+g) factor).
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


# ---------------------------------------------------------------------------
# Clean nonlinear toy problem
# ---------------------------------------------------------------------------

def make_toy(
    nx:        int   = 100,
    ny:        int   = 100,
    t:         float = 0.0,
    alpha:     float = 2.0,
    sigma_v:   float = 0.5,
    noise_std: float = 1e-8,
    seed:      int   = 42,
):
    """
    Build the clean nonlinear toy where I(Phi^I_true; Phi^R_true) = 0 exactly.

    u and v are nearly orthogonal spatial fields (incommensurate frequencies).
    The mixing is nonlinear but additive, so the ground-truth split is clean.

    Parameters
    ----------
    alpha   : controls the nonlinearity strength of tanh(alpha*u).
              alpha=0: linear;  alpha -> inf: binary (hard sign).
    sigma_v : variance fraction carried by the residual component.
    """
    np.random.seed(seed)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    # Independent latent fields
    u = 2.0 * np.sin(4 * np.pi * X - 1.5 * t) * np.sin(4 * np.pi * Y)
    v = (1 / 5) * np.sin(7 * np.sqrt(2) * np.pi * X - 0.1 * t) * np.sin(
        8 * np.sqrt(3) * np.pi * Y - 0.5 * t
    )

    # Ground truth (additive, truly independent)
    Phi_I_true = np.tanh(alpha * u)
    Phi_R_true = sigma_v * v

    # Source field (additive-nonlinear mixing)
    Phi = Phi_I_true + Phi_R_true

    # Target (nonlinear in u only)
    eps     = np.random.normal(0, noise_std, size=u.shape)
    Psi_plus = 0.5 * u ** 2 - 0.2 * u + eps

    return Phi, Psi_plus, Phi_I_true, Phi_R_true, u, v

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_decomposition(Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true=None):
    """
    mutual_information : I(Phi_I; Phi_R)  (lower = more independent)
    residual_energy    : MSE(Phi, Phi_I)
    mi_phiI_psi        : MI(Phi_I, Psi+)  (higher = informative)
    mi_phiR_psi        : MI(Phi_R, Psi+)  (lower  = non-informative)
    gt_error           : MSE(Phi_I, Phi_I_true)
    """
    m = {}
    m["mutual_information"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Phi_I, random_state=42
    )[0]
    m["residual_energy"] = mean_squared_error(Phi, Phi_I)
    m["mi_phiI_psi"] = mutual_info_regression(
        Phi_I.reshape(-1, 1), Psi_plus, random_state=42
    )[0]
    m["mi_phiR_psi"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Psi_plus, random_state=42
    )[0]
    m["gt_error"] = (
        mean_squared_error(Phi_I_true, Phi_I)
        if Phi_I_true is not None
        else float("nan")
    )
    m["total_loss"] = m["mutual_information"] + m["residual_energy"]
    return m


# ---------------------------------------------------------------------------
# Monotone Normalizing Flow  (1D element-wise)
# ---------------------------------------------------------------------------

class MonotoneFlow(nn.Module):
    """
    Strictly monotone 1D flow:
        z = (raw(x) - shift) / exp(log_scale)
        raw(x) = softplus(slope)*x + sum_k softplus(w_k)*tanh(softplus(a_k)*x + b_k)
    Log-Jacobian: log|dz/dx| = log|d(raw)/dx| - log_scale.
    Inverse via 60-step bisection (error < 2e-17).
    """

    def __init__(self, n_terms: int = 32):
        super().__init__()
        self.log_slope = nn.Parameter(torch.tensor(0.0))
        self.log_w     = nn.Parameter(torch.zeros(n_terms) - 1.0)
        self.log_a     = nn.Parameter(torch.zeros(n_terms))
        self.b         = nn.Parameter(torch.linspace(-3.0, 3.0, n_terms))
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.shift     = nn.Parameter(torch.tensor(0.0))

    def _raw(self, x):
        slope = F.softplus(self.log_slope)
        w = F.softplus(self.log_w)
        a = F.softplus(self.log_a)
        h = torch.tanh(a[None, :] * x[:, None] + self.b[None, :])
        return slope * x + (w[None, :] * h).sum(-1)

    def _log_d_raw(self, x):
        slope = F.softplus(self.log_slope)
        w = F.softplus(self.log_w)
        a = F.softplus(self.log_a)
        h     = torch.tanh(a[None, :] * x[:, None] + self.b[None, :])
        dtanh = (1.0 - h ** 2) * a[None, :]
        df    = slope + (w[None, :] * dtanh).sum(-1)
        return torch.log(df.clamp(min=1e-8))

    def forward(self, x):
        return (self._raw(x) - self.shift) / (self.log_scale.exp() + 1e-8)

    def log_abs_jacobian(self, x):
        return self._log_d_raw(x) - self.log_scale

    @torch.no_grad()
    def inverse(self, z, n_iter: int = 60):
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
# Joint Gaussian Flow
# ---------------------------------------------------------------------------

class JointGaussianFlow(nn.Module):
    """
    Two MonotoneFlows + one scalar rho, trained with bivariate Gaussian NLL.

        (z_phi, z_psi) = (f_phi(Phi), f_psi(Psi+))

    Target distribution: N(0, [[1, rho], [rho, 1]])

    The bivariate NLL couples the two flows through the cross-term:
        -rho / (1 - rho^2) * z_phi * z_psi
    This gradient term pushes each flow to align its latent code with the
    other field's code by the amount rho — the joint aspect of the training.

    After training the OLS decomposition is exact for Gaussian joint:
        z_phi^I = rho * z_psi
        z_phi^R = z_phi - rho * z_psi     (independent of z_psi)
        I(Phi; Psi) ~= -0.5 * log(1 - rho^2)
    """

    def __init__(self, n_terms: int = 32):
        super().__init__()
        self.f_phi   = MonotoneFlow(n_terms=n_terms)
        self.f_psi   = MonotoneFlow(n_terms=n_terms)
        # rho_raw is unconstrained; rho = tanh(rho_raw) lies in (-1, 1)
        self.rho_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def rho(self) -> torch.Tensor:
        return torch.tanh(self.rho_raw)

    def forward(self, phi: torch.Tensor, psi: torch.Tensor):
        z_phi   = self.f_phi(phi)
        z_psi   = self.f_psi(psi)
        ldj_phi = self.f_phi.log_abs_jacobian(phi)
        ldj_psi = self.f_psi.log_abs_jacobian(psi)
        return z_phi, z_psi, ldj_phi, ldj_psi

    def joint_nll(
        self,
        z_phi:   torch.Tensor,
        z_psi:   torch.Tensor,
        ldj_phi: torch.Tensor,
        ldj_psi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean bivariate Gaussian NLL per sample.

        L = 0.5/(1-rho^2) * (z_phi^2 + z_psi^2 - 2*rho*z_phi*z_psi)
          + 0.5 * log(1 - rho^2)
          - ldj_phi - ldj_psi
        """
        rho  = self.rho
        det  = (1.0 - rho ** 2).clamp(min=1e-6)
        quad = (z_phi ** 2 + z_psi ** 2 - 2.0 * rho * z_phi * z_psi) / det
        return (0.5 * quad + 0.5 * torch.log(det) - ldj_phi - ldj_psi).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_jgf(
    model:      JointGaussianFlow,
    phi_t:      torch.Tensor,
    psi_t:      torch.Tensor,
    num_epochs: int   = 1000,
    lr:         float = 1e-3,
    grad_clip:  float = 5.0,
    verbose:    bool  = True,
) -> dict:
    """Train the joint Gaussian flow with bivariate NLL."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    log_every = max(1, num_epochs // 10)
    history   = {"nll": [], "rho": [], "mi_nats": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        z_phi, z_psi, ldj_phi, ldj_psi = model(phi_t, psi_t)
        loss = model.joint_nll(z_phi, z_psi, ldj_phi, ldj_psi)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == num_epochs:
            rho_val = model.rho.item()
            mi_val  = -0.5 * np.log(max(1 - rho_val ** 2, 1e-8))
            history["nll"].append(loss.item())
            history["rho"].append(rho_val)
            history["mi_nats"].append(mi_val)

            if verbose:
                print(
                    f"  [Epoch {epoch:5d}]  "
                    f"NLL={loss.item():+.4f}  "
                    f"rho={rho_val:+.4f}  "
                    f"I(Phi;Psi)~={mi_val:.4f} nats"
                )

    model.eval()
    return history


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def aind_decomposition_jgf(
    Phi,
    Psi_plus,
    n_terms:    int   = 32,
    num_epochs: int   = 1000,
    lr:         float = 1e-3,
    verbose:    bool  = True,
    seed:       int   = 42,
    device=None,
):
    """
    aIND via Joint Gaussian Flow.

    Steps
    -----
    1. Pre-standardize Phi and Psi+ to zero mean / unit variance.
    2. Train f_phi, f_psi, rho jointly with bivariate Gaussian NLL.
    3. Decompose: z_phi^I = rho * z_psi,  z_phi^R = z_phi - z_phi^I.
    4. Decode: Phi^I = f_phi^{-1}(z_phi^I) via bisection.
    5. Phi^R = Phi - Phi^I.

    Returns
    -------
    dict with: Phi_I, Phi_R, z_phi, z_psi, z_phi_I, z_phi_R,
               rho, mi_estimate, loss_history, model
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

    # Pre-standardize
    phi_mean, phi_std = phi_flat.mean(), phi_flat.std() + 1e-8
    psi_mean, psi_std = psi_flat.mean(), psi_flat.std() + 1e-8
    Phi_std = (phi_flat - phi_mean) / phi_std
    Psi_std = (psi_flat - psi_mean) / psi_std

    phi_t = torch.tensor(Phi_std, dtype=torch.float32, device=device)
    psi_t = torch.tensor(Psi_std, dtype=torch.float32, device=device)

    model = JointGaussianFlow(n_terms=n_terms).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params} parameters  "
              f"(f_phi + f_psi: {n_terms} terms each, + 1 rho)")
        print(f"Training: {num_epochs} epochs, lr={lr}")
        print("=" * 60)

    history = train_jgf(
        model, phi_t, psi_t,
        num_epochs=num_epochs,
        lr=lr,
        verbose=verbose,
    )

    # --- OLS decomposition in latent space ---
    model.eval()
    with torch.no_grad():
        z_phi, z_psi, _, _ = model(phi_t, psi_t)

        rho         = model.rho.item()
        z_phi_I     = rho * z_psi
        z_phi_R     = z_phi - z_phi_I

        # Decode informative latent back to standardized Phi space
        Phi_I_std_flat = model.f_phi.inverse(z_phi_I).cpu().numpy()

    # Undo pre-standardization
    Phi_I_flat = Phi_I_std_flat * phi_std + phi_mean
    Phi_R_flat = phi_flat - Phi_I_flat

    mi_estimate = -0.5 * np.log(max(1.0 - rho ** 2, 1e-8))

    if verbose:
        r_I = float(np.corrcoef(z_phi_I.cpu().numpy(), z_psi.cpu().numpy())[0, 1])
        r_R = float(np.corrcoef(z_phi_R.cpu().numpy(), z_psi.cpu().numpy())[0, 1])
        print(f"\nLearned rho = {rho:+.4f}")
        print(f"I(Phi; Psi) ~= {mi_estimate:.4f} nats  [= -0.5*log(1-rho^2)]")
        print(f"corr(z_phi^I, z_psi) = {r_I:.4f}  (= rho by construction)")
        print(f"corr(z_phi^R, z_psi) = {r_R:.4f}  (should be ~0)")

    return {
        "Phi_I":        Phi_I_flat,
        "Phi_R":        Phi_R_flat,
        "z_phi":        z_phi.cpu().numpy(),
        "z_psi":        z_psi.cpu().numpy(),
        "z_phi_I":      z_phi_I.cpu().numpy(),
        "z_phi_R":      z_phi_R.cpu().numpy(),
        "rho":          rho,
        "mi_estimate":  mi_estimate,
        "loss_history": history,
        "model":        model,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_latent_space(
    z_phi, z_psi, z_phi_I, z_phi_R, rho,
    output_dir: str = "results_jgf",
    prefix:     str = "aind_jgf",
):
    """
    Three panels:
      (z_phi, z_psi)   — joint latent: should look like bivariate N(0, [[1,rho],[rho,1]])
      (z_phi^I, z_psi) — informative:  exactly correlated by rho
      (z_phi^R, z_psi) — residual:     should be uncorrelated (and independent)
    """
    os.makedirs(output_dir, exist_ok=True)

    N   = z_phi.size
    idx = np.random.choice(N, min(3000, N), replace=False)
    zp  = z_phi[idx];   zpsi = z_psi[idx]
    zI  = z_phi_I[idx]; zR   = z_phi_R[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    data = [
        (zp,  zpsi, f"Joint latent (z\u03a6, z\u03a8)\nLearned \u03c1 = {rho:+.4f}",        "steelblue"),
        (zI,  zpsi, "Informative z\u03a6\u1d35 = \u03c1\u00b7z\u03a8\nShould lie on a line",  "darkorange"),
        (zR,  zpsi, "Residual z\u03a6\u1d3f = z\u03a6 \u2212 z\u03a6\u1d35\nShould be cloud",  "firebrick"),
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
    output_dir: str = "results_jgf",
    prefix:     str = "aind_jgf",
):
    """Training dynamics: NLL, learned rho, and implied MI over checkpoints."""
    os.makedirs(output_dir, exist_ok=True)

    n  = len(history["nll"])
    xs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(xs, history["nll"], color="steelblue", linewidth=2)
    axes[0].set_xlabel("Log checkpoint")
    axes[0].set_ylabel("Joint NLL")
    axes[0].set_title("Bivariate Gaussian NLL\n(should decrease)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, history["rho"], color="darkorange", linewidth=2)
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("Log checkpoint")
    axes[1].set_ylabel("\u03c1  (learned)")
    axes[1].set_title("Learned correlation \u03c1\n(magnitude = signal strength)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, history["mi_nats"], color="firebrick", linewidth=2)
    axes[2].set_xlabel("Log checkpoint")
    axes[2].set_ylabel("I \u2248 \u22120.5\u00b7log(1\u2212\u03c1\u00b2)  [nats]")
    axes[2].set_title("Implied mutual information\nI(Phi; Psi)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_training.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training dynamics saved to: {path}")


def visualize_decomposition(
    Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true, Phi_R_true,
    output_dir: str = "results_jgf",
    prefix:     str = "aind_jgf",
):
    """2 x 3 field layout with per-subplot colorbars, plus scatter diagnostics."""
    os.makedirs(output_dir, exist_ok=True)

    if Phi_I.ndim == 1:
        Phi_I = Phi_I.reshape(Phi.shape)
        Phi_R = Phi_R.reshape(Phi.shape)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = [
        "\u03a6 (Source)\n\u03a6 = tanh(\u03b1u) + \u03c3_v\u00b7v",
        "\u03a6\u1d35_true = tanh(\u03b1u)\n(True Informative)",
        "\u03a6\u1d35 (Reconstructed Informative)",
        "\u03a8\u207a (Target)\n\u03a8\u207a = 0.5u\u00b2 \u2212 0.2u + \u03b5",
        "\u03a6\u1d3f_true = \u03c3_v\u00b7v\n(True Residual,  I=0 exactly)",
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
    plt.xlabel("\u03a6\u1d35_true = tanh(\u03b1u)")
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
    nx:         int   = 100,
    ny:         int   = 100,
    t:          float = 0.0,
    # Toy parameters
    alpha:      float = 2.0,
    sigma_v:    float = 0.5,
    # Flow
    n_terms:    int   = 32,
    # Training
    num_epochs: int   = 1000,
    lr:         float = 1e-3,
    # Output
    output_dir: str   = "results_jgf",
    seed:       int   = 1,
    verbose:    bool  = True,
):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    Phi, Psi_plus, Phi_I_true, Phi_R_true, u, v = make_toy(
        nx=nx, ny=ny, t=t, alpha=alpha, sigma_v=sigma_v,
        noise_std=1e-8, seed=seed,
    )

    if verbose:
        print("=" * 60)
        print("aIND  --  Joint Gaussian Flow")
        print("=" * 60)
        print(f"Toy:  Phi = tanh({alpha}*u) + {sigma_v}*v,  "
              f"Psi+ = 0.5*u^2 - 0.2*u")
        print(f"Grid: {nx} x {ny}   t={t}")
        print(f"Flow: {n_terms} terms per flow | epochs={num_epochs} | lr={lr}")

        # Verify ground truth independence
        mi_gt = mutual_info_regression(
            Phi_I_true.ravel().reshape(-1, 1),
            Phi_R_true.ravel(), random_state=42
        )[0]
        print(f"\nGround truth check:")
        print(f"  I(Phi^I_true; Phi^R_true) = {mi_gt:.6f}  (target: ~0)")
        r_gt = float(np.corrcoef(Phi_I_true.ravel(), Phi_R_true.ravel())[0, 1])
        print(f"  corr(Phi^I_true, Phi^R_true) = {r_gt:.4f}  (target: ~0)")
        print()

    results = aind_decomposition_jgf(
        Phi, Psi_plus,
        n_terms=n_terms,
        num_epochs=num_epochs,
        lr=lr,
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
        print("Final Evaluation Metrics  (Joint Gaussian Flow aIND)")
        print("=" * 60)
        print(f"  Learned rho                        : {results['rho']:+.6f}")
        print(f"  I(Phi;Psi) ~= -0.5*log(1-rho^2)   : {results['mi_estimate']:.6f} nats")
        print(f"  I(Phi_I ; Phi_R)                   : {metrics['mutual_information']:.6f}  (lower = more independent)")
        print(f"  Residual energy ||Phi - Phi_I||^2  : {metrics['residual_energy']:.6f}")
        print(f"  MI(Phi_I , Psi+)                   : {metrics['mi_phiI_psi']:.6f}  (higher = informative)")
        print(f"  MI(Phi_R , Psi+)                   : {metrics['mi_phiR_psi']:.6f}  (lower = non-informative)")
        print(f"  GT error ||Phi_I - Phi_I_true||^2  : {metrics['gt_error']:.6f}")
        print("=" * 60)

    # Save metrics
    metrics_path = os.path.join(output_dir, "aind_jgf_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write("aIND Metrics  (Joint Gaussian Flow)\n")
        fh.write(f"Toy: tanh({alpha}*u) + {sigma_v}*v  |  "
                 f"n_terms={n_terms}  |  epochs={num_epochs}\n")
        fh.write(f"Learned rho = {results['rho']:+.6f}\n")
        fh.write(f"I(Phi;Psi) ~= {results['mi_estimate']:.6f} nats\n")
        fh.write("=" * 60 + "\n")
        for name, key in [
            ("I(Phi_I ; Phi_R)",              "mutual_information"),
            ("Residual energy",               "residual_energy"),
            ("MI(Phi_I, Psi+)",               "mi_phiI_psi"),
            ("MI(Phi_R, Psi+)",               "mi_phiR_psi"),
            ("GT error ||Phi_I-Phi_I_true||", "gt_error"),
        ]:
            fh.write(f"{name:<35}: {metrics[key]:.6f}\n")
    print(f"Metrics saved to: {metrics_path}")

    visualize_latent_space(
        results["z_phi"], results["z_psi"],
        results["z_phi_I"], results["z_phi_R"],
        rho=results["rho"],
        output_dir=output_dir, prefix="aind_jgf",
    )
    visualize_training(
        results["loss_history"],
        output_dir=output_dir, prefix="aind_jgf",
    )
    visualize_decomposition(
        Phi, Psi_plus, Phi_I, Phi_R,
        Phi_I_true, Phi_R_true,
        output_dir=output_dir, prefix="aind_jgf",
    )

    return results, metrics


if __name__ == "__main__":
    main(
        nx=100,
        ny=100,
        t=0.0,
        alpha=2.0,
        sigma_v=0.5,
        n_terms=32,
        num_epochs=1000,
        lr=1e-3,
        output_dir="results_jgf",
        seed=1,
        verbose=True,
    )
