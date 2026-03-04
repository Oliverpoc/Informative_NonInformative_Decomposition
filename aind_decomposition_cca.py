"""
aIND via CCA-in-Latent  (Canonical Correlation Split)

Key idea
--------
Train two marginal MonotoneFlows with SEPARATE NLLs plus an explicit
CCA term that maximises the canonical correlation between z_phi and z_psi:

    L = L_NLL(f_phi) + L_NLL(f_psi)  -  lambda_cca * rho(z_phi, z_psi)^2

    L_NLL(f) = 0.5*E[z^2] - E[log|dz/dx|]     keeps z ~ N(0,1)
    rho       = Cov(z_phi, z_psi) / (Std(z_phi)*Std(z_psi))  [Pearson / CCA]

For scalar latent codes (m=1) CCA = Pearson r, so the canonical direction
is trivially ±1 and the canonical correlation = |rho|.

How this differs from jgf.py (Joint Gaussian Flow)
----------------------------------------------------
  jgf.py  : bivariate Gaussian NLL with learnable rho.
             The log(1-rho^2) term acts as a built-in regulariser that
             pulls rho away from ±1.

  cca.py  : separate marginal NLLs (each uncoupled from the other)
             PLUS an explicit -lambda_cca*rho^2 maximisation term.
             * The two NLL objectives are kept independent, so each flow
               is only responsible for its own marginal Gaussianity.
             * The CCA gradient acts as an external "alignment signal"
               pushing the flows to agree on a shared latent representation.
             * lambda_cca controls how hard we push alignment vs. Gaussianity.
               Large lambda -> high rho (aggressive alignment).
               Small lambda -> better marginal Gaussianity, moderate rho.

Decomposition (exact for Gaussian joint)
-----------------------------------------
After training, the empirical rho is computed and the OLS split is:

    z_phi^I = rho * z_psi          (shared / informative part)
    z_phi^R = z_phi - z_phi^I      (residual, orthogonal to z_psi)

    Phi^I = f_phi^{-1}(z_phi^I)    (bisection inverse)
    Phi^R = Phi - Phi^I

CCA interpretation
------------------
The latent space has canonical variables:
    U = z_phi,    V = z_psi     (unit-variance, by marginal NLL)
The canonical correlation is rho = Corr(U, V).
The informative subspace is spanned by the first (and only, for m=1)
canonical direction.  I(Phi; Psi) ~= -0.5*log(1-rho^2) nats.

Toy: same additive-nonlinear clean problem as jgf.py
    Phi = tanh(alpha*u) + sigma_v*v,  Psi+ = 0.5*u^2 - 0.2*u + eps
    I(Phi^I_true; Phi^R_true) = 0 exactly.
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
# Clean nonlinear toy  (same as jgf.py)
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
    Phi = tanh(alpha*u) + sigma_v*v,   Psi+ = 0.5*u^2 - 0.2*u + eps
    u ⊥ v  ->  I(Phi^I_true; Phi^R_true) = 0 exactly.
    """
    np.random.seed(seed)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    u = 2.0 * np.sin(4 * np.pi * X - 1.5 * t) * np.sin(4 * np.pi * Y)
    v = (1 / 5) * np.sin(7 * np.sqrt(2) * np.pi * X - 0.1 * t) * np.sin(
        8 * np.sqrt(3) * np.pi * Y - 0.5 * t
    )

    Phi_I_true = np.tanh(alpha * u)
    Phi_R_true = sigma_v * v
    Phi        = Phi_I_true + Phi_R_true
    Psi_plus   = 0.5 * u ** 2 - 0.2 * u + np.random.normal(0, noise_std, size=u.shape)

    return Phi, Psi_plus, Phi_I_true, Phi_R_true, u, v


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
# Monotone Normalizing Flow  (1D element-wise)
# ---------------------------------------------------------------------------

class MonotoneFlow(nn.Module):
    """
    Strictly monotone 1D flow:
        z = (raw(x) - shift) / exp(log_scale)
        raw(x) = softplus(slope)*x + sum_k softplus(w_k)*tanh(softplus(a_k)*x + b_k)
    Inverse via 60-step bisection.
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
# CCA Flow model
# ---------------------------------------------------------------------------

class CCAFlow(nn.Module):
    """
    Two independent MonotoneFlows whose training is coupled by a CCA loss.

    Each flow is responsible only for its own marginal Gaussianity.
    The CCA term is applied externally during training to align both flows.

    No learnable rho: the canonical correlation is computed empirically
    from the batch at each step and used as a gradient signal.
    """

    def __init__(self, n_terms: int = 32):
        super().__init__()
        self.f_phi = MonotoneFlow(n_terms=n_terms)
        self.f_psi = MonotoneFlow(n_terms=n_terms)

    def forward(self, phi: torch.Tensor, psi: torch.Tensor):
        z_phi   = self.f_phi(phi)
        z_psi   = self.f_psi(psi)
        ldj_phi = self.f_phi.log_abs_jacobian(phi)
        ldj_psi = self.f_psi.log_abs_jacobian(psi)
        return z_phi, z_psi, ldj_phi, ldj_psi


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _pearson_r(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable Pearson correlation (scalar CCA for m=1)."""
    xc = x - x.mean()
    yc = y - y.mean()
    cov = (xc * yc).mean()
    return cov / (xc.std() * yc.std() + 1e-8)


def cca_loss(
    z_phi:      torch.Tensor,
    z_psi:      torch.Tensor,
    ldj_phi:    torch.Tensor,
    ldj_psi:    torch.Tensor,
    lambda_cca: float = 1.0,
):
    """
    L = L_NLL(f_phi) + L_NLL(f_psi)  -  lambda_cca * rho^2

    The -rho^2 term maximises the squared canonical correlation.
    The two NLL terms keep each marginal Gaussian independently.
    """
    L_phi = 0.5 * (z_phi ** 2).mean() - ldj_phi.mean()
    L_psi = 0.5 * (z_psi ** 2).mean() - ldj_psi.mean()
    rho   = _pearson_r(z_phi, z_psi)
    L_cca = -lambda_cca * rho ** 2

    total = L_phi + L_psi + L_cca
    return total, {"nll_phi": L_phi.item(), "nll_psi": L_psi.item(),
                   "rho": rho.item(), "cca_term": L_cca.item()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cca(
    model:      CCAFlow,
    phi_t:      torch.Tensor,
    psi_t:      torch.Tensor,
    num_epochs: int   = 1000,
    lr:         float = 1e-3,
    lambda_cca: float = 1.0,
    grad_clip:  float = 5.0,
    verbose:    bool  = True,
) -> dict:
    """
    Two-phase training to avoid degenerate initialisation:
      Phase 1 (first 20%): lambda_cca = 0  — warm up both flows independently.
      Phase 2 (rest):      lambda_cca = target — enable CCA alignment.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    warmup_epochs = num_epochs // 5
    log_every     = max(1, num_epochs // 10)
    history       = {"total": [], "nll_phi": [], "nll_psi": [], "rho": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Linearly ramp up lambda_cca after the warm-up phase
        effective_lambda = 0.0 if epoch <= warmup_epochs else (
            lambda_cca * min(1.0, (epoch - warmup_epochs) / warmup_epochs)
        )

        z_phi, z_psi, ldj_phi, ldj_psi = model(phi_t, psi_t)
        loss, bd = cca_loss(z_phi, z_psi, ldj_phi, ldj_psi,
                            lambda_cca=effective_lambda)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == num_epochs:
            mi_val = -0.5 * np.log(max(1 - bd["rho"] ** 2, 1e-8))
            history["total"].append(loss.item())
            history["nll_phi"].append(bd["nll_phi"])
            history["nll_psi"].append(bd["nll_psi"])
            history["rho"].append(bd["rho"])

            if verbose:
                print(
                    f"  [Epoch {epoch:5d}]  "
                    f"\u03bb_cca={effective_lambda:.2f}  "
                    f"NLL_\u03a6={bd['nll_phi']:.4f}  "
                    f"NLL_\u03a8={bd['nll_psi']:.4f}  "
                    f"\u03c1={bd['rho']:+.4f}  "
                    f"I\u2248{mi_val:.4f} nats"
                )

    model.eval()
    return history


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def aind_decomposition_cca(
    Phi,
    Psi_plus,
    n_terms:    int   = 32,
    num_epochs: int   = 1000,
    lr:         float = 1e-3,
    lambda_cca: float = 1.0,
    verbose:    bool  = True,
    seed:       int   = 42,
    device=None,
):
    """
    aIND via CCA-in-Latent decomposition.

    Steps
    -----
    1. Pre-standardize Phi, Psi+ to zero mean / unit variance.
    2. Train f_phi, f_psi with marginal NLLs + CCA maximization.
    3. Compute empirical rho = Corr(z_phi, z_psi) after training.
    4. Decompose: z_phi^I = rho * z_psi,  z_phi^R = z_phi - z_phi^I.
    5. Decode: Phi^I = f_phi^{-1}(z_phi^I),  Phi^R = Phi - Phi^I.
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

    model = CCAFlow(n_terms=n_terms).to(device)

    if verbose:
        n_p = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_p} params  (f_phi + f_psi, {n_terms} terms each)")
        print(f"Training: {num_epochs} epochs  lr={lr}  lambda_cca={lambda_cca}")
        print(f"Warm-up: first {num_epochs // 5} epochs at lambda_cca=0")
        print("=" * 60)

    history = train_cca(
        model, phi_t, psi_t,
        num_epochs=num_epochs,
        lr=lr,
        lambda_cca=lambda_cca,
        verbose=verbose,
    )

    # --- Empirical canonical correlation and OLS split ---
    model.eval()
    with torch.no_grad():
        z_phi, z_psi, _, _ = model(phi_t, psi_t)

        # rho = empirical canonical correlation (Pearson, scalar CCA)
        rho = _pearson_r(z_phi, z_psi).item()

        # OLS split in latent space
        z_phi_I = rho * z_psi
        z_phi_R = z_phi - z_phi_I

        # Decode informative component
        Phi_I_std_flat = model.f_phi.inverse(z_phi_I).cpu().numpy()

    Phi_I_flat = Phi_I_std_flat * phi_std + phi_mean
    Phi_R_flat = phi_flat - Phi_I_flat
    mi_estimate = -0.5 * np.log(max(1.0 - rho ** 2, 1e-8))

    if verbose:
        r_R = float(np.corrcoef(z_phi_R.cpu().numpy(), z_psi.cpu().numpy())[0, 1])
        print(f"\nEmpirical canonical correlation rho = {rho:+.4f}")
        print(f"I(Phi; Psi) ~= {mi_estimate:.4f} nats  [= -0.5*log(1-rho^2)]")
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
    output_dir: str = "results_cca",
    prefix:     str = "aind_cca",
):
    os.makedirs(output_dir, exist_ok=True)
    N   = z_phi.size
    idx = np.random.choice(N, min(3000, N), replace=False)
    zp  = z_phi[idx];   zpsi = z_psi[idx]
    zI  = z_phi_I[idx]; zR   = z_phi_R[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    data = [
        (zp,  zpsi, f"Joint latent (z\u03a6, z\u03a8)\nCCA \u03c1 = {rho:+.4f}",           "steelblue"),
        (zI,  zpsi, "Informative z\u03a6\u1d35 = \u03c1\u00b7z\u03a8\nShared canonical var.",  "darkorange"),
        (zR,  zpsi, "Residual z\u03a6\u1d3f\nNon-shared / independent",                         "firebrick"),
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
    lambda_cca: float,
    output_dir: str = "results_cca",
    prefix:     str = "aind_cca",
):
    os.makedirs(output_dir, exist_ok=True)
    n  = len(history["rho"])
    xs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(xs, history["nll_phi"], label="NLL \u03a6", linewidth=2)
    axes[0].plot(xs, history["nll_psi"], label="NLL \u03a8", linewidth=2, linestyle="--")
    axes[0].plot(xs, history["total"],   label="Total",    linewidth=1, color="k")
    axes[0].set_xlabel("Log checkpoint"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, history["rho"], color="darkorange", linewidth=2)
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("Log checkpoint"); axes[1].set_ylabel("\u03c1")
    axes[1].set_title(f"Canonical correlation \u03c1\n(\u03bb_cca={lambda_cca})")
    axes[1].grid(True, alpha=0.3)

    mi_vals = [-0.5 * np.log(max(1 - r ** 2, 1e-8)) for r in history["rho"]]
    axes[2].plot(xs, mi_vals, color="firebrick", linewidth=2)
    axes[2].set_xlabel("Log checkpoint")
    axes[2].set_ylabel("I \u2248 \u22120.5\u00b7log(1\u2212\u03c1\u00b2) [nats]")
    axes[2].set_title("Implied mutual information")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_training.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training dynamics saved to: {path}")


def visualize_decomposition(
    Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true, Phi_R_true,
    output_dir: str = "results_cca",
    prefix:     str = "aind_cca",
):
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
        "\u03a6\u1d3f (Reconstructed Residual)",
    ]
    fields = [Phi, Phi_I_true, Phi_I, Psi_plus, Phi_R_true, Phi_R]
    for ax, field, title in zip(axes.flat, fields, titles):
        im = ax.imshow(field, origin="lower", cmap="RdBu",
                       extent=[0, 1, 0, 2], aspect=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fields_path = os.path.join(output_dir, f"{prefix}_fields.png")
    fig.savefig(fields_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Field visualization saved to: {fields_path}")

    # Scatter
    pf = Phi.ravel(); psif = Psi_plus.ravel()
    pIf = Phi_I.ravel(); pRf = Phi_R.ravel()
    pIt = Phi_I_true.ravel()
    idx = np.random.choice(len(pf), min(5000, len(pf)), replace=False)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(pIf[idx], psif[idx], s=6, alpha=0.5); plt.grid(True, alpha=0.3)
    plt.xlabel("\u03a6\u1d35"); plt.ylabel("\u03a8\u207a"); plt.title("\u03a8\u207a \u2248 F(\u03a6\u1d35)")

    plt.subplot(1, 3, 2)
    plt.scatter(pIf[idx], pRf[idx], s=6, alpha=0.5, color="red"); plt.grid(True, alpha=0.3)
    plt.xlabel("\u03a6\u1d35"); plt.ylabel("\u03a6\u1d3f"); plt.title("Independence check")

    plt.subplot(1, 3, 3)
    plt.scatter(pIt[idx], pIf[idx], s=6, alpha=0.5, color="green"); plt.grid(True, alpha=0.3)
    rng = np.linspace(pIt.min(), pIt.max(), 100)
    plt.plot(rng, rng, "k--", lw=2, label="y = x")
    plt.xlabel("\u03a6\u1d35_true"); plt.ylabel("\u03a6\u1d35 (recon)"); plt.title("Reconstruction quality")
    plt.legend()

    plt.tight_layout()
    sp = os.path.join(output_dir, f"{prefix}_scatter.png")
    plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plots saved to: {sp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    nx:         int   = 100,
    ny:         int   = 100,
    t:          float = 0.0,
    alpha:      float = 2.0,
    sigma_v:    float = 0.5,
    n_terms:    int   = 32,
    num_epochs: int   = 1000,
    lr:         float = 1e-3,
    lambda_cca: float = 1.0,
    output_dir: str   = "results_cca",
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
        print("aIND  --  CCA-in-Latent  (Canonical Correlation Split)")
        print("=" * 60)
        print(f"Toy:  Phi = tanh({alpha}*u) + {sigma_v}*v,  Psi+ = 0.5*u^2 - 0.2*u")
        print(f"Grid: {nx} x {ny}   t={t}")
        print(f"Flow: {n_terms} terms | epochs={num_epochs} | lr={lr} | lambda_cca={lambda_cca}")
        mi_gt = mutual_info_regression(
            Phi_I_true.ravel().reshape(-1, 1), Phi_R_true.ravel(), random_state=42)[0]
        print(f"\nGround truth I(Phi^I_true; Phi^R_true) = {mi_gt:.6f}  (target: ~0)")
        print()

    results = aind_decomposition_cca(
        Phi, Psi_plus,
        n_terms=n_terms,
        num_epochs=num_epochs,
        lr=lr,
        lambda_cca=lambda_cca,
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
        print("Final Evaluation Metrics  (CCA-in-Latent aIND)")
        print("=" * 60)
        print(f"  Canonical correlation rho              : {results['rho']:+.6f}")
        print(f"  I(Phi;Psi) ~= -0.5*log(1-rho^2)       : {results['mi_estimate']:.6f} nats")
        print(f"  I(Phi_I ; Phi_R)                       : {metrics['mutual_information']:.6f}")
        print(f"  Residual energy ||Phi - Phi_I||^2      : {metrics['residual_energy']:.6f}")
        print(f"  MI(Phi_I , Psi+)                       : {metrics['mi_phiI_psi']:.6f}")
        print(f"  MI(Phi_R , Psi+)                       : {metrics['mi_phiR_psi']:.6f}")
        print(f"  GT error ||Phi_I - Phi_I_true||^2      : {metrics['gt_error']:.6f}")
        print("=" * 60)

    path = os.path.join(output_dir, "aind_cca_metrics.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("aIND Metrics  (CCA-in-Latent)\n")
        fh.write(f"lambda_cca={lambda_cca} | n_terms={n_terms} | epochs={num_epochs}\n")
        fh.write(f"rho = {results['rho']:+.6f}\n")
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
    print(f"Metrics saved to: {path}")

    visualize_latent_space(
        results["z_phi"], results["z_psi"],
        results["z_phi_I"], results["z_phi_R"],
        rho=results["rho"],
        output_dir=output_dir, prefix="aind_cca",
    )
    visualize_training(
        results["loss_history"],
        lambda_cca=lambda_cca,
        output_dir=output_dir, prefix="aind_cca",
    )
    visualize_decomposition(
        Phi, Psi_plus, Phi_I, Phi_R,
        Phi_I_true, Phi_R_true,
        output_dir=output_dir, prefix="aind_cca",
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
        lambda_cca=1.0,
        output_dir="results_cca",
        seed=1,
        verbose=True,
    )
