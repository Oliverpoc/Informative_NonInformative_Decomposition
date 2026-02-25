"""
aIND (approximate Informative-Non-Informative Decomposition) - Toy v2

Source field: Phi = c^2 + cf + cg + fg = (c+f)(c+g)

Ground truth decomposition:
    Phi_I_true = f(c+g) = cf + fg   (f-containing terms)
    Phi_R_true = c(c+g) = c^2 + cg  (f-free terms)

Target: Psi_plus = 0.5*f^2 - 0.2*f + epsilon

Everything else (loss, training loop, architecture) is identical to
aind_decomposition.py. Only the data source and visualization/evaluation
are adapted for the v2 toy problem.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error

from aIND.IND import dsf, mikde
from toy_settings_v2 import (
    phi_func,
    phi_informative_true,
    phi_residual_true,
    psi_plus_func,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TwoBranchDSF(nn.Module):
    """
    Two DSF branches B1, B2 with either hard selection (Eq. A7)
    or soft gating (Eq. A10) using the *true* Phi.

    Forward signature:
        Phi_I, Phi_R = model(Psi, Phi)

    Psi and Phi are 1D tensors with the same length N.
    """

    def __init__(
        self,
        num_layers: int = 5,
        num_neurons: int = 16,
        gating: str = "hard",
        k: float = 10.0,
        p: float = 0.99,
    ):
        super().__init__()
        self.B1 = dsf.DSF(num_layers=num_layers, num_neurons=num_neurons)
        self.B2 = dsf.DSF(num_layers=num_layers, num_neurons=num_neurons)
        self.gating = gating
        self.k = k
        self.p = p

    def forward(self, psi: torch.Tensor, phi: torch.Tensor):
        assert psi.shape == phi.shape
        psi_in = psi.view(1, -1)
        phi1 = self.B1(psi_in).view(-1)
        phi2 = self.B2(psi_in).view(-1)

        if self.gating == "hard":
            err1 = (phi - phi1) ** 2
            err2 = (phi - phi2) ** 2
            use1 = err1 <= err2
            phi_I = torch.where(use1, phi1, phi2)
        else:
            self.r = nn.Parameter(torch.tensor(0.0, device=phi.device, dtype=phi.dtype))
            sigma = torch.sigmoid
            k, p = self.k, self.p
            offset = torch.log(torch.tensor(p / (1.0 - p), device=phi.device, dtype=phi.dtype)) / k
            r_tilde = self.r + offset
            m1 = sigma(k * (r_tilde - phi))
            m2 = sigma(k * (phi - r_tilde))
            phi_I = m1 * phi1 + m2 * phi2

        phi_R = phi - phi_I
        return phi_I, phi_R


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class InfoDecompLoss:
    """
    L = I(Phi_R; Phi_I) + gamma * ||Phi - Phi_I||^2

    MI is computed on standardized variables via KDE.
    """

    def __init__(self, mi_obj: mikde.MutualInformation, gamma: float) -> None:
        self.mi_obj = mi_obj
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction="mean")

    def __call__(
        self,
        phi_I: torch.Tensor,
        phi_R: torch.Tensor,
        phi_true: torch.Tensor,
    ):
        phi_I_std = (phi_I - phi_I.mean()) / (phi_I.std() + 1e-8)
        phi_R_std = (phi_R - phi_R.mean()) / (phi_R.std() + 1e-8)

        mi = self.mi_obj(phi_I_std.view(1, -1), phi_R_std.view(1, -1))
        mse = self.mse(phi_I, phi_true)
        loss = mi + self.gamma * mse
        return loss, mi, mse


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def aind_decomposition(
    Phi,
    Psi_plus,
    num_epochs=5000,
    lr=1e-3,
    gamma=1.0,
    device=None,
    verbose=True,
    seed=42,
    num_layers=5,
    num_neurons=16,
    mi_num_bins=50,
    mi_sigma="scott",
    mi_normalize=True,
):
    """
    Perform aIND decomposition: Phi = Phi_I + Phi_R.

    Objective:
        L = I(Phi_R; Phi_I) + gamma * ||Phi - Phi_I||^2
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

    Phi_t = torch.tensor(Phi_flat, dtype=torch.float32, device=device)
    Psi_t = torch.tensor(Psi_flat, dtype=torch.float32, device=device)

    model = TwoBranchDSF(
        num_layers=num_layers, num_neurons=num_neurons, gating="hard", k=500, p=0.99
    ).to(device)

    mi_obj = mikde.MutualInformation(
        num_bins=mi_num_bins,
        sigma=mi_sigma,
        normalize=mi_normalize,
        device=device,
    )
    criterion = InfoDecompLoss(mi_obj=mi_obj, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = {"total": [], "mi": [], "residual_energy": []}

    for epoch in range(1, num_epochs + 1):
        model.train()

        phi_I_tr, phi_R_tr = model(Psi_t, Phi_t)
        loss_tr, mi_tr, mse_tr = criterion(phi_I_tr, phi_R_tr, Phi_t)

        optimizer.zero_grad()
        loss_tr.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == num_epochs:
            loss_history["total"].append(loss_tr.item())
            loss_history["mi"].append(mi_tr.item())
            loss_history["residual_energy"].append(mse_tr.item())

            if verbose:
                print(
                    f"[Epoch {epoch:5d}] "
                    f"loss={loss_tr.item():.6f}  "
                    f"(MI={mi_tr.item():.4f}, MSE={mse_tr.item():.4f})"
                )

    model.eval()
    with torch.no_grad():
        phi_I_all, phi_R_all = model(Psi_t, Phi_t)

    Phi_I_final = phi_I_all.cpu().numpy().flatten()
    Phi_R_final = Phi_flat - Phi_I_final

    return {
        "Phi_I": Phi_I_final,
        "Phi_R": Phi_R_final,
        "loss_history": loss_history,
        "model": model,
        "mi_obj": mi_obj,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_decomposition(Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true=None):
    """
    Evaluate the quality of the v2 aIND decomposition.

    Metrics
    -------
    mutual_information   : I(Phi_I; Phi_R) via sklearn  (lower = more independent)
    residual_energy      : MSE(Phi, Phi_I)              (lower = smaller residual)
    mi_phiI_psi          : MI(Phi_I, Psi_plus)          (higher = Phi_I is informative)
    mi_phiR_psi          : MI(Phi_R, Psi_plus)          (lower  = Phi_R is non-informative)
    gt_error             : MSE(Phi_I, Phi_I_true)       (lower = closer to ground truth)
    """
    metrics = {}

    # I(Phi_I; Phi_R)
    metrics["mutual_information"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Phi_I, random_state=42
    )[0]

    # Residual energy
    metrics["residual_energy"] = mean_squared_error(Phi, Phi_I)

    # MI(Phi_I; Psi_plus) — informative component should be high
    metrics["mi_phiI_psi"] = mutual_info_regression(
        Phi_I.reshape(-1, 1), Psi_plus, random_state=42
    )[0]

    # MI(Phi_R; Psi_plus) — residual should be low
    metrics["mi_phiR_psi"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Psi_plus, random_state=42
    )[0]

    # Ground-truth comparison
    if Phi_I_true is not None:
        metrics["gt_error"] = mean_squared_error(Phi_I_true, Phi_I)
    else:
        metrics["gt_error"] = float("nan")

    metrics["total_loss"] = (
        metrics["mutual_information"] + metrics["residual_energy"]
    )

    return metrics


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_decomposition(
    Phi,
    Psi_plus,
    Phi_I,
    Phi_R,
    Phi_I_true,
    Phi_R_true,
    output_dir="results_v2",
    prefix="aind_v2",
):
    """
    Visualize aIND decomposition for toy v2 (Phi = c^2+cf+cg+fg).
    Plotting style mirrors aind_decomposition.py exactly.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reshape if needed
    if Phi_I.ndim == 1:
        nx, ny = Phi.shape
        Phi_I = Phi_I.reshape(nx, ny)
        Phi_R = Phi_R.reshape(nx, ny)

    # Figure 1: Field comparison (2 x 3 layout)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = [
        "\u03a6 (Source Field)\n\u03a6 = c\u00b2+cf+cg+fg",
        "\u03a6\u1d35_true = f(c+g)\n(True Informative)",
        "\u03a6\u1d35 (Reconstructed Informative)",
        "\u03a8\u207a (Target Field)\n\u03a8\u207a = 0.5f\u00b2 - 0.2f + \u03b5",
        "\u03a6\u1d3f_true = c(c+g)\n(True Residual)",
        "\u03a6\u1d3f (Reconstructed Residual)\n\u03a6\u1d3f = \u03a6 - \u03a6\u1d35",
    ]
    fields = [Phi, Phi_I_true, Phi_I, Psi_plus, Phi_R_true, Phi_R]

    vabs = max(np.abs(f).max() for f in fields)

    for ax, field, title in zip(axes.flat, fields, titles):
        im = ax.imshow(field, origin="lower", cmap="RdBu", extent=[0, 1, 0, 2], aspect=0.5,
                       vmin=-vabs, vmax=vabs)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Field value")
    fig.tight_layout()
    fields_path = os.path.join(output_dir, f"{prefix}_fields.png")
    fig.savefig(fields_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Field visualization saved to: {fields_path}")

    # Figure 2: Scatter plots
    Phi_flat       = Phi.ravel()
    Psi_flat       = Psi_plus.ravel()
    Phi_I_flat     = Phi_I.ravel()
    Phi_R_flat     = Phi_R.ravel()
    Phi_I_true_flat = Phi_I_true.ravel()

    # Subsample for plotting
    n_plot = min(5000, len(Phi_flat))
    idx = np.random.choice(len(Phi_flat), n_plot, replace=False)

    plt.figure(figsize=(15, 5))

    # Subplot 1: Mapping Phi_I -> Psi+
    plt.subplot(1, 3, 1)
    plt.scatter(Phi_I_flat[idx], Psi_flat[idx], s=6, alpha=0.5, label="Data")
    plt.xlabel("\u03a6\u1d35 (Informative Component)")
    plt.ylabel("\u03a8\u207a (Target Field)")
    plt.title("Mapping: \u03a8\u207a \u2248 F(\u03a6\u1d35)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Independence Phi_I vs Phi_R
    plt.subplot(1, 3, 2)
    plt.scatter(Phi_I_flat[idx], Phi_R_flat[idx], s=6, alpha=0.5, color="red")
    plt.xlabel("\u03a6\u1d35 (Informative Component)")
    plt.ylabel("\u03a6\u1d3f (Residual Component)")
    plt.title("Independence: I(\u03a6\u1d35; \u03a6\u1d3f) should be minimized")
    plt.grid(True, alpha=0.3)

    # Subplot 3: Comparison with ground truth Phi_I_true
    plt.subplot(1, 3, 3)
    plt.scatter(Phi_I_true_flat[idx], Phi_I_flat[idx], s=6, alpha=0.5, color="green")
    phi_I_true_range = np.linspace(Phi_I_true_flat.min(), Phi_I_true_flat.max(), 100)
    plt.plot(phi_I_true_range, phi_I_true_range, 'k--', linewidth=2,
             label="Perfect: \u03a6\u1d35 = \u03a6\u1d35_true")
    plt.xlabel("\u03a6\u1d35_true = f(c+g) (True Informative)")
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
    nx=100,
    ny=100,
    t=0.0,
    num_epochs=1000,
    lr=7e-3,
    gamma=1.0,
    output_dir="results_v2",
    seed=1,
    verbose=True,
):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # --- Build grid ---
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    # --- Generate fields ---
    Phi        = phi_func(X, Y, t)
    Psi_plus   = psi_plus_func(X, Y, t, deltaT=0.0, noise_std=1e-8)
    Phi_I_true = phi_informative_true(X, Y, t)   # f(c+g)
    Phi_R_true = phi_residual_true(X, Y, t)       # c(c+g)

    if verbose:
        print("=" * 60)
        print("aIND Decomposition  —  Toy v2:  Phi = c^2+cf+cg+fg")
        print("=" * 60)
        print(f"Grid: {nx} x {ny}   t={t}   epochs={num_epochs}")
        print()

    # --- Run decomposition ---
    results = aind_decomposition(
        Phi,
        Psi_plus,
        num_epochs=num_epochs,
        lr=lr,
        gamma=gamma,
        verbose=verbose,
        seed=seed,
    )

    Phi_I = results["Phi_I"].reshape(Phi.shape)
    Phi_R = results["Phi_R"].reshape(Phi.shape)

    # --- Evaluate ---
    metrics = evaluate_decomposition(
        Phi.ravel(),
        Psi_plus.ravel(),
        results["Phi_I"],
        results["Phi_R"],
        Phi_I_true=Phi_I_true.ravel(),
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Final Evaluation Metrics")
        print("=" * 60)
        print(f"  I(Phi_I ; Phi_R)          : {metrics['mutual_information']:.6f}  (lower = more independent)")
        print(f"  Residual energy ||Phi-Phi_I||^2 : {metrics['residual_energy']:.6f}")
        print(f"  MI(Phi_I , Psi+)          : {metrics['mi_phiI_psi']:.6f}  (higher = Phi_I is informative)")
        print(f"  MI(Phi_R , Psi+)          : {metrics['mi_phiR_psi']:.6f}  (lower  = Phi_R is not informative)")
        print(f"  GT error ||Phi_I - Phi_I_true||^2 : {metrics['gt_error']:.6f}")
        print(f"  Total loss                : {metrics['total_loss']:.6f}")
        print("=" * 60)

    # --- Save metrics ---
    metrics_path = os.path.join(output_dir, "aind_v2_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write("aIND Decomposition Metrics  (Toy v2: Phi = c^2+cf+cg+fg)\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"I(Phi_I ; Phi_R)                   : {metrics['mutual_information']:.6f}\n")
        fh.write(f"Residual energy ||Phi - Phi_I||^2  : {metrics['residual_energy']:.6f}\n")
        fh.write(f"MI(Phi_I , Psi+)                   : {metrics['mi_phiI_psi']:.6f}\n")
        fh.write(f"MI(Phi_R , Psi+)                   : {metrics['mi_phiR_psi']:.6f}\n")
        fh.write(f"GT error ||Phi_I - Phi_I_true||^2  : {metrics['gt_error']:.6f}\n")
        fh.write(f"Total loss                         : {metrics['total_loss']:.6f}\n")
    print(f"Metrics saved to: {metrics_path}")

    # --- Visualise ---
    visualize_decomposition(
        Phi,
        Psi_plus,
        Phi_I,
        Phi_R,
        Phi_I_true,
        Phi_R_true,
        output_dir=output_dir,
        prefix="aind_v2",
    )

    # Plot training dynamics
    if results["loss_history"]["total"]:
        plt.figure(figsize=(12, 4))

        epochs_plot = np.arange(0, len(results["loss_history"]["total"])) * 100

        plt.subplot(1, 2, 1)
        plt.plot(epochs_plot, results["loss_history"]["total"], label="Val Loss")
        plt.plot(epochs_plot, results["loss_history"]["mi"], label="MI Term (val)")
        plt.plot(epochs_plot, results["loss_history"]["residual_energy"], label="Residual Energy (val)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Dynamics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_plot, results["loss_history"]["mi"], label="I(\u03a6\u1d35; \u03a6\u1d3f)")
        plt.xlabel("Epoch")
        plt.ylabel("Mutual Information")
        plt.title("Mutual Information Minimization")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        training_path = os.path.join(output_dir, "aind_v2_training.png")
        plt.savefig(training_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training dynamics saved to: {training_path}")

    return results, metrics


if __name__ == "__main__":
    main(
        nx=100,
        ny=100,
        t=0.0,
        num_epochs=1000,
        lr=7e-3,
        gamma=1.0,
        output_dir="results_v2",
        seed=1,
        verbose=True,
    )
