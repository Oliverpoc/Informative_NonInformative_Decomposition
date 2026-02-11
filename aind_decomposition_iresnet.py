"""
aIND (approximate Informative–Non-Informative Decomposition) Implementation

This variant uses an invertible ResNet (IResNet1D) instead of DSF branches.
Everything else (loss, training loop, evaluation, visualization, main entry)
is kept as close as possible to `aind_decomposition.py`.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
from torch.nn.utils import spectral_norm

from aIND.IND import mikde
from toy_settings import (
    phi_func,
    psi_plus_func,
    f_func,
    g_func,
    F_analytical,
)


class ElementwiseIResBlock(nn.Module):
    """
    Truly element-wise invertible ResNet block with spectral normalization:
        y = x + α · g(x)
    where g(x) is a nonlinear function applied independently to each element.

    Parameters
    ----------
    alpha : float
        Contraction factor (0 < alpha < 1). Pick ≤0.97 for fast convergence.
    hidden_dim : int
        Dimension of hidden layers in the nonlinear transformation.
    iters_res : int
        Fixed-point iterations when calling inverse().
    """

    def __init__(self, alpha: float = 0.9, hidden_dim: int = 128, iters_res: int = 10):
        super().__init__()
        self.alpha = alpha
        self.iters_res = iters_res

        # Element-wise nonlinear transformation with spectral norm
        self.g = nn.Sequential(
            spectral_norm(nn.Linear(1, hidden_dim)),
            nn.ELU(inplace=True),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ELU(inplace=True),
            spectral_norm(nn.Linear(hidden_dim, 1)),
        )

        # Initialize to ensure Lipschitz constant < 1
        with torch.no_grad():
            for m in self.g.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data *= 0.9  # Ensure spectral norm starts < 1

    def g_elementwise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply g to each element independently."""
        shape = x.shape
        # Reshape to (N, 1) where N is total number of elements
        x_flat = x.reshape(-1, 1)
        # Apply g
        out = self.g(x_flat)
        # Restore original shape
        return out.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.g_elementwise(x)
        return x + self.alpha * gx

    def inverse(self, y: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        """Solve x s.t. y = x + α g(x) via simple fixed-point iteration."""
        x = y 
        for _ in range(self.iters_res):
            x_prev = x
            x = y - self.alpha * self.g_elementwise(x)
            if (x - x_prev).abs().max() < tol:
                break
        return x


class IResNetStack(nn.Module):
    """
    Stack of multiple ElementwiseIResBlock modules forming a 1D invertible
    mapping. This represents one monotone branch of the parabola inverse.
    """

    def __init__(self, num_blocks: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ElementwiseIResBlock(alpha=0.8, hidden_dim=hidden_dim)
                for _ in range(num_blocks)
            ]
        )
        # Learnable affine to match scales/shifts
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is 1D: Psi samples; apply elementwise mapping shared across coordinates
        z = x.view(-1, 1)
        for block in self.blocks:
            z = block(z)
        z = z * self.scale + self.bias
        return z.view_as(x)


class BijectiveSelectorNet_IResNet(nn.Module):
    """
    Two-branch selector built from IResNetStack.

    Given Ψ⁺, produces two candidate Φᴵ values corresponding to the two
    monotone branches of the quadratic inverse.
    """

    def __init__(self, num_blocks: int = 5, hidden_dim: int = 64):
        super().__init__()
        # Branch 1: e.g., f below the vertex
        self.branch1 = IResNetStack(num_blocks=num_blocks, hidden_dim=hidden_dim)
        # Branch 2: e.g., f above the vertex
        self.branch2 = IResNetStack(num_blocks=num_blocks, hidden_dim=hidden_dim)

    def forward(self, psi: torch.Tensor):
        # psi comes in as [N]; we apply elementwise shared mapping
        psi_flat = psi.view(-1)
        out1 = self.branch1(psi_flat)
        out2 = self.branch2(psi_flat)
        return out1, out2


class TwoBranchIResNet(nn.Module):
    """
    Two IResNet1D branches B1, B2 with either hard selection (Eq. A7)
    or soft gating (Eq. A10) using the *true* Φ.

    Forward signature:
        Phi_I, Phi_R = model(Psi, Phi)

    Psi and Phi are 1D tensors with the same length N.
    """

    def __init__(
        self,
        input_dim: int,
        gating: str = "hard",  # "hard" (Eq. A7) or "soft" (Eq. A10)
        k: float = 10.0, 
        p: float = 0.99,  
        alpha: float = 0.9,
        num_blocks: int = 3,
        keep_freqs: int = 10,
        hidden_dim: int = 128,
        iters_res: int = 10,
    ):
        super().__init__()
        # Use two IResNetStack branches as bijective candidates
        self.selector = BijectiveSelectorNet_IResNet(
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
        )
        self.gating = gating
        self.k = k
        self.p = p

    def forward(self, psi: torch.Tensor, phi: torch.Tensor):
        assert psi.shape == phi.shape
        # Work on flattened representation; selection is elementwise
        psi_flat = psi.view(-1)
        phi_flat = phi.view(-1)

        phi1 = self.selector.branch1(psi_flat)
        phi2 = self.selector.branch2(psi_flat)

        if self.gating == "hard":
            # Hard selection (paper Eq. A7)
            err1 = (phi_flat - phi1) ** 2
            err2 = (phi_flat - phi2) ** 2
            use1 = err1 <= err2
            phi_I_flat = torch.where(use1, phi1, phi2)
        else:
            # Soft gating (paper Eq. A10, adapted for 2 branches)
            # Estimate single boundary r in Φ-space between branches
            self.r = nn.Parameter(
                torch.tensor(0.0, device=phi_flat.device, dtype=phi_flat.dtype)
            )
            sigma = torch.sigmoid
            k, p = self.k, self.p

            # shifted boundary \tilde{r} so masks give ~p at the boundary
            offset = torch.log(
                torch.tensor(p / (1.0 - p), device=phi_flat.device, dtype=phi_flat.dtype)
            ) / k
            r_tilde = self.r + offset

            # smooth masks in Φ-space
            m1 = sigma(k * (r_tilde - phi_flat))  # ~1 if Φ < r, ~0 if Φ > r
            m2 = sigma(k * (phi_flat - r_tilde))  # ~0 if Φ < r, ~1 if Φ > r

            # union of bijections: B∪(Ψ+) = m1*B1(Ψ+) + m2*B2(Ψ+)
            phi_I_flat = m1 * phi1 + m2 * phi2

        phi_I = phi_I_flat.view_as(phi)
        phi_R = phi - phi_I
        return phi_I, phi_R


class InfoDecompLoss:
    """
    Implements:
        L = I(Φᴿ ; Φᴵ) + γ * ||Φ - Φᴵ||²

    MI is computed via a mikde.MutualInformation object on standardized variables.
    """

    def __init__(
        self,
        mi_obj: mikde.MutualInformation,
        gamma: float,
    ) -> None:
        self.mi_obj = mi_obj
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction="mean")

    def __call__(
        self,
        phi_I: torch.Tensor,
        phi_R: torch.Tensor,
        phi_true: torch.Tensor,
    ):
        # Standardize for MI estimation
        phi_I_std = (phi_I - phi_I.mean()) / (phi_I.std() + 1e-8)
        phi_R_std = (phi_R - phi_R.mean()) / (phi_R.std() + 1e-8)

        mi = self.mi_obj(
            phi_I_std.view(1, -1),
            phi_R_std.view(1, -1),
        )

        mse = self.mse(phi_I, phi_true)
        loss = mi + self.gamma * mse
        return loss, mi, mse


def aind_decomposition(
    Phi,
    Psi_plus,
    num_epochs=100,
    lr=1e-3,
    gamma=1.0,
    device=None,
    verbose=True,
    seed=42,
    # IResNet / MI hyperparameters
    alpha=0.9,
    num_blocks=3,
    keep_freqs=10,
    hidden_dim=128,
    iters_res=10,
    mi_num_bins=50,
    mi_sigma="scott",
    mi_normalize=True,
):
    """
    Perform aIND decomposition: Φ = Φᴵ + Φᴿ using TwoBranchIResNet.

    Joint optimization objective (two-branch IResNet, hard selection):
        L = I(Φᴿ; Φᴵ) + γ * ||Φ - Φᴵ||²

    Parameters
    ----------
    Phi : np.ndarray
        Source field Φ, shape [nx, ny] or flattened
    Psi_plus : np.ndarray
        Target field Ψ⁺, shape [nx, ny] or flattened
    num_epochs : int
        Number of training epochs
    lr : float
        Learning rate
    gamma : float
        Weight for residual energy term
    device : torch.device
        Device to run on (None for auto-detection)
    verbose : bool
        Whether to print training progress
    seed : int
        Random seed

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'Phi_I': Reconstructed informative component
        - 'Phi_R': Reconstructed residual component
        - 'loss_history': Training loss history
        - 'model': Trained two-branch IResNet model
        - 'mi_obj': Trained MI estimator
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Flatten and convert to tensors
    Phi_flat = Phi.ravel()
    Psi_flat = Psi_plus.ravel()
    N = len(Phi_flat)

    # Use 1D tensors for compatibility
    Phi_t = torch.tensor(Phi_flat, dtype=torch.float32, device=device)
    Psi_t = torch.tensor(Psi_flat, dtype=torch.float32, device=device)

    # Use full data for training only (no validation split)
    psi_tr = Psi_t
    phi_tr = Phi_t

    # Initialize model and loss
    model = TwoBranchIResNet(
        input_dim=N,
        gating="hard",
        k=500,
        p=0.99,
        alpha=alpha,
        num_blocks=num_blocks,
        keep_freqs=keep_freqs,
        hidden_dim=hidden_dim,
        iters_res=iters_res,
    ).to(device)

    mi_obj = mikde.MutualInformation(
        num_bins=mi_num_bins,
        sigma=mi_sigma,
        normalize=mi_normalize,
        device=device,
    )
    criterion = InfoDecompLoss(mi_obj=mi_obj, gamma=gamma)

    # Optimizer (no validation-based scheduler)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training history
    loss_history = {
        "total": [],
        "mi": [],
        "residual_energy": [],
    }

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()

        phi_I_tr, phi_R_tr = model(psi_tr, phi_tr)
        loss_tr, mi_tr, mse_tr = criterion(phi_I_tr, phi_R_tr, phi_tr)

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
                    f"loss={loss_tr.item():.6f} (MI={mi_tr.item():.4f}, MSE={mse_tr.item():.4f})"
                )

    # Final evaluation on the full set
    model.eval()
    with torch.no_grad():
        phi_I_all, phi_R_all = model(Psi_t, Phi_t)

    Phi_I_final = phi_I_all.cpu().numpy().flatten()
    Phi_R_final = Phi_flat - Phi_I_final

    results = {
        "Phi_I": Phi_I_final,
        "Phi_R": Phi_R_final,
        "loss_history": loss_history,
        "model": model,
        "mi_obj": mi_obj,
    }

    return results


def evaluate_decomposition(
    Phi,
    Psi_plus,
    Phi_I,
    Phi_R,
    mi_obj: mikde.MutualInformation | None = None,
):
    """
    Evaluate the quality of aIND decomposition.

    Parameters
    ----------
    Phi : np.ndarray
        Original source field (flattened)
    Psi_plus : np.ndarray
        Target field (flattened)
    Phi_I : np.ndarray
        Reconstructed informative component (flattened)
    Phi_R : np.ndarray
        Reconstructed residual component (flattened)

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    metrics = {}

    # Mutual information I(Φᴵ; Φᴿ) using sklearn
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.metrics import mean_squared_error

    mi_sklearn = mutual_info_regression(
        Phi_R.reshape(-1, 1),
        Phi_I,
        random_state=42,
    )[0]
    metrics["mutual_information"] = mi_sklearn

    # Residual energy ||Φ - Φᴵ||²
    residual_energy = mean_squared_error(Phi, Phi_I)
    metrics["residual_energy"] = residual_energy

    # Mapping error ||Ψ⁺ - F(Φᴵ)||² using analytical mapping
    Psi_pred = F_analytical(Phi_I)
    mapping_error = mean_squared_error(Psi_plus, Psi_pred)
    metrics["mapping_error"] = mapping_error

    # Total loss
    metrics["total_loss"] = mi_sklearn + residual_energy + mapping_error

    return metrics


def visualize_decomposition(
    Phi,
    Psi_plus,
    Phi_I,
    Phi_R,
    f_true,
    g_true,
    output_dir: str = "results",
    prefix: str = "aind_iresnet",
):
    """
    Visualize aIND decomposition results.
    Mostly identical to the visualization in `aind_decomposition.py`.
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Reshape if needed
    if Phi_I.ndim == 1:
        nx, ny = Phi.shape
        Phi_I = Phi_I.reshape(nx, ny)
        Phi_R = Phi_R.reshape(nx, ny)

    # Figure 1: Field comparison (2 x 3 layout)
    plt.figure(figsize=(14, 8))
    titles = [
        "Φ (Source Field)\nΦ = f + g",
        "Ψ⁺ (Target Field)\nΨ⁺ = 0.5f² - 0.2f + ε",
        "Φᴵ (Reconstructed Informative)",
        "Φᴿ (Reconstructed Residual)\nΦᴿ = Φ - Φᴵ",
        "f (True Informative)",
        "g (True Residual)",
    ]
    fields = [Phi, Psi_plus, Phi_I, Phi_R, f_true, g_true]

    for i, (field, title) in enumerate(zip(fields, titles), 1):
        plt.subplot(2, 3, i)
        im = plt.imshow(
            field,
            origin="lower",
            cmap="RdBu",
            extent=[0, 1, 0, 2],
            aspect=0.5,
        )
        plt.title(title, fontsize=10)
        plt.xlabel("x")
        plt.ylabel("y")

    plt.tight_layout()
    fields_path = os.path.join(output_dir, f"{prefix}_fields.png")
    plt.savefig(fields_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Field visualization saved to: {fields_path}")

    # Figure 2: Scatter plots
    Phi_flat = Phi.ravel()
    Psi_flat = Psi_plus.ravel()
    Phi_I_flat = Phi_I.ravel()
    Phi_R_flat = Phi_R.ravel()
    f_flat = f_true.ravel()
    g_flat = g_true.ravel()

    # Subsample for plotting
    n_plot = min(5000, len(Phi_flat))
    idx = np.random.choice(len(Phi_flat), n_plot, replace=False)

    plt.figure(figsize=(15, 5))

    # Subplot 1: Mapping Φᴵ → Ψ⁺
    plt.subplot(1, 3, 1)
    plt.scatter(Phi_I_flat[idx], Psi_flat[idx], s=6, alpha=0.5, label="Data")
    # Overlay analytical mapping
    phi_I_range = np.linspace(Phi_I_flat.min(), Phi_I_flat.max(), 200)
    psi_analytical = F_analytical(phi_I_range)
    plt.plot(
        phi_I_range,
        psi_analytical,
        "r--",
        linewidth=2,
        label="F(Φᴵ) = 0.5Φᴵ² - 0.2Φᴵ",
    )
    plt.xlabel("Φᴵ (Informative Component)")
    plt.ylabel("Ψ⁺ (Target Field)")
    plt.title("Mapping: Ψ⁺ ≈ F(Φᴵ)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Independence Φᴵ vs Φᴿ
    plt.subplot(1, 3, 2)
    plt.scatter(Phi_I_flat[idx], Phi_R_flat[idx], s=6, alpha=0.5, color="red")
    plt.xlabel("Φᴵ (Informative Component)")
    plt.ylabel("Φᴿ (Residual Component)")
    plt.title("Independence: I(Φᴵ; Φᴿ) should be minimized")
    plt.grid(True, alpha=0.3)

    # Subplot 3: Comparison with true f
    plt.subplot(1, 3, 3)
    plt.scatter(f_flat[idx], Phi_I_flat[idx], s=6, alpha=0.5, color="green")
    # Perfect reconstruction line
    f_range = np.linspace(f_flat.min(), f_flat.max(), 100)
    plt.plot(
        f_range,
        f_range,
        "k--",
        linewidth=2,
        label="Perfect: Φᴵ = f",
    )
    plt.xlabel("f (True Informative Component)")
    plt.ylabel("Φᴵ (Reconstructed Informative)")
    plt.title("Reconstruction Quality")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"{prefix}_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plots saved to: {scatter_path}")


def main(
    nx=100,
    ny=100,
    t=0.0,
    num_epochs=2000,
    lr=5e-3,
    gamma=1e-3,
    output_dir="results",
    seed=1,
    verbose=True,
):
    """
    Main function to run aIND decomposition on toy problem using IResNet.
    Mirrors the `main` in `aind_decomposition.py`.
    """
    import matplotlib.pyplot as plt

    # Set random seed
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate grid and fields
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    # Generate fields
    Phi = phi_func(X, Y, t)
    # For the quadratic toy problem we use deltaT = 0
    Psi_plus = psi_plus_func(X, Y, t, deltaT=0.0, noise_std=1e-8)
    f_true = f_func(X, Y, t)
    g_true = g_func(X, Y, t)

    if verbose:
        print("=" * 60)
        print("aIND Decomposition (IResNet)")
        print("=" * 60)
        print(f"Grid size: {nx} x {ny}")
        print(f"Time: t = {t}")
        print(f"Training epochs: {num_epochs}")
        print()

    # Perform aIND decomposition
    results = aind_decomposition(
        Phi,
        Psi_plus,
        num_epochs=num_epochs,
        lr=lr,
        gamma=gamma,
        verbose=verbose,
        seed=seed,
    )

    # Reshape results
    Phi_I = results["Phi_I"].reshape(Phi.shape)
    Phi_R = results["Phi_R"].reshape(Phi.shape)

    # Evaluate
    metrics = evaluate_decomposition(
        Phi.ravel(),
        Psi_plus.ravel(),
        results["Phi_I"],
        results["Phi_R"],
        mi_obj=results.get("mi_obj"),
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Final Evaluation Metrics (IResNet)")
        print("=" * 60)
        print(f"Mutual Information I(Φᴵ; Φᴿ): {metrics['mutual_information']:.6f}")
        print(f"Residual Energy ||Φ - Φᴵ||²: {metrics['residual_energy']:.6f}")
        print(f"Mapping Error ||Ψ⁺ - F(Φᴵ)||²: {metrics['mapping_error']:.6f}")
        print(f"Total Loss: {metrics['total_loss']:.6f}")
        print("=" * 60)

    # Save metrics
    metrics_path = os.path.join(output_dir, "aind_iresnet_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("aIND Decomposition Metrics (IResNet)\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"Mutual Information I(Φᴵ; Φᴿ): {metrics['mutual_information']:.6f}\n"
        )
        f.write(
            f"Residual Energy ||Φ - Φᴵ||²: {metrics['residual_energy']:.6f}\n"
        )
        f.write(
            f"Mapping Error ||Ψ⁺ - F(Φᴵ)||²: {metrics['mapping_error']:.6f}\n"
        )
        f.write(f"Total Loss: {metrics['total_loss']:.6f}\n")

    # Visualize
    visualize_decomposition(
        Phi,
        Psi_plus,
        Phi_I,
        Phi_R,
        f_true,
        g_true,
        output_dir=output_dir,
        prefix="aind_iresnet",
    )

    # Plot training dynamics
    if results["loss_history"]["total"]:
        plt.figure(figsize=(12, 4))

        epochs_plot = np.arange(0, len(results["loss_history"]["total"])) * 100

        plt.subplot(1, 2, 1)
        plt.plot(epochs_plot, results["loss_history"]["total"], label="Val Loss")
        plt.plot(epochs_plot, results["loss_history"]["mi"], label="MI Term (val)")
        plt.plot(
            epochs_plot,
            results["loss_history"]["residual_energy"],
            label="Residual Energy (val)",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Dynamics (IResNet)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_plot, results["loss_history"]["mi"], label="I(Φᴵ; Φᴿ)")
        plt.xlabel("Epoch")
        plt.ylabel("Mutual Information")
        plt.title("Mutual Information Minimization (IResNet)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        training_path = os.path.join(output_dir, "aind_iresnet_training.png")
        plt.savefig(training_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training dynamics saved to: {training_path}")

    return results, metrics


if __name__ == "__main__":
    nx = 100
    ny = 100
    t = 0.0
    num_epochs = 1000
    lr = 3e-3
    gamma = 1
    output_dir = "results_iresnet"
    seed = 1
    verbose = True

    main(
        nx=nx,
        ny=ny,
        t=t,
        num_epochs=num_epochs,
        lr=lr,
        gamma=gamma,
        output_dir=output_dir,
        seed=seed,
        verbose=verbose,
    )

