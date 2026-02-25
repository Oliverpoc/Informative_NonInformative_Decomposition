"""
aIND with LES + Gaussianization preprocessing for toy v2.

Extended pipeline (compared to aind_decomposition_v2.py):

    Phi  ->  T(Phi)           GaussianizationTransform  (quantile -> N(0,1))
          -> G * T(Phi)       LES spatial filter in Gaussianized space
          -> LinearProjection Mori-Zwanzig linear projection onto Psi subspace
          -> T^-1(Phi_I)      invert Gaussianization back to physical space

Why linear regression (not a neural network) works here:
  - Phi = (c+f)(c+g) is non-Gaussian in the original space.
  - After quantile Gaussianization T(Phi) ~ N(0,1), the joint distribution
    (T(Phi), Psi+) is approximately Gaussian.
  - For jointly Gaussian variables, I(X;Y) = 0  <=>  Cov(X,Y) = 0.
    Therefore minimizing MI is equivalent to minimizing linear correlation,
    which is solved analytically by OLS (ordinary least squares).
  - This is exactly the Mori-Zwanzig linear projection onto the subspace
    spanned by Psi+.  No neural-network training required.
  - MI invariance under invertible maps: I(T(Phi)^I ; T(Phi)^R) =
    I(Phi^I ; Phi^R), so the theoretical optimum is unchanged.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error

from toy_settings_v2 import (
    phi_func,
    phi_informative_true,
    phi_residual_true,
    psi_plus_func,
    F_analytical,
)

# Re-use evaluation helper from v2 — no duplication
from aind_decomposition_v2 import evaluate_decomposition


# ---------------------------------------------------------------------------
# Gaussianization transform
# ---------------------------------------------------------------------------

class GaussianizationTransform:
    """
    Element-wise Gaussianization of a 2D spatial field via quantile mapping.

    Wraps sklearn.preprocessing.QuantileTransformer.  The transform is
    fitted on the *flattened* field (all grid points treated as i.i.d.
    samples) and applied element-wise, so it preserves the spatial shape.

    Usage
    -----
    T = GaussianizationTransform()
    T.fit(Phi)                       # fit on 2D field
    Phi_g  = T.transform(Phi)        # -> approximately N(0,1), same shape
    Phi_r  = T.inverse_transform(Phi_g)  # -> back to original distribution
    """

    def __init__(self, n_quantiles: int = 1000, random_state: int = 42):
        self.qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=random_state,
        )

    def fit(self, field: np.ndarray) -> "GaussianizationTransform":
        self.qt.fit(field.ravel().reshape(-1, 1))
        return self

    def transform(self, field: np.ndarray) -> np.ndarray:
        """Map field values to N(0,1).  Preserves array shape."""
        return self.qt.transform(field.ravel().reshape(-1, 1)).reshape(field.shape)

    def inverse_transform(self, field_g: np.ndarray) -> np.ndarray:
        """Invert quantile mapping back to original distribution."""
        return self.qt.inverse_transform(
            field_g.ravel().reshape(-1, 1)
        ).reshape(field_g.shape)


# ---------------------------------------------------------------------------
# LES filter
# ---------------------------------------------------------------------------

def les_filter_2d(
    field: np.ndarray,
    sigma: float,
    filter_type: str = "gaussian",
) -> np.ndarray:
    """
    Apply a 2D LES spatial low-pass filter.

    Parameters
    ----------
    field       : 2D np.ndarray
    sigma       : filter width in grid points
                  (Gaussian sigma, or half-width for box filter,
                   or cutoff wavenumber divisor for spectral)
    filter_type : 'gaussian' | 'box' | 'spectral'

    Returns
    -------
    field_resolved : 2D np.ndarray
        Large-scale (resolved) component.
    """
    if filter_type == "gaussian":
        return gaussian_filter(field, sigma=sigma)

    elif filter_type == "box":
        size = max(1, int(2 * sigma + 1))
        return uniform_filter(field, size=size)

    elif filter_type == "spectral":
        F = np.fft.rfft2(field)
        ny, nx = field.shape
        k_cut = max(1, int(min(ny, nx) / (2.0 * sigma)))
        ky = np.fft.fftfreq(ny) * ny
        kx = np.fft.rfftfreq(nx) * nx
        KX, KY = np.meshgrid(kx, ky)
        mask = (np.abs(KY) <= k_cut) & (np.abs(KX) <= k_cut)
        return np.fft.irfft2(F * mask, s=field.shape)

    else:
        raise ValueError(f"Unknown filter_type: '{filter_type}'")


# ---------------------------------------------------------------------------
# Linear projection (Mori-Zwanzig)  —  target-centric information set
# ---------------------------------------------------------------------------

class LinearProjection:
    """
    Target-centric aIND decomposition via conditional expectation (OLS).

    Information set: Omega = {Psi+}   (target-centric)

    For any field U, the decomposition is:

        U_I = E[U | Psi+]  =  mu_U  +  beta_U * (Psi+ - mu_Psi)
        U_N = U - U_I

    where  beta_U = Cov(U, Psi+) / Var(Psi+)  is the OLS coefficient.

    Under the linear-Gaussian assumption:
      - Cov(U_I, U_N) = 0  by construction (orthogonal projection)
      - For Gaussian variables: uncorrelated => independent => I(U_I; U_N) = 0
      - This is the Mori-Zwanzig linear projection onto span{Psi+}

    After Gaussianization T(Phi) ~ N(0,1), the Gaussian assumption holds
    approximately, making this the optimal aIND solution with no training.

    Usage
    -----
    proj = LinearProjection().fit(Psi_flat)     # fix the information set
    U_I, U_N, beta = proj.project(U_flat)       # decompose any field U
    """

    def fit(self, Psi_flat: np.ndarray) -> "LinearProjection":
        """Fix the information set by storing Psi+ statistics."""
        self.psi_mean = float(np.mean(Psi_flat))
        self.psi_var  = float(np.var(Psi_flat)) + 1e-8
        self._Psi_centered = Psi_flat - self.psi_mean
        return self

    def project(self, U_flat: np.ndarray):
        """
        Decompose U into informative (U_I) and non-informative (U_N) parts.

        Parameters
        ----------
        U_flat : 1D np.ndarray  (any field flattened to match Psi+ samples)

        Returns
        -------
        U_I   : informative part  E[U | Psi+]
        U_N   : non-informative residual  U - U_I
        beta  : regression coefficient Cov(U, Psi+) / Var(Psi+)
        """
        mu_U   = float(np.mean(U_flat))
        beta   = float(np.mean((U_flat - mu_U) * self._Psi_centered) / self.psi_var)
        U_I    = mu_U + beta * self._Psi_centered
        U_N    = U_flat - U_I
        return U_I, U_N, beta


# ---------------------------------------------------------------------------
# Full LES + Gaussianization → aIND pipeline
# ---------------------------------------------------------------------------

def aind_decomposition_les(
    Phi,
    Psi_plus,
    # Preprocessing
    les_sigma: float = 3.0,
    les_filter_type: str = "gaussian",
    n_quantiles: int = 1000,
    verbose: bool = True,
    seed: int = 42,
):
    """
    aIND decomposition with LES + Gaussianization + linear projection.

    Steps
    -----
    1. Gaussianize Phi:          T_Phi  = T(Phi)        via QuantileTransformer
    2. LES filter in T-space:    T_Phi_res = G * T_Phi
                                 T_Phi_sgs = T_Phi - T_Phi_res
    3. Linear projection (OLS):  T_Phi_I = alpha + beta * Psi  (Mori-Zwanzig)
                                 T_Phi_R = T_Phi_res - T_Phi_I
    4. Invert to physical space: Phi_I  = T^-1(T_Phi_I)
                                 Phi_R  = Phi - Phi_I

    Note: The SGS component T_Phi_sgs is treated as additional residual.
    The total non-informative content is T_Phi_R + T_Phi_sgs in
    Gaussianized space, which maps to Phi_R = Phi - Phi_I in physical space.

    Returns
    -------
    dict with keys:
        Phi_I, Phi_R           decomposition in physical space (flattened)
        T_Phi                  Gaussianized source field (2D)
        T_Phi_resolved         LES-filtered Gaussianized field (2D)
        T_Phi_sgs              SGS component in Gaussianized space (2D)
        T_Phi_I                informative component in Gaussianized space (2D)
        T_Phi_R                residual in LES-filtered Gaussianized space (2D)
        T_phi                  fitted GaussianizationTransform (for inspection)
        proj                   fitted LinearProjection object
    """
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Step 1: Gaussianize Phi
    # ------------------------------------------------------------------
    T_phi = GaussianizationTransform(n_quantiles=n_quantiles)
    T_phi.fit(Phi)
    T_Phi = T_phi.transform(Phi)          # shape (ny, nx), ~ N(0,1)

    if verbose:
        print(
            f"Gaussianize: Phi   mean={Phi.mean():.3f}  std={Phi.std():.3f}  "
            f"skew={float(((Phi - Phi.mean())**3).mean() / Phi.std()**3):.3f}"
        )
        print(
            f"             T(Phi) mean={T_Phi.mean():.3f}  std={T_Phi.std():.3f}  "
            f"skew={float(((T_Phi - T_Phi.mean())**3).mean() / T_Phi.std()**3):.3f}"
        )

    # ------------------------------------------------------------------
    # Step 2: LES filter in Gaussianized space
    # ------------------------------------------------------------------
    T_Phi_resolved = les_filter_2d(T_Phi, sigma=les_sigma, filter_type=les_filter_type)
    T_Phi_sgs = T_Phi - T_Phi_resolved

    if verbose:
        energy_total    = float(np.var(T_Phi))
        energy_resolved = float(np.var(T_Phi_resolved))
        print(f"LES filter (type={les_filter_type}, sigma={les_sigma}):")
        print(f"  Resolved energy fraction : {energy_resolved / energy_total:.3f}")
        print(f"  SGS    energy fraction   : {1.0 - energy_resolved / energy_total:.3f}")

    # ------------------------------------------------------------------
    # Step 3: Linear projection (Mori-Zwanzig) in Gaussianized+LES space
    # ------------------------------------------------------------------
    # Information set: Omega = {Psi+}  (target-centric)
    # For any U: U_I = E[U | Psi+] = mu_U + beta_U * (Psi+ - mu_Psi)
    #            U_N = U - U_I
    # beta_U = Cov(U, Psi+) / Var(Psi+)  -- analytical OLS, no training.
    Psi_std = (Psi_plus - Psi_plus.mean()) / (Psi_plus.std() + 1e-8)

    T_res_flat = T_Phi_resolved.ravel()
    Psi_flat   = Psi_std.ravel()

    # Fix the information set once; project any field via proj.project(U)
    proj = LinearProjection().fit(Psi_flat)
    T_Phi_I_flat, T_Phi_R_flat, beta = proj.project(T_res_flat)

    if verbose:
        corr = np.corrcoef(T_Phi_I_flat, T_Phi_R_flat)[0, 1]
        print(f"Linear projection (OLS):")
        print(f"  beta={beta:.6f}  (Cov(T_Phi_res, Psi) / Var(Psi))")
        print(f"  Corr(T_Phi_I, T_Phi_R) after projection: {corr:.6f}  (should be ~0)")

    # ------------------------------------------------------------------
    # Step 4: Invert Gaussianization -> physical space
    # ------------------------------------------------------------------
    T_Phi_I_2d = T_Phi_I_flat.reshape(Phi.shape)
    T_Phi_R_2d = T_Phi_R_flat.reshape(Phi.shape)

    Phi_I = T_phi.inverse_transform(T_Phi_I_2d)  # physical space
    Phi_R = Phi - Phi_I

    return {
        "Phi_I":          Phi_I.ravel(),
        "Phi_R":          Phi_R.ravel(),
        "T_Phi":          T_Phi,
        "T_Phi_resolved": T_Phi_resolved,
        "T_Phi_sgs":      T_Phi_sgs,
        "T_Phi_I":        T_Phi_I_2d,
        "T_Phi_R":        T_Phi_R_2d,
        "T_phi":          T_phi,
        "proj":           proj,
        "beta":           beta,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_preprocessing(
    Phi,
    T_Phi,
    T_Phi_resolved,
    T_Phi_sgs,
    output_dir: str = "results_v2_les",
    prefix: str = "aind_v2_les",
):
    """
    Figure 0: show the three stages of the LES + Gaussianization pipeline.

    Layout (1 x 4):
        Phi  |  T(Phi)  |  G*T(Phi) [resolved]  |  T(Phi)_sgs [SGS]
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(18, 4))
    titles = [
        "\u03a6 (original)\nnon-Gaussian",
        "T(\u03a6) (Gaussianized)\n\u2248 N(0,1)",
        "G\u2217T(\u03a6) (LES resolved)\nlarge scales",
        "T(\u03a6)\u2212G\u2217T(\u03a6) (SGS)\nsmall scales",
    ]
    fields = [Phi, T_Phi, T_Phi_resolved, T_Phi_sgs]

    for i, (field, title) in enumerate(zip(fields, titles), 1):
        plt.subplot(1, 4, i)
        im = plt.imshow(field, origin="lower", cmap="RdBu",
                        extent=[0, 1, 0, 2], aspect=0.5)
        plt.title(title, fontsize=10)
        plt.xlabel("x")
        plt.ylabel("y")

    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_preprocessing.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Preprocessing visualization saved to: {path}")


def visualize_decomposition(
    Phi,
    Psi_plus,
    Phi_I,
    Phi_R,
    Phi_I_true,
    Phi_R_true,
    output_dir: str = "results_v2_les",
    prefix: str = "aind_v2_les",
):
    """
    Same 2x3 field layout and 1x3 scatter layout as aind_decomposition_v2.py.

    Field layout:
        [1,1] Phi (source)          [1,2] Phi_I_true (true inf.)  [1,3] Phi_I (recon.)
        [2,1] Psi+ (target)         [2,2] Phi_R_true (true res.)  [2,3] Phi_R (recon.)
    """
    os.makedirs(output_dir, exist_ok=True)

    if Phi_I.ndim == 1:
        nx, ny = Phi.shape
        Phi_I = Phi_I.reshape(nx, ny)
        Phi_R = Phi_R.reshape(nx, ny)

    # Figure 1: Field comparison (2 x 3 layout)
    plt.figure(figsize=(14, 8))
    titles = [
        "\u03a6 (Source Field)\n\u03a6 = c\u00b2+cf+cg+fg",
        "\u03a6\u1d35_true = f(c+g)\n(True Informative)",
        "\u03a6\u1d35 (Reconstructed Informative)",
        "\u03a8\u207a (Target Field)\n\u03a8\u207a = 0.5f\u00b2 - 0.2f + \u03b5",
        "\u03a6\u1d3f_true = c(c+g)\n(True Residual)",
        "\u03a6\u1d3f (Reconstructed Residual)\n\u03a6\u1d3f = \u03a6 - \u03a6\u1d35",
    ]
    fields = [Phi, Phi_I_true, Phi_I, Psi_plus, Phi_R_true, Phi_R]

    for i, (field, title) in enumerate(zip(fields, titles), 1):
        plt.subplot(2, 3, i)
        im = plt.imshow(field, origin="lower", cmap="RdBu",
                        extent=[0, 1, 0, 2], aspect=0.5)
        plt.title(title, fontsize=10)
        plt.xlabel("x")
        plt.ylabel("y")

    plt.tight_layout()
    fields_path = os.path.join(output_dir, f"{prefix}_fields.png")
    plt.savefig(fields_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Field visualization saved to: {fields_path}")

    # Figure 2: Scatter plots
    Phi_flat        = Phi.ravel()
    Psi_flat        = Psi_plus.ravel()
    Phi_I_flat      = Phi_I.ravel()
    Phi_R_flat      = Phi_R.ravel()
    Phi_I_true_flat = Phi_I_true.ravel()

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
    plt.plot(phi_I_true_range, phi_I_true_range, "k--", linewidth=2,
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
    nx: int = 100,
    ny: int = 100,
    t: float = 0.0,
    # LES + Gaussianization hyperparameters
    les_sigma: float = 1.5,
    les_filter_type: str = "gaussian",
    n_quantiles: int = 1000,
    output_dir: str = "results_v2_les",
    seed: int = 1,
    verbose: bool = True,
):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # --- Build grid and fields ---
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    Phi        = phi_func(X, Y, t)
    Psi_plus   = psi_plus_func(X, Y, t, deltaT=0.0, noise_std=1e-8)
    Phi_I_true = phi_informative_true(X, Y, t)   # f(c+g)
    Phi_R_true = phi_residual_true(X, Y, t)       # c(c+g)

    if verbose:
        print("=" * 60)
        print("aIND  +  LES + Gaussianization  --  Toy v2")
        print("=" * 60)
        print(f"Grid: {nx} x {ny}   t={t}")
        print(f"LES:  filter_type={les_filter_type}  sigma={les_sigma}")
        print(f"Step 3: analytical OLS (Mori-Zwanzig linear projection)")
        print()

    # --- Run LES + Gaussianization -> aIND pipeline ---
    results = aind_decomposition_les(
        Phi,
        Psi_plus,
        les_sigma=les_sigma,
        les_filter_type=les_filter_type,
        n_quantiles=n_quantiles,
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
        print("Final Evaluation Metrics  (LES + Gaussianization aIND)")
        print("=" * 60)
        print(f"  I(Phi_I ; Phi_R)                   : {metrics['mutual_information']:.6f}  (lower = more independent)")
        print(f"  Residual energy ||Phi - Phi_I||^2  : {metrics['residual_energy']:.6f}")
        print(f"  MI(Phi_I , Psi+)                   : {metrics['mi_phiI_psi']:.6f}  (higher = Phi_I is informative)")
        print(f"  MI(Phi_R , Psi+)                   : {metrics['mi_phiR_psi']:.6f}  (lower  = Phi_R not informative)")
        print(f"  GT error ||Phi_I - Phi_I_true||^2  : {metrics['gt_error']:.6f}")
        print(f"  Total loss                         : {metrics['total_loss']:.6f}")
        print("=" * 60)

    # --- Save metrics ---
    metrics_path = os.path.join(output_dir, "aind_v2_les_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write("aIND Metrics  (LES + Gaussianization + OLS, Toy v2)\n")
        fh.write(f"LES: filter_type={les_filter_type}  sigma={les_sigma}\n")
        fh.write(f"OLS: beta={results['beta']:.6f}\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"I(Phi_I ; Phi_R)                   : {metrics['mutual_information']:.6f}\n")
        fh.write(f"Residual energy ||Phi - Phi_I||^2  : {metrics['residual_energy']:.6f}\n")
        fh.write(f"MI(Phi_I , Psi+)                   : {metrics['mi_phiI_psi']:.6f}\n")
        fh.write(f"MI(Phi_R , Psi+)                   : {metrics['mi_phiR_psi']:.6f}\n")
        fh.write(f"GT error ||Phi_I - Phi_I_true||^2  : {metrics['gt_error']:.6f}\n")
        fh.write(f"Total loss                         : {metrics['total_loss']:.6f}\n")
    print(f"Metrics saved to: {metrics_path}")

    # --- Visualise preprocessing pipeline ---
    visualize_preprocessing(
        Phi,
        results["T_Phi"],
        results["T_Phi_resolved"],
        results["T_Phi_sgs"],
        output_dir=output_dir,
        prefix="aind_v2_les",
    )

    # --- Visualise decomposition (same style as v2) ---
    visualize_decomposition(
        Phi,
        Psi_plus,
        Phi_I,
        Phi_R,
        Phi_I_true,
        Phi_R_true,
        output_dir=output_dir,
        prefix="aind_v2_les",
    )

    return results, metrics


if __name__ == "__main__":
    main(
        nx=100,
        ny=100,
        t=0.0,
        les_sigma=0.3,
        les_filter_type="gaussian",
        n_quantiles=1000,
        output_dir="results_v2_les",
        seed=1,
        verbose=True,
    )
