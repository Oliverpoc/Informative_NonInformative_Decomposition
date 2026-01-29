"""
Toy problem for aIND decomposition: Source and Target fields generation.

Problem Setup:
- Φ (source field): The object to be decomposed, composed of f + g
- Ψ⁺ (target field): Reference for information discrimination
  Ψ⁺(x,t) = Ψ(x,t + deltaT) = 0.5f(x,t)² - 0.2f(x,t) + epsilon(x,t)
  where deltaT = 1 and epsilon(x,t) ~ N(0, σ) with σ = 0.1
- Goal: Decompose Φ into Φᴵ + Φᴿ, such that Φᴵ contains all information about Ψ⁺
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

def f_func(x, y, t):
    """
    Informative component f(x, y, t).
    This is the component that truly contains information about Ψ⁺.
    """
    return 2 * np.sin(2 * np.pi * x - 2 * t) * np.sin(2 * np.pi * y)


def g_func(x, y, t):
    """
    Residual component g(x, y, t).
    This is the residual component unrelated to Ψ⁺.
    """
    return (1 / 5) * np.sin(7 * np.sqrt(2) * np.pi * x - 0.1 * t) * np.sin(
        8 * np.sqrt(3) * np.pi * y - 0.5 * t
    )


def phi_func(x, y, t):
    """
    Source field Φ = f + g.
    The object to be decomposed, containing both informative (f) and residual (g) parts.
    """
    return f_func(x, y, t) + g_func(x, y, t)

def F_analytical(f):
    """
    Analytical mapping F: Φᴵ → Ψ⁺.
    Ψ⁺ = F(Φᴵ) = 0.5 * Φᴵ² - 0.2 * Φᴵ
    """
    return 0.5 * f ** 2 - 0.2 * f

def psi_plus_func(x, y, t, deltaT=1.0, noise_std=0.1):
    """
    Target field Ψ⁺(x,t) = Ψ(x,t + deltaT) = 0.5f(x,t)² - 0.2f(x,t) + epsilon(x,t).
    
    Reference for information discrimination, used to determine which components are "useful".
    Represents the field at time t + deltaT, where deltaT = 1.
    
    Parameters:
    - x, y: Spatial coordinates
    - t: Time
    - deltaT: Time step (default = 1.0)
    - noise_std: Standard deviation of noise epsilon, where epsilon(x,t) ~ N(0, sigma)
                 with sigma = 0.1
    
    Returns:
    - Ψ⁺(x,t) = 0.5f(x,t)² - 0.2f(x,t) + epsilon(x,t)
    """
    # f(x,t): informative component at time t
    f = f_func(x, y, t + deltaT)
    
    # epsilon(x,t) ~ N(0, sigma) with sigma = noise_std = 0.1
    # Pointwise normal distribution with zero mean and standard deviation 0.1
    epsilon = np.random.normal(0, noise_std, size=f.shape)
    
    # Ψ⁺(x,t) = 0.5f(x,t)² - 0.2f(x,t) + epsilon(x,t)
    return 0.5 * f ** 2 - 0.2 * f + epsilon


def main(
    nx: int = 100,
    ny: int = 100,
    t: float = 0.0,
    output_dir: str = "results",
    seed: int = 42,
) -> None:
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ==== Step 1: Build grid and field data ====
    # Generate spatial grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Generate field data
    # Φ: Object to be decomposed (source field, contains f + g)
    # Ψ⁺: Reference for information discrimination (target field at t + deltaT)
    #     Ψ⁺(x,t) = 0.5f(x,t)² - 0.2f(x,t) + epsilon(x,t), where deltaT = 1
    Phi = phi_func(X, Y, t)  # Φ = f + g at time t
    print(Phi.shape)
    Psi_plus = psi_plus_func(X, Y, t, deltaT=0.0, noise_std=0.1)  # Ψ⁺(x,t) = Ψ(x,t+1)
    
    # Also generate f and g for comparison
    f = f_func(X, Y, t)  # informative component
    g = g_func(X, Y, t)  # residual component

    # ==== Step 2: Visualize field data ====
    plt.figure(figsize=(16, 4))
    titles = [
        "Φ (Source Field, object to be decomposed)\nΦ = f + g",
        "Ψ⁺ (Target Field, reference for discrimination)\nΨ⁺(x,t) = 0.5f² - 0.2f + ε",
        "f (Informative Component)\nContains all information about Ψ⁺",
        "g (Residual Component)\nUnrelated to Ψ⁺",
    ]
    fields = [Phi, Psi_plus, f, g]

    for i, (field, title) in enumerate(zip(fields, titles), 1):
        plt.subplot(1, 4, i)
        im = plt.imshow(field, origin="lower", cmap="coolwarm",extent=[x.min(), x.max(), y.min(), y.max()])
        plt.title(title, fontsize=10)
        plt.colorbar(im, fraction=0.046)
        plt.xlabel("x")
        plt.ylabel("y")
    plt.gca().set_aspect("auto")  # or "equal" for equal scaling
    plt.tight_layout()
    fields_fig_path = os.path.join(output_dir, "source_target_fields.png")
    plt.savefig(fields_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Field data visualization saved to: {fields_fig_path}")

    # # Subplot 1: Contours of joint probability p(Φ, Ψ⁺)
    # plt.subplot(1, 2, 1)
    
    # # Use 2D histogram to estimate joint probability distribution p(Φ, Ψ⁺)
    # H, xedges, yedges = np.histogram2d(
    #     Phi_flat, Psi_plus_flat,
    #     bins=80,  # Adjust for smoothness
    #     density=True
    # )
    
    # # Calculate bin centers for contour plotting
    # xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    # ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    # Xc, Yc = np.meshgrid(xcenters, ycenters)
    
    # # Contour plot: from lighter (near white) to darker (near blue)
    # levels = np.linspace(H.min(), H.max(), 20)
    # cf = plt.contourf(
    #     Xc, Yc, H.T,  # Note: H.T to match x and y axes
    #     levels=levels,
    #     cmap="Blues"  # From white to blue
    # )
    # plt.colorbar(cf, label="p(Φ, Ψ⁺)")
    
    # # Overlay analytical solution line F_true(Φᴵ)
    # # Note: F_true(Φᴵ) = 0.5Φᴵ² - 0.2Φᴵ, where Φᴵ = f (informative component)
    # # In the (Φ, Ψ⁺) space, we plot F_true(f) where f is the informative component
    # f_range = np.linspace(f_flat.min(), f_flat.max(), 400)
    # psi_true = F_analytical(f_range)  # F_true(Φᴵ) = F_true(f) = 0.5f² - 0.2f
    
    # # Plot the analytical line: this shows F_true(Φᴵ) where Φᴵ = f
    # plt.plot(
    #     f_range, psi_true,
    #     "k--", linewidth=2,
    #     label=r"$F_{\mathrm{true}}(\Phi^I)$"
    # )
    
    # plt.xlabel(r"$\Phi$ (Source Field)")
    # plt.ylabel(r"$\Psi^+$ (Target Field)")
    # plt.title(r"Joint density $p(\Phi,\Psi^+)$ with $F_{\mathrm{true}}(\Phi^I)$")
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # # Subplot 2: Relationship between f and g (should be independent)
    # plt.subplot(1, 2, 2)
    # plt.scatter(f_flat, g_flat, s=6, alpha=0.5, color="red")
    # plt.xlabel("f (Informative Component)")
    # plt.ylabel("g (Residual Component)")
    # plt.title("Independence: f and g should be independent\n(Goal: I(g; f) = 0)")
    # plt.grid(True, alpha=0.3)

    # plt.tight_layout()
    # scatter_fig_path = os.path.join(output_dir, "source_target_relations.png")
    # plt.savefig(scatter_fig_path, dpi=300, bbox_inches="tight")
    # plt.close()
    # print(f"Relationship plots saved to: {scatter_fig_path}")

if __name__ == "__main__":
    output_dir = "results"
    main(output_dir=output_dir,seed=42)
