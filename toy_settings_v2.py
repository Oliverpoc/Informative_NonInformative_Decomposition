"""
Toy problem v2 for aIND decomposition: Nonlinear source field with cross-term interactions.

Problem Setup:
- c(x,y,t): Common background component
- f(x,y,t): Informative component  (truly drives the target Ψ⁺)
- g(x,y,t): Non-informative residual component

- Φ (source field): Φ = c² + cf + cg + fg  [nonlinear cross-term interactions]
  Factored form:    Φ = (c + f)(c + g)

- Ground truth additive decomposition:
    Φᴵ = cf + fg = f(c + g)   ← all terms containing f
    Φᴿ = c² + cg = c(c + g)  ← all terms NOT containing f
    Check: Φᴵ + Φᴿ = Φ ✓

- Ψ⁺ (target field): Reference for information discrimination
    Ψ⁺(x,t) = 0.5·f(x,t+ΔT)² - 0.2·f(x,t+ΔT) + ε(x,t)
    where ε ~ N(0, 0.1)

- Goal: Decompose Φ into Φᴵ + Φᴿ such that Φᴵ captures all info about Ψ⁺.

Note on difficulty vs. toy_settings.py (v1):
  v1: Φ = f + g  (linear sum — no cross-terms, clean independence)
  v2: Φ = (c+f)(c+g) — Φᴵ and Φᴿ share the (c+g) factor, so they are
      not trivially independent even after removing f. The MI minimization
      must work harder to find the optimal split.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Component fields
# ---------------------------------------------------------------------------

def c_func(x, y, t):
    """
    Common background component c(x, y, t).
    Appears in all cross-terms of Φ. Not directly informative about Ψ⁺,
    but mediates the nonlinear coupling between f and g.
    """
    return np.sin(2 * np.pi * x - 2.0 * t) * np.cos(2 * np.pi * y)


def f_func(x, y, t):
    """
    Informative component f(x, y, t).
    This is the component that truly contains information about Ψ⁺.
    Higher spatial frequency than c to make separation non-trivial.
    """
    return 2.0 * np.sin(4 * np.pi * x - 1.5 * t) * np.sin(4 * np.pi * y)


def g_func(x, y, t):
    """
    Non-informative residual component g(x, y, t).
    Incommensurate spatial frequencies ensure near-independence from f and c.
    """
    return (1.0 / 5.0) * np.sin(7 * np.sqrt(2) * np.pi * x - 0.1 * t) * np.sin(
        8 * np.sqrt(3) * np.pi * y - 0.5 * t
    )


# ---------------------------------------------------------------------------
# Source and target fields
# ---------------------------------------------------------------------------

def phi_func(x, y, t):
    """
    Source field Φ = c² + cf + cg + fg = (c + f)(c + g).
    Contains nonlinear interactions between all three components.
    """
    c = c_func(x, y, t)
    f = f_func(x, y, t)
    g = g_func(x, y, t)
    return c**2 + c * f + c * g + f * g


def phi_informative_true(x, y, t):
    """
    Ground truth informative component Φᴵ = f(c + g) = cf + fg.
    Contains all terms that carry information about Ψ⁺ through f.
    """
    c = c_func(x, y, t)
    f = f_func(x, y, t)
    g = g_func(x, y, t)
    return f * c + f * g  # = f * (c + g)


def phi_residual_true(x, y, t):
    """
    Ground truth residual component Φᴿ = c(c + g) = c² + cg.
    Contains no f → carries no information about Ψ⁺.
    """
    c = c_func(x, y, t)
    g = g_func(x, y, t)
    return c**2 + c * g  # = c * (c + g)


def F_analytical(f):
    """
    Analytical mapping F: f → Ψ⁺  (same as v1 for comparability).
    Ψ⁺ = 0.5·f² - 0.2·f
    """
    return 0.5 * f**2 - 0.2 * f


def psi_plus_func(x, y, t, deltaT=1.0, noise_std=0.1):
    """
    Target field Ψ⁺(x,t) = 0.5·f(x,t+ΔT)² - 0.2·f(x,t+ΔT) + ε(x,t).

    Parameters
    ----------
    x, y      : Spatial coordinates
    t         : Current time
    deltaT    : Time lag (default 1.0)
    noise_std : Noise std ε ~ N(0, noise_std)
    """
    f = f_func(x, y, t + deltaT)
    epsilon = np.random.normal(0, noise_std, size=f.shape)
    return F_analytical(f) + epsilon


# ---------------------------------------------------------------------------
# Visualisation / standalone test
# ---------------------------------------------------------------------------

def main(
    nx: int = 100,
    ny: int = 100,
    t: float = 0.0,
    output_dir: str = "results_v2",
    seed: int = 42,
) -> None:
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Component fields
    c  = c_func(X, Y, t)
    f  = f_func(X, Y, t)
    g  = g_func(X, Y, t)

    # Derived fields
    Phi        = phi_func(X, Y, t)
    Phi_I_true = phi_informative_true(X, Y, t)
    Phi_R_true = phi_residual_true(X, Y, t)
    Psi_plus   = psi_plus_func(X, Y, t, deltaT=0.0, noise_std=0.1)

    # Sanity check: Φᴵ + Φᴿ == Φ
    recon_err = np.max(np.abs(Phi_I_true + Phi_R_true - Phi))
    print(f"Reconstruction error |Phi_I + Phi_R - Phi|_max = {recon_err:.2e}  (should be ~0)")

    # ------------------------------------------------------------------
    # Plot 1: Component fields (c, f, g, Φ)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    specs = [
        (c,   "c  (common background)\nsin(2πx)·cos(2πy)"),
        (f,   "f  (informative)\n2·sin(4πx)·sin(4πy)"),
        (g,   "g  (residual)\n(1/5)·sin(7√2·πx)·sin(8√3·πy)"),
        (Phi, "Φ = c²+cf+cg+fg\n= (c+f)(c+g)"),
    ]
    for ax, (field, title) in zip(axes, specs):
        im = ax.imshow(field, origin="lower", cmap="RdBu",
                       extent=[x.min(), x.max(), y.min(), y.max()])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Toy v2 — Component fields", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v2_components.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Decomposition ground truth (Φ, Φᴵ, Φᴿ, Ψ⁺)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    specs2 = [
        (Phi,        "Φ = (c+f)(c+g)\nsource field"),
        (Phi_I_true, "Φᴵ = f(c+g)\nground truth informative"),
        (Phi_R_true, "Φᴿ = c(c+g)\nground truth residual"),
        (Psi_plus,   "Ψ⁺ = F(f) + ε\ntarget field"),
    ]
    for ax, (field, title) in zip(axes, specs2):
        im = ax.imshow(field, origin="lower", cmap="RdBu",
                       extent=[x.min(), x.max(), y.min(), y.max()])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Toy v2 — Ground truth decomposition", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v2_decomposition.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: Scatter — Ψ⁺ vs Φᴵ and Φᴿ
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].scatter(Phi_I_true.ravel(), Psi_plus.ravel(), s=3, alpha=0.3, color="steelblue")
    axes[0].set_xlabel("Φᴵ (ground truth)"); axes[0].set_ylabel("Ψ⁺")
    axes[0].set_title("Φᴵ vs Ψ⁺\n(should be structured)")

    axes[1].scatter(Phi_R_true.ravel(), Psi_plus.ravel(), s=3, alpha=0.3, color="tomato")
    axes[1].set_xlabel("Φᴿ (ground truth)"); axes[1].set_ylabel("Ψ⁺")
    axes[1].set_title("Φᴿ vs Ψ⁺\n(should be cloud — no info)")

    axes[2].scatter(Phi_I_true.ravel(), Phi_R_true.ravel(), s=3, alpha=0.3, color="purple")
    axes[2].set_xlabel("Φᴵ"); axes[2].set_ylabel("Φᴿ")
    axes[2].set_title("Φᴵ vs Φᴿ\n(shared (c+g) factor → residual correlation)")

    plt.suptitle("Toy v2 — Scatter relationships", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v2_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to: {output_dir}/")
    print(f"  v2_components.png    -- c, f, g, Phi")
    print(f"  v2_decomposition.png -- Phi, Phi_I, Phi_R, Psi_plus")
    print(f"  v2_scatter.png       -- Psi_plus vs Phi_I/Phi_R, Phi_I vs Phi_R")


if __name__ == "__main__":
    main(output_dir="results_v2", seed=42)
