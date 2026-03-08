"""
aIND via Convolutional Autoencoder  (AE)

Practical setting:
    - ONLY data points of source field Phi and target field Psi+ are known.
    - No analytical form of Phi, Psi, or the mapping F is assumed.
    - Structural assumption: some informative part Phi_I of Phi determines Psi;
      the residual Phi_R does not.
    - Encoder sees ONLY Phi — no Psi during forward pass (no information leak).

Architecture:
    Encoder E    : Phi (H,W)          -> z_I, z_R
    Decoder D_I  : z_I                -> Phi_I (H,W)
    Decoder D_R  : z_R                -> Phi_R (H,W)
    Reconstructor R: (Phi_I, Phi_R)  -> Phi_hat = Phi_I + Phi_R
    Target head H  : Phi_I            -> Psi_hat   [pixel-wise MLP, learns F from data]

Training data:
    Multiple (Phi, Psi+) data point pairs across time snapshots.
    No functional form of F assumed — H learns it end-to-end.

Loss:
    L_recon   : ||Phi - Phi_hat||^2              reconstruction
    L_func    : ||Psi - H(Phi_I)||^2             H learns the unknown F
    L_indep   : ||Cov(z_I, z_R)||_F^2            latent decorrelation
    L_indep_r : corr(Phi_R, Psi)^2              Phi_R must not correlate with Psi

    L_total = L_recon + lambda1*L_func + lambda2*L_indep + lambda3*L_indep_r

All parameters trained jointly with one optimiser.
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

from toy_settings import phi_func, psi_plus_func, f_func, g_func, F_analytical


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim_i=32, latent_dim_r=32):
        super().__init__()
        self.c1 = ConvBlock(1, 32, stride=2)    # H/2, W/2
        self.c2 = ConvBlock(32, 64, stride=2)   # H/4, W/4
        self.c3 = ConvBlock(64, 128, stride=2)  # H/8, W/8
        self.pool    = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        hidden_dim   = 128 * 4 * 4
        self.fc_i = nn.Linear(hidden_dim, latent_dim_i)
        self.fc_r = nn.Linear(hidden_dim, latent_dim_r)

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = self.pool(h)
        h = self.flatten(h)
        return self.fc_i(h), self.fc_r(h)


class FieldDecoder(nn.Module):
    def __init__(self, latent_dim, out_h, out_w):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.fc  = nn.Linear(latent_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h   = self.fc(z).view(z.shape[0], 128, 4, 4)
        out = self.net(h)
        if out.shape[-2:] != (self.out_h, self.out_w):
            out = F.interpolate(out, size=(self.out_h, self.out_w),
                                mode="bilinear", align_corners=False)
        return out


class TargetHead(nn.Module):
    """
    H: Phi_I (1, H, W) -> Psi_hat (1, H, W).
    Pixel-wise MLP via 1x1 convolutions — learns the unknown mapping F pointwise.
    No analytical form of F assumed.
    """

    def __init__(self, hidden_dim=32, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(1, hidden_dim, kernel_size=1), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1), nn.GELU()]
        layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, phi_i):
        return self.net(phi_i)


class DecompositionNet(nn.Module):
    """
    Full autoencoder for aIND decomposition.

    Forward pass (Phi only — no Psi):
        z_I, z_R  = E(Phi)
        Phi_I     = D_I(z_I)
        Phi_R     = D_R(z_R)
        Phi_hat   = R(Phi_I, Phi_R) = Phi_I + Phi_R
        Psi_hat   = H(Phi_I)   [pixel-wise MLP, F is learned — not assumed known]
    """

    def __init__(self, H, W, latent_dim_i=16, latent_dim_r=32,
                 target_hidden=32, target_layers=3):
        super().__init__()
        self.encoder     = Encoder(latent_dim_i=latent_dim_i, latent_dim_r=latent_dim_r)
        self.decoder_i   = FieldDecoder(latent_dim_i, H, W)
        self.decoder_r   = FieldDecoder(latent_dim_r, H, W)
        self.target_head = TargetHead(target_hidden, target_layers)

    def forward(self, phi):
        z_i, z_r = self.encoder(phi)
        phi_i    = self.decoder_i(z_i)
        phi_r    = self.decoder_r(z_r)
        phi_hat  = phi_i + phi_r                 # Reconstructor R
        psi_hat  = self.target_head(phi_i)       # Target head H (learned F)
        return {
            "z_i": z_i, "z_r": z_r,
            "phi_i": phi_i, "phi_r": phi_r,
            "phi_hat": phi_hat, "psi_hat": psi_hat,
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_decomposition(Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true=None):
    m = {}
    m["mutual_information"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Phi_I.ravel(), random_state=42)[0]
    m["residual_energy"] = mean_squared_error(Phi.ravel(), Phi_I.ravel())
    m["mi_phiI_psi"] = mutual_info_regression(
        Phi_I.reshape(-1, 1), Psi_plus.ravel(), random_state=42)[0]
    m["mi_phiR_psi"] = mutual_info_regression(
        Phi_R.reshape(-1, 1), Psi_plus.ravel(), random_state=42)[0]
    m["gt_error"] = (
        mean_squared_error(Phi_I_true.ravel(), Phi_I.ravel())
        if Phi_I_true is not None else float("nan")
    )
    return m


# ---------------------------------------------------------------------------
# Independence losses
# ---------------------------------------------------------------------------

def latent_cross_cov(z_i: torch.Tensor, z_r: torch.Tensor) -> torch.Tensor:
    """
    Squared Frobenius norm of the cross-covariance matrix Cov(z_i, z_r).
    z_i: (B, d_i), z_r: (B, d_r).
    Reliable even for small batch sizes (no O(B^2) kernel needed).
    """
    zi_c = z_i - z_i.mean(0, keepdim=True)
    zr_c = z_r - z_r.mean(0, keepdim=True)
    cov  = (zi_c.T @ zr_c) / z_i.shape[0]   # (d_i, d_r)
    return (cov ** 2).sum()


def spatial_corr_sq(phi_r: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    Mean squared spatial (pixel-wise) Pearson correlation between phi_r and psi,
    averaged over the batch.
    phi_r, psi: (B, 1, H, W)
    Returns a scalar ≥ 0.  Should be ~0 if Phi_R is independent of Psi.
    """
    B   = phi_r.shape[0]
    pr  = phi_r.reshape(B, -1)          # (B, H*W)
    ps  = psi.reshape(B, -1)
    pr  = pr - pr.mean(1, keepdim=True)
    ps  = ps - ps.mean(1, keepdim=True)
    pr_std = pr.std(1, keepdim=True).clamp(min=1e-8)
    ps_std = ps.std(1, keepdim=True).clamp(min=1e-8)
    corr = ((pr / pr_std) * (ps / ps_std)).mean(1)   # (B,)
    return (corr ** 2).mean()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ae_loss(out, phi, psi,
            lambda1=5.0, lambda2=10.0, lambda3=5.0):
    """
    L_recon   : ||Phi - phi_hat||^2         reconstruction
    L_func    : ||Psi - H(Phi_I)||^2        H learns the unknown F from data
    L_indep   : ||Cov(z_I, z_R)||_F^2       latent decorrelation
    L_indep_r : corr(Phi_R, Psi)^2          Phi_R must not correlate with Psi
    """
    L_recon   = F.mse_loss(out["phi_hat"], phi)
    L_func    = F.mse_loss(out["psi_hat"], psi)
    L_indep   = latent_cross_cov(out["z_i"], out["z_r"])
    L_indep_r = spatial_corr_sq(out["phi_r"], psi)
    total = L_recon + lambda1 * L_func + lambda2 * L_indep + lambda3 * L_indep_r
    bd = dict(
        total=total.item(), recon=L_recon.item(), func=L_func.item(),
        indep=L_indep.item(), indep_r=L_indep_r.item(),
    )
    return total, bd


# ---------------------------------------------------------------------------
# Dataset — multiple time snapshots
# ---------------------------------------------------------------------------

def make_dataset(nx=100, ny=100, n_samples=200,
                 t_max=4 * np.pi, noise_std=1e-3, seed=42):
    """
    Generate n_samples (Phi, Psi+) 2D field pairs across time t ∈ [0, t_max].
    Returns float32 tensors of shape (N, 1, ny, nx).
    """
    np.random.seed(seed)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)
    t_values = np.linspace(0, t_max, n_samples)

    Phi_list = [phi_func(X, Y, t) for t in t_values]
    Psi_list = [psi_plus_func(X, Y, t, deltaT=0.0, noise_std=noise_std)
                for t in t_values]

    Phi_arr = np.stack(Phi_list, axis=0)[:, None, :, :]   # (N, 1, ny, nx)
    Psi_arr = np.stack(Psi_list, axis=0)[:, None, :, :]

    Phi_t = torch.tensor(Phi_arr, dtype=torch.float32)
    Psi_t = torch.tensor(Psi_arr, dtype=torch.float32)
    return Phi_t, Psi_t, t_values


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ae(model, Phi_t, Psi_t,
             num_epochs=300, lr=1e-3, batch_size=16,
             lambda1=5.0, lambda2=10.0, lambda3=5.0,
             grad_clip=5.0, verbose=True):
    device = next(model.parameters()).device
    Phi_t  = Phi_t.to(device)
    Psi_t  = Psi_t.to(device)
    N      = Phi_t.shape[0]

    # Joint optimiser: encoder + decoder_i + decoder_r all together
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    log_every = max(1, num_epochs // 10)
    keys      = ("total", "recon", "func", "indep", "indep_r")
    history   = {k: [] for k in keys}

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        Phi_s, Psi_s = Phi_t[perm], Psi_t[perm]

        epoch_buf = {k: [] for k in keys}

        for start in range(0, N, batch_size):
            phi_b = Phi_s[start : start + batch_size]
            psi_b = Psi_s[start : start + batch_size]
            if phi_b.shape[0] < 2:
                continue

            optimizer.zero_grad()
            out        = model(phi_b)
            loss, bd   = ae_loss(out, phi_b, psi_b, lambda1, lambda2, lambda3)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            for k in keys:
                epoch_buf[k].append(bd[k])

        scheduler.step()

        if epoch % log_every == 0 or epoch == num_epochs:
            mean_bd = {k: float(np.mean(epoch_buf[k])) for k in keys}
            for k in keys:
                history[k].append(mean_bd[k])
            if verbose:
                print(f"  [Epoch {epoch:5d}]  "
                      f"recon={mean_bd['recon']:.4f}  "
                      f"func={mean_bd['func']:.4f}  "
                      f"indep={mean_bd['indep']:.3e}  "
                      f"indep_r={mean_bd['indep_r']:.4f}  "
                      f"total={mean_bd['total']:.4f}")

    model.eval()
    return history


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def aind_decomposition_ae(Phi, Psi_plus,
                          latent_dim_i=16, latent_dim_r=32,
                          num_epochs=300, lr=1e-3, batch_size=16,
                          n_samples=200, t_max=4 * np.pi,
                          lambda1=5.0, lambda2=10.0, lambda3=5.0,
                          verbose=True, seed=42, device=None):
    """
    aIND decomposition via convolutional autoencoder.

    Parameters
    ----------
    Phi, Psi_plus : 2D np.ndarray (H, W) — test/evaluation fields (e.g. t=0)
    Training data is generated internally across n_samples time steps.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Device: {device}")

    H, W = Phi.shape

    # Generate training dataset
    if verbose:
        print(f"Generating training data: {n_samples} time snapshots ...")
    Phi_t, Psi_t, t_values = make_dataset(
        nx=W, ny=H, n_samples=n_samples, t_max=t_max, seed=seed)

    # Build model
    model = DecompositionNet(H, W, latent_dim_i=latent_dim_i,
                             latent_dim_r=latent_dim_r).to(device)
    if verbose:
        n_p = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_p} params  "
              f"(z_i={latent_dim_i}, z_r={latent_dim_r})")
        print(f"Training: {num_epochs} epochs  lr={lr}  batch={batch_size}  "
              f"λ1={lambda1}  λ2={lambda2}  λ3={lambda3}")
        print("=" * 60)

    history = train_ae(model, Phi_t, Psi_t,
                       num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                       lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                       verbose=verbose)

    # Evaluate on the supplied test field
    model.eval()
    phi_input = torch.tensor(Phi[None, None, :, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(phi_input)

    Phi_I   = out["phi_i"][0, 0].cpu().numpy()       # (H, W)
    Phi_R   = out["phi_r"][0, 0].cpu().numpy()   # direct model output
    # Phi_R   = Phi - Phi_I                           # residual: Phi minus informative part
    psi_hat = out["psi_hat"][0, 0].cpu().numpy()
    z_i     = out["z_i"].cpu().numpy()
    z_r     = out["z_r"].cpu().numpy()

    if verbose:
        r_IR = float(np.corrcoef(Phi_I.ravel(), Phi_R.ravel())[0, 1])
        r_RP = float(np.corrcoef(Phi_R.ravel(), Psi_plus.ravel())[0, 1])
        func_err = float(np.mean((Psi_plus - psi_hat) ** 2))
        print(f"\ncorr(Phi_I, Phi_R)  = {r_IR:.4f}  (should be ~0)")
        print(f"corr(Phi_R, Psi+)   = {r_RP:.4f}  (should be ~0)")
        print(f"H(Phi_I) vs Psi MSE = {func_err:.6f}")

    return {
        "Phi_I": Phi_I, "Phi_R": Phi_R,
        "psi_hat": psi_hat,
        "z_i": z_i, "z_r": z_r,
        "loss_history": history,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_training(history, output_dir="results_ae", prefix="aind_ae"):
    os.makedirs(output_dir, exist_ok=True)
    n  = len(history["total"])
    xs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    specs = [
        ("total",   "Total loss",                      "steelblue",   False),
        ("recon",   "L_recon  ||Phi - Phi_hat||^2",    "teal",        False),
        ("func",    "L_func   ||Psi - H(Phi_I)||^2",   "darkorange",  True),
        ("indep",   "L_indep  ||Cov(z_i,z_r)||_F^2",  "firebrick",   True),
        ("indep_r", "L_indep_r  corr(Phi_R,Psi)^2",   "purple",      True),
    ]
    for ax, (key, title, color, logy) in zip(axes, specs):
        vals = [max(v, 1e-12) for v in history[key]]
        if logy:
            ax.semilogy(xs, vals, color=color, lw=2)
        else:
            ax.plot(xs, vals, color=color, lw=2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Checkpoint")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_training.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training dynamics saved to: {path}")


def visualize_decomposition(Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true, Phi_R_true,
                            psi_hat=None, output_dir="results_ae", prefix="aind_ae"):
    os.makedirs(output_dir, exist_ok=True)

    # Field grid
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = [
        "\u03a6 (Source)\n\u03a6 = f + g",
        "\u03a6\u1d35_true = f\n(True Informative)",
        "\u03a6\u1d35 (AE Reconstructed)",
        "\u03a8\u207a (Target)",
        "\u03a6\u1d3f_true = g\n(True Residual)",
        "\u03a6\u1d3f (AE Reconstructed)",
    ]
    fields = [Phi, Phi_I_true, Phi_I, Psi_plus, Phi_R_true, Phi_R]
    vmin_I = min(Phi_I_true.min(), Phi_I.min())
    vmax_I = max(Phi_I_true.max(), Phi_I.max())
    vmin_R = min(Phi_R_true.min(), Phi_R.min())
    vmax_R = max(Phi_R_true.max(), Phi_R.max())
    vmin_list = [None, vmin_I, vmin_I, None, vmin_R, vmin_R]
    vmax_list = [None, vmax_I, vmax_I, None, vmax_R, vmax_R]
    for ax, field, title, vmin, vmax in zip(axes.flat, fields, titles, vmin_list, vmax_list):
        kw = {"origin": "lower", "cmap": "RdBu", "extent": [0, 1, 0, 2], "aspect": 0.5}
        if vmin is not None:
            kw["vmin"], kw["vmax"] = vmin, vmax
        im = ax.imshow(field, **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fp = os.path.join(output_dir, f"{prefix}_fields.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Field visualization saved to: {fp}")

    # Scatter diagnostics
    Pf  = Phi.ravel(); Psf = Psi_plus.ravel()
    PIf = Phi_I.ravel(); PRf = Phi_R.ravel()
    ft  = Phi_I_true.ravel()
    idx = np.random.choice(len(Pf), min(5000, len(Pf)), replace=False)

    n_cols = 4 if psi_hat is not None else 3
    plt.figure(figsize=(5 * n_cols, 5))

    plt.subplot(1, n_cols, 1)
    plt.scatter(PIf[idx], Psf[idx], s=6, alpha=0.5, label="Data")
    phi_I_range = np.linspace(PIf.min(), PIf.max(), 200)
    plt.plot(phi_I_range, F_analytical(phi_I_range), "r--", lw=2, label="F(\u03a6\u1d35)")
    plt.xlabel("\u03a6\u1d35"); plt.ylabel("\u03a8\u207a")
    plt.title("Mapping \u03a8\u207a \u2248 F(\u03a6\u1d35)")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, n_cols, 2)
    plt.scatter(PIf[idx], PRf[idx], s=6, alpha=0.5, color="red")
    plt.xlabel("\u03a6\u1d35"); plt.ylabel("\u03a6\u1d3f")
    plt.title("Independence I(\u03a6\u1d35;\u03a6\u1d3f)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, n_cols, 3)
    plt.scatter(ft[idx], PIf[idx], s=6, alpha=0.5, color="green")
    f_range = np.linspace(ft.min(), ft.max(), 100)
    plt.plot(f_range, f_range, "k--", lw=2, label="Perfect")
    plt.xlabel("f (true)"); plt.ylabel("\u03a6\u1d35")
    plt.title("Reconstruction Quality")
    plt.legend(); plt.grid(True, alpha=0.3)

    if psi_hat is not None:
        plt.subplot(1, n_cols, 4)
        phf = psi_hat.ravel()
        plt.scatter(Psf[idx], phf[idx], s=6, alpha=0.5, color="purple")
        r = float(np.corrcoef(Psf, phf)[0, 1])
        lim = [min(Psf.min(), phf.min()), max(Psf.max(), phf.max())]
        plt.plot(lim, lim, "k--", lw=2, label="Perfect")
        plt.xlabel("\u03a8\u207a (true)"); plt.ylabel("F(\u03a6\u1d35) predicted")
        plt.title(f"Functional fit  r={r:.4f}")
        plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    sp = os.path.join(output_dir, f"{prefix}_scatter.png")
    plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plots saved to: {sp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(nx=100, ny=100, t=0.0,
         latent_dim_i=16, latent_dim_r=32,
         num_epochs=300, lr=1e-3, batch_size=16,
         n_samples=200, t_max=4 * np.pi,
         lambda1=5.0, lambda2=10.0, lambda3=5.0,
         output_dir="results_ae", seed=1, verbose=True):

    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    # Evaluation fields at time t
    Phi        = phi_func(X, Y, t)
    Psi_plus   = psi_plus_func(X, Y, t, deltaT=0.0, noise_std=1e-3)
    Phi_I_true = f_func(X, Y, t)
    Phi_R_true = g_func(X, Y, t)

    if verbose:
        print("=" * 60)
        print("aIND  --  Convolutional Autoencoder")
        print("E(Phi)->(z_I,z_R) | H(Phi_I)->Psi_hat  [H learned, F unknown]")
        print("=" * 60)
        print(f"Grid: {nx}x{ny}  |  eval at t={t}")
        print(f"z_i={latent_dim_i}  z_r={latent_dim_r}  "
              f"epochs={num_epochs}  lr={lr}  batch={batch_size}")
        print(f"λ1={lambda1}  λ2={lambda2}  λ3={lambda3}")
        mi_gt = mutual_info_regression(
            Phi_I_true.ravel().reshape(-1, 1), Phi_R_true.ravel(), random_state=42)[0]
        print(f"\nGround truth I(f;g) = {mi_gt:.6f}  (target: ~0)\n")

    results = aind_decomposition_ae(
        Phi, Psi_plus,
        latent_dim_i=latent_dim_i, latent_dim_r=latent_dim_r,
        num_epochs=num_epochs, lr=lr, batch_size=batch_size,
        n_samples=n_samples, t_max=t_max,
        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
        verbose=verbose, seed=seed,
    )

    Phi_I = results["Phi_I"]
    Phi_R = results["Phi_R"]
    metrics = evaluate_decomposition(
        Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true=Phi_I_true)

    if verbose:
        print("\n" + "=" * 60)
        print("Final Evaluation Metrics  (AE)")
        print("=" * 60)
        print(f"  I(Phi_I ; Phi_R)                   : {metrics['mutual_information']:.6f}")
        print(f"  Residual energy ||Phi - Phi_I||^2  : {metrics['residual_energy']:.6f}")
        print(f"  MI(Phi_I , Psi+)                   : {metrics['mi_phiI_psi']:.6f}")
        print(f"  MI(Phi_R , Psi+)                   : {metrics['mi_phiR_psi']:.6f}")
        print(f"  GT error ||Phi_I - f||^2           : {metrics['gt_error']:.6f}")
        print("=" * 60)

    path = os.path.join(output_dir, "aind_ae_metrics.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("aIND Metrics  (Convolutional Autoencoder)\n")
        fh.write(f"z_i={latent_dim_i}  z_r={latent_dim_r}  epochs={num_epochs}\n")
        fh.write("=" * 60 + "\n")
        for name, key in [("I(Phi_I ; Phi_R)",  "mutual_information"),
                          ("Residual energy",    "residual_energy"),
                          ("MI(Phi_I, Psi+)",    "mi_phiI_psi"),
                          ("MI(Phi_R, Psi+)",    "mi_phiR_psi"),
                          ("GT error",           "gt_error")]:
            fh.write(f"{name:<35}: {metrics[key]:.6f}\n")
    print(f"Metrics saved to: {path}")

    visualize_training(results["loss_history"], output_dir=output_dir, prefix="aind_ae")
    visualize_decomposition(
        Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true, Phi_R_true,
        psi_hat=results["psi_hat"],
        output_dir=output_dir, prefix="aind_ae",
    )

    return results, metrics


if __name__ == "__main__":
    main(
        nx=100, ny=100, t=0.0,
        latent_dim_i=16, latent_dim_r=32,
        num_epochs=300, lr=1e-3, batch_size=16,
        n_samples=200, t_max=4 * np.pi,
        lambda1=5.0, lambda2=10.0, lambda3=5.0,
        output_dir="results_ae",
        seed=1, verbose=True,
    )
