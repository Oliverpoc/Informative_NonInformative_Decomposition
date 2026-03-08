"""
aIND via Triangular Flow + Nonlinear Projection Network  (JCF2)

Architecture
------------
    z_psi = f_psi(Psi)            [MonotoneFlow — Psi ONLY, clean]
    z_phi = f_phi(Phi | z_psi)    [ConditionalMonotoneFlow — transforms copula]
    → (z_phi, z_psi) ~ N(0, [[1,rho],[rho,1]])   [bivariate Gaussian via L_Flow]

    z_phi_I = g_net(z_psi)        [ProjectionNet MLP — nonlinear projection]
    z_phi_R = z_phi - z_phi_I

Projection: g_net replaces scalar OLS  A * z_psi
-------------------------------------------------
    In perfect joint Gaussian:  E[z_phi | z_psi] = rho * z_psi  (linear = OLS)
    In practice (imperfect Gaussian):  g_net learns the nonlinear conditional
    expectation E[z_phi | z_psi] → smaller residual variance, better independence.
    g_net is a natural generalisation — reduces to OLS when joint Gaussian holds.

    Joint training: f_psi, f_phi, g_net, rho_raw all trained together with one
    optimiser.  g_net adapts to the latent representation the flow learns.

Decode (LEAK-FREE)
------------------
    z_phi_I = g_net(z_psi)          [g_net input = z_psi = f_psi(Psi) only]
    Phi_I   = f_phi^{-1}(z_phi_I | z_psi)
            = h(z_psi) = h(f_psi(Psi))   [pure function of Psi — ZERO LEAK]

Loss  (L_total = L_Flow + lambda1*L_MSE + lambda2*L_Indep + lambda3*L_Smooth)
    L_Flow  : Bivariate NLL — drives (z_phi, z_psi) → joint Gaussian (rho != 0)
    L_MSE   : ||z_phi - g_net(z_psi)||^2 — trains g_net = E[z_phi | z_psi]
              (replaces L_Func; the constraint Psi=F(Phi_I) is now architectural)
    L_Indep : HSIC(Phi_I, Phi_R) — physical-space independence
    L_Smooth: ||Phi - Phi_I||^2 — anti-collapse

    rho = tanh(rho_raw)  [learnable, used in L_Flow bivariate NLL only]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from toy_settings import phi_func, psi_plus_func, f_func, g_func, F_analytical


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
        mean_squared_error(Phi_I_true, Phi_I) if Phi_I_true is not None else float("nan")
    )
    m["total_loss"] = m["mutual_information"] + m["residual_energy"]
    return m


# ---------------------------------------------------------------------------
# Unconditional Monotone Flow  (for Psi)
# ---------------------------------------------------------------------------

class MonotoneFlow(nn.Module):
    """
    z = a*x + b + (1/K)*sum_k tanh(c_k*x + d_k),   dz/dx > 0 always.
    Inverse via bisection (n_bisect) + one Newton correction.
    """
    def __init__(self, n_terms: int = 16, n_bisect: int = 60):
        super().__init__()
        self.n_terms  = n_terms
        self.n_bisect = n_bisect
        self.log_a = nn.Parameter(torch.zeros(1))
        self.b     = nn.Parameter(torch.zeros(1))
        self.log_c = nn.Parameter(torch.zeros(n_terms))
        self.d     = nn.Parameter(torch.zeros(n_terms))

    def _eval(self, x):
        a       = self.log_a.exp()
        c       = self.log_c.exp()
        arg     = x.unsqueeze(-1) * c + self.d
        th      = torch.tanh(arg)
        z       = a * x + self.b + th.sum(-1) / self.n_terms
        dz_dx   = a + (c * (1.0 - th ** 2)).sum(-1) / self.n_terms
        log_jac = dz_dx.clamp(min=1e-8).log()
        return z, log_jac

    def forward(self, x):
        return self._eval(x)

    def inverse(self, z_target):
        with torch.no_grad():
            a  = self.log_a.exp()
            b  = self.b
            lo = ((z_target - b - 1.2) / a).clamp(-1e6, 1e6)
            hi = ((z_target - b + 1.2) / a).clamp(-1e6, 1e6)
            for _ in range(self.n_bisect):
                mid      = 0.5 * (lo + hi)
                f_mid, _ = self._eval(mid)
                go_right = f_mid < z_target
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi,  mid)
            x_b = 0.5 * (lo + hi)
        f_x, log_jac = self._eval(x_b)
        return x_b + (z_target - f_x) / log_jac.exp().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Conditional Monotone Flow  (for Phi | z_psi)
# ---------------------------------------------------------------------------

def _make_param_net(out_dim: int, hidden_dim: int, n_hidden: int) -> nn.Sequential:
    """MLP: R^1 → R^out_dim  (scalar conditioning → parameter vector)."""
    layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    nn.init.zeros_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


class ConditionalMonotoneFlow(nn.Module):
    """
    z = a(c)*x + b(c) + (1/K)*sum_k tanh(c_k(c)*x + d_k(c))

    Strictly monotone in x for fixed conditioning scalar c (a(c)>0, c_k(c)>0).
    Networks (net_a, net_b, net_c, net_d) map c → flow parameters.
    At initialisation (last-layer zeros): a=1, b=0, c_k=1, d_k=0 → identity-like.

    Inverse (for fixed c): bisection + Newton, same as MonotoneFlow.
    """
    def __init__(self, n_terms: int = 16, n_bisect: int = 60,
                 hidden_dim: int = 32, n_hidden: int = 2):
        super().__init__()
        self.n_terms  = n_terms
        self.n_bisect = n_bisect
        K = n_terms
        self.net_log_a = _make_param_net(1, hidden_dim, n_hidden)
        self.net_b     = _make_param_net(1, hidden_dim, n_hidden)
        self.net_log_c = _make_param_net(K, hidden_dim, n_hidden)
        self.net_d     = _make_param_net(K, hidden_dim, n_hidden)

    def _params(self, cond):
        """cond: (N,) → a:(N,), b:(N,), c:(N,K), d:(N,K)"""
        ci = cond.unsqueeze(-1)                       # (N, 1)
        a  = self.net_log_a(ci).squeeze(-1).exp()     # (N,) > 0
        b  = self.net_b(ci).squeeze(-1)               # (N,)
        c  = self.net_log_c(ci).exp()                 # (N, K) > 0
        d  = self.net_d(ci)                           # (N, K)
        return a, b, c, d

    def _eval(self, x, cond):
        a, b, c, d = self._params(cond)
        arg     = x.unsqueeze(-1) * c + d            # (N, K)
        th      = torch.tanh(arg)
        z       = a * x + b + th.sum(-1) / self.n_terms
        dz_dx   = a + (c * (1.0 - th ** 2)).sum(-1) / self.n_terms
        log_jac = dz_dx.clamp(min=1e-8).log()
        return z, log_jac

    def forward(self, x, cond):
        return self._eval(x, cond)

    def inverse(self, z_target, cond):
        """Bisection (fixed cond) + Newton correction."""
        with torch.no_grad():
            a, b, _, _ = self._params(cond)
            lo = ((z_target - b - 1.2) / a).clamp(-1e6, 1e6)
            hi = ((z_target - b + 1.2) / a).clamp(-1e6, 1e6)
            for _ in range(self.n_bisect):
                mid      = 0.5 * (lo + hi)
                f_mid, _ = self._eval(mid, cond)
                go_right = f_mid < z_target
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi,  mid)
            x_b = 0.5 * (lo + hi)
        f_x, log_jac = self._eval(x_b, cond)
        return x_b + (z_target - f_x) / log_jac.exp().clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Projection Network  (replaces scalar OLS  A * z_psi)
# ---------------------------------------------------------------------------

class ProjectionNet(nn.Module):
    """
    Small MLP: z_psi → z_phi_I   (nonlinear projection in latent space).

    Replaces the scalar OLS coefficient A.  In a perfect bivariate Gaussian,
    E[z_phi | z_psi] = rho * z_psi (linear), so g_net reduces to OLS.
    In practice it captures residual nonlinear conditional structure.

    Trained end-to-end with L_MSE = ||z_phi - g_net(z_psi)||^2.
    Initialised to near-zero output so training starts stable.
    """

    def __init__(self, hidden_dim: int = 32, n_hidden: int = 2):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward(self, z_psi: torch.Tensor) -> torch.Tensor:
        return self.net(z_psi.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Triangular Flow  (joint, leak-free)
# ---------------------------------------------------------------------------

class TriangularFlow(nn.Module):
    """
    Triangular normalizing flow + nonlinear projection, all trained jointly.

        z_psi = f_psi(Psi)            [MonotoneFlow — Psi ONLY, clean]
        z_phi = f_phi(Phi | z_psi)    [ConditionalMonotoneFlow — transforms copula]
        z_phi_I = g_net(z_psi)        [ProjectionNet — E[z_phi | z_psi], nonlinear]
        z_phi_R = z_phi - z_phi_I

    All four components (f_psi, f_phi, g_net, rho_raw) share one optimiser.

    LEAK-FREE decode:
        Phi_I = f_phi^{-1}(g_net(z_psi) | z_psi)
              = h(z_psi) = h(f_psi(Psi))   [pure function of Psi]
    """

    def __init__(self, n_terms: int = 16, n_bisect: int = 60,
                 hidden_dim: int = 32, n_hidden: int = 2,
                 proj_hidden_dim: int = 32, proj_n_hidden: int = 2):
        super().__init__()
        self.f_psi   = MonotoneFlow(n_terms, n_bisect)
        self.f_phi   = ConditionalMonotoneFlow(n_terms, n_bisect, hidden_dim, n_hidden)
        self.g_net   = ProjectionNet(proj_hidden_dim, proj_n_hidden)
        self.rho_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def rho(self) -> torch.Tensor:
        return torch.tanh(self.rho_raw)

    def forward(self, phi, psi):
        z_psi, ldj_psi = self.f_psi(psi)
        z_phi, ldj_phi = self.f_phi(phi, z_psi)
        log_det = ldj_psi + ldj_phi
        z_phi_I = self.g_net(z_psi)          # nonlinear projection
        z_phi_R = z_phi - z_phi_I
        return z_phi, z_psi, z_phi_I, z_phi_R, log_det

    def decode_phi(self, z_phi_target: torch.Tensor, z_psi: torch.Tensor) -> torch.Tensor:
        """Phi_I = f_phi^{-1}(z_phi_target | z_psi).
        With z_phi_target = g_net(z_psi), this is a pure function of Psi."""
        return self.f_phi.inverse(z_phi_target, z_psi)


# ---------------------------------------------------------------------------
# OLS  /  HSIC  /  loss helpers
# ---------------------------------------------------------------------------

def compute_A(z_phi: torch.Tensor, z_psi: torch.Tensor) -> torch.Tensor:
    zpc = z_phi - z_phi.mean()
    zsc = z_psi - z_psi.mean()
    return (zpc * zsc).mean() / (zsc ** 2).mean().clamp(min=1e-8)


def hsic_rbf(X: torch.Tensor, Y: torch.Tensor, max_n: int = 2000) -> torch.Tensor:
    N = X.shape[0]
    if N > max_n:
        idx = torch.randperm(N, device=X.device)[:max_n]
        X, Y = X[idx], Y[idx]
    n  = X.shape[0]
    dX = (X.unsqueeze(0) - X.unsqueeze(1)).pow(2)
    dY = (Y.unsqueeze(0) - Y.unsqueeze(1)).pow(2)
    with torch.no_grad():
        sX = dX.median().sqrt().clamp(min=1e-4)
        sY = dY.median().sqrt().clamp(min=1e-4)
    KX = torch.exp(-dX / (2.0 * sX ** 2))
    KY = torch.exp(-dY / (2.0 * sY ** 2))
    KX = KX - KX.mean(0, keepdim=True) - KX.mean(1, keepdim=True) + KX.mean()
    KY = KY - KY.mean(0, keepdim=True) - KY.mean(1, keepdim=True) + KY.mean()
    return (KX * KY).sum() / (n - 1) ** 2


def _correlation_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(); yc = y - y.mean()
    cov = (xc * yc).mean()
    return (cov / ((xc**2).mean().clamp(1e-10) * (yc**2).mean().clamp(1e-10)).sqrt()) ** 2


def jcf2_loss(z_phi, z_psi, z_phi_I, z_phi_R, log_det, rho,
              Phi_I_std, Phi_R_std, phi_t,
              lambda1=1.0, lambda2=1.0, lambda3=1.0,
              indep_mode="hsic", max_hsic_n=2000):
    """
    L_Flow  : Bivariate Gaussian NLL — drives (z_phi,z_psi) → joint Gaussian
    L_MSE   : ||z_phi - g_net(z_psi)||^2 — trains g_net = E[z_phi|z_psi]
              (replaces L_Func; constraint Psi=F(Phi_I) is now architectural)
    L_Indep : HSIC(Phi_I, Phi_R) — physical-space independence
    L_Smooth: ||Phi - Phi_I||^2 — anti-collapse
    """
    det    = (1.0 - rho ** 2).clamp(min=1e-6)
    quad   = (z_phi**2 + z_psi**2 - 2.0*rho*z_phi*z_psi) / det
    L_Flow = (0.5*quad + 0.5*torch.log(det) - log_det).mean()
    L_MSE  = ((z_phi - z_phi_I) ** 2).mean()          # train g_net → E[z_phi|z_psi]
    L_Indep = (_correlation_sq(z_phi_I, z_phi_R) if indep_mode == "corr"
               else hsic_rbf(Phi_I_std, Phi_R_std, max_n=max_hsic_n))
    L_Smooth = ((phi_t - Phi_I_std) ** 2).mean()
    L_recon  = ((phi_t - (Phi_I_std + Phi_R_std)) ** 2).mean()
    total    = L_Flow + lambda1*L_MSE + lambda2*L_Indep + lambda3*L_Smooth
    # Effective linear correlation of g_net output with z_psi (diagnostic)
    A_eff  = compute_A(z_phi_I, z_psi)
    mi_rho = float(-0.5 * np.log(max(1.0 - rho.item()**2, 1e-8)))
    mi_A   = float(-0.5 * np.log(max(1.0 - A_eff.item()**2, 1e-8)))
    bd = dict(total=total.item(), flow=L_Flow.item(), mse=L_MSE.item(),
              indep=L_Indep.item(), smooth=L_Smooth.item(), recon=L_recon.item(),
              A=A_eff.item(), rho=rho.item(), mi_rho=mi_rho, mi_A=mi_A)
    return total, bd


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_jcf2(model: TriangularFlow, phi_t, psi_t,
               num_epochs=1000, lr=1e-3,
               lambda1=1.0, lambda2=1.0, lambda3=1.0,
               indep_mode="hsic", max_hsic_n=2000,
               grad_clip=5.0, verbose=True):
    # Single optimiser covers f_psi, f_phi, g_net, rho_raw — joint training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    log_every = max(1, num_epochs // 10)
    history   = {k: [] for k in ("total","flow","mse","indep","smooth","recon","A","rho","mi_rho","mi_A")}

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward: flow + g_net projection, all in one pass
        z_phi, z_psi, z_phi_I, z_phi_R, log_det = model(phi_t, psi_t)

        # Decode — LEAK-FREE: f_phi^{-1}(g_net(z_psi) | z_psi), pure function of Psi
        Phi_I_std = model.decode_phi(z_phi_I, z_psi)
        Phi_R_std = phi_t - Phi_I_std

        loss, bd = jcf2_loss(
            z_phi, z_psi, z_phi_I, z_phi_R, log_det, model.rho,
            Phi_I_std, Phi_R_std, phi_t,
            lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
            indep_mode=indep_mode, max_hsic_n=max_hsic_n,
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == num_epochs:
            for k in history: history[k].append(bd[k])
            if verbose:
                print(f"  [Epoch {epoch:5d}]  "
                      f"flow={bd['flow']:+.4f}  mse={bd['mse']:.4f}  "
                      f"indep={bd['indep']:.3e}  smooth={bd['smooth']:.4f}  "
                      f"rho={bd['rho']:+.4f}  A={bd['A']:+.4f}  I~={bd['mi_A']:.4f} nats")

    model.eval()
    return history


# ---------------------------------------------------------------------------
# Inference test
# ---------------------------------------------------------------------------

def run_inference_test(model: TriangularFlow, phi_t, psi_t,
                       phi_std, phi_mean, Phi_I_with_psi, Psi_flat):
    """
    With TriangularFlow, Phi_I = f_phi^{-1}(A*z_psi | z_psi) is a pure function
    of Psi (no leak by construction).  Test: permuting Psi should change Phi_I.
    """
    model.eval()
    with torch.no_grad():
        _, z_psi, z_phi_I, _, _ = model(phi_t, psi_t)
        Phi_I_normal = (model.decode_phi(z_phi_I, z_psi).cpu().numpy() * phi_std + phi_mean).ravel()

        # Permuted Psi — Phi_I should decorrelate with Psi+
        perm = torch.randperm(psi_t.shape[0], device=psi_t.device)
        _, z_psi_perm, z_phi_I_perm, _, _ = model(phi_t, psi_t[perm])
        Phi_I_perm = (model.decode_phi(z_phi_I_perm, z_psi_perm).cpu().numpy() * phi_std + phi_mean).ravel()

    r_normal  = float(np.corrcoef(Phi_I_normal, Psi_flat)[0, 1])
    r_perm    = float(np.corrcoef(Phi_I_perm,   Psi_flat)[0, 1])
    r_results = float(np.corrcoef(Phi_I_with_psi, Psi_flat)[0, 1])
    return {
        "corr_PhiI_Psi_from_results": r_results,
        "corr_PhiI_Psi_normal":       r_normal,
        "corr_PhiI_Psi_permuted":     r_perm,
        "inference_test_pass":        abs(r_normal) > 3.0 * abs(r_perm),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def aind_decomposition_jcf2(Phi, Psi_plus,
                             n_terms=16, n_bisect=60,
                             hidden_dim=32, n_hidden=2,
                             num_epochs=1000, lr=1e-3,
                             lambda1=1.0, lambda2=1.0, lambda3=1.0,
                             indep_mode="hsic", max_hsic_n=2000,
                             verbose=True, seed=42, device=None):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"Device: {device}")

    phi_flat = Phi.ravel(); psi_flat = Psi_plus.ravel()
    phi_mean, phi_std = phi_flat.mean(), phi_flat.std() + 1e-8
    psi_mean, psi_std = psi_flat.mean(), psi_flat.std() + 1e-8
    phi_t = torch.tensor((phi_flat - phi_mean) / phi_std, dtype=torch.float32, device=device)
    psi_t = torch.tensor((psi_flat - psi_mean) / psi_std, dtype=torch.float32, device=device)

    model = TriangularFlow(n_terms=n_terms, n_bisect=n_bisect,
                           hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)

    if verbose:
        n_p = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_p} params  (TriangularFlow: n_terms={n_terms}, hidden_dim={hidden_dim})")
        print(f"Training: {num_epochs} epochs  lr={lr}  "
              f"lambda1={lambda1}  lambda2={lambda2}  lambda3={lambda3}  indep={indep_mode}")
        print("=" * 60)

    history = train_jcf2(model, phi_t, psi_t,
                         num_epochs=num_epochs, lr=lr,
                         lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                         indep_mode=indep_mode, max_hsic_n=max_hsic_n, verbose=verbose)

    model.eval()
    with torch.no_grad():
        z_phi, z_psi, z_phi_I, z_phi_R, _ = model(phi_t, psi_t)
        # A_eff: effective linear correlation of g_net output with z_psi (diagnostic)
        A_val  = float(compute_A(z_phi_I, z_psi))
        Phi_I_std_flat = model.decode_phi(z_phi_I, z_psi).cpu().numpy()

    Phi_I_flat  = Phi_I_std_flat * phi_std + phi_mean
    Phi_R_flat  = phi_flat - Phi_I_flat
    rho_val     = float(model.rho.item())
    mi_estimate = float(-0.5 * np.log(max(1.0 - A_val**2, 1e-8)))

    if verbose:
        r_R   = float(np.corrcoef(z_phi_R.cpu().numpy(), z_psi.cpu().numpy())[0, 1])
        r_var = float((z_phi_R.cpu().numpy() ** 2).mean())
        print(f"\nLearned rho = {rho_val:+.4f}")
        print(f"g_net effective A = {A_val:+.4f}")
        print(f"I(Phi; Psi) ~= {mi_estimate:.4f} nats")
        print(f"corr(z_phi^R, z_psi) = {r_R:.4f}  (should be ~0)")
        print(f"Var(z_phi^R) = {r_var:.4f}")

    infer_results = run_inference_test(model, phi_t, psi_t, phi_std, phi_mean,
                                       Phi_I_flat, psi_flat)
    if verbose:
        print(f"\n--- Inference test (permuted Psi) ---")
        print(f"  corr(Phi_I, Psi) normal   : {infer_results['corr_PhiI_Psi_normal']:.4f}")
        print(f"  corr(Phi_I, Psi) permuted : {infer_results['corr_PhiI_Psi_permuted']:.4f}  (target: ~0)")
        if infer_results["inference_test_pass"]:
            print("  OK: Phi_I depends on Psi meaningfully")
        else:
            print("  WARNING: Phi_I not clearly Psi-dependent")

    return {
        "Phi_I": Phi_I_flat, "Phi_R": Phi_R_flat,
        "z_phi": z_phi.cpu().numpy(), "z_psi": z_psi.cpu().numpy(),
        "z_phi_I": z_phi_I.cpu().numpy(), "z_phi_R": z_phi_R.cpu().numpy(),
        "A": A_val, "rho": rho_val, "mi_estimate": mi_estimate,
        "loss_history": history, "model": model, "inference_test": infer_results,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_latent_space(z_phi, z_psi, z_phi_I, z_phi_R, A, rho,
                           output_dir="results_jcf2", prefix="aind_jcf2"):
    os.makedirs(output_dir, exist_ok=True)
    N   = z_phi.size
    idx = np.random.choice(N, min(3000, N), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    data = [
        (z_phi[idx],   z_psi[idx], f"Joint latent (z\u03a6, z\u03a8)\nrho={rho:+.4f}  A={A:+.4f}", "steelblue"),
        (z_phi_I[idx], z_psi[idx], f"Informative z\u03a6\u1d35 = A\u00b7z\u03a8", "darkorange"),
        (z_phi_R[idx], z_psi[idx], "Residual z\u03a6\u1d3f = z\u03a6 \u2212 A\u00b7z\u03a8", "firebrick"),
    ]
    for ax, (xd, yd, title, col) in zip(axes, data):
        r = float(np.corrcoef(xd, yd)[0, 1])
        ax.scatter(xd, yd, s=4, alpha=0.35, color=col)
        ax.set_title(f"{title}\nPearson r = {r:.4f}", fontsize=9)
        ax.set_xlabel("z\u03a6 component"); ax.set_ylabel("z\u03a8"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_latent.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Latent space saved to: {path}")


def visualize_training(history, output_dir="results_jcf2", prefix="aind_jcf2"):
    os.makedirs(output_dir, exist_ok=True)
    n  = len(history["total"])
    xs = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 6, figsize=(30, 4))
    axes[0].plot(xs, history["flow"],   color="steelblue",  lw=2); axes[0].set_title("L_Flow (bivariate NLL)")
    axes[1].semilogy(xs, [max(v,1e-12) for v in history["mse"]],    color="darkorange", lw=2); axes[1].set_title("L_MSE ||z_phi-g_net(z_psi)||^2")
    axes[2].semilogy(xs, [max(v,1e-12) for v in history["indep"]],  color="firebrick",  lw=2); axes[2].set_title("L_Indep HSIC(Phi_I,Phi_R)")
    axes[3].semilogy(xs, [max(v,1e-12) for v in history["smooth"]], color="teal",       lw=2); axes[3].set_title("L_Smooth ||Phi-Phi_I||^2")
    axes[4].plot(xs, history["rho"], color="purple", lw=2, label="\u03c1")
    axes[4].plot(xs, history["A"],   color="green",  lw=2, ls="--", label="A")
    axes[4].axhline(0, color="k", lw=0.5, ls=":"); axes[4].set_title("\u03c1 vs A"); axes[4].legend(fontsize=8)
    axes[5].plot(xs, history["mi_A"],   color="green",  lw=2, label="I~A")
    axes[5].plot(xs, history["mi_rho"], color="purple", lw=1.5, ls="--", label="I~\u03c1")
    axes[5].set_title("MI estimates [nats]"); axes[5].legend(fontsize=8)
    for ax in axes: ax.set_xlabel("Checkpoint"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_training.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Training dynamics saved to: {path}")


def visualize_decomposition(Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true, Phi_R_true,
                            output_dir="results_jcf2", prefix="aind_jcf2"):
    os.makedirs(output_dir, exist_ok=True)
    if Phi_I.ndim == 1:
        Phi_I = Phi_I.reshape(Phi.shape); Phi_R = Phi_R.reshape(Phi.shape)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = ["\u03a6 (Source)\n\u03a6 = f + g",
              "\u03a6\u1d35_true = f\n(True Informative)",
              "\u03a6\u1d35 (Reconstructed)\n[Triangular Flow]",
              "\u03a8\u207a (Target)\n\u03a8\u207a = 0.5f\u00b2 \u2212 0.2f + \u03b5",
              "\u03a6\u1d3f_true = g\n(True Residual)",
              "\u03a6\u1d3f (Reconstructed Residual)"]
    fields = [Phi, Phi_I_true, Phi_I, Psi_plus, Phi_R_true, Phi_R]
    vmin_I = min(Phi_I_true.min(), Phi_I.min()); vmax_I = max(Phi_I_true.max(), Phi_I.max())
    vmin_R = min(Phi_R_true.min(), Phi_R.min()); vmax_R = max(Phi_R_true.max(), Phi_R.max())
    vmin_list = [None, vmin_I, vmin_I, None, vmin_R, vmin_R]
    vmax_list = [None, vmax_I, vmax_I, None, vmax_R, vmax_R]
    for ax, field, title, vmin, vmax in zip(axes.flat, fields, titles, vmin_list, vmax_list):
        kwargs = {"origin": "lower", "cmap": "RdBu", "extent": [0,1,0,2], "aspect": 0.5}
        if vmin is not None: kwargs["vmin"], kwargs["vmax"] = vmin, vmax
        im = ax.imshow(field, **kwargs)
        ax.set_title(title, fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fp = os.path.join(output_dir, f"{prefix}_fields.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Field visualization saved to: {fp}")

    Phi_flat = Phi.ravel(); Psi_flat = Psi_plus.ravel()
    Phi_I_flat = Phi_I.ravel(); Phi_R_flat = Phi_R.ravel(); f_flat = Phi_I_true.ravel()
    idx = np.random.choice(len(Phi_flat), min(5000, len(Phi_flat)), replace=False)
    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    plt.scatter(Phi_I_flat[idx], Psi_flat[idx], s=6, alpha=0.5, label="Data")
    phi_I_range = np.linspace(Phi_I_flat.min(), Phi_I_flat.max(), 200)
    plt.plot(phi_I_range, F_analytical(phi_I_range), "r--", lw=2, label="F(\u03a6\u1d35)")
    plt.xlabel("\u03a6\u1d35"); plt.ylabel("\u03a8\u207a"); plt.title("Mapping \u03a8\u207a \u2248 F(\u03a6\u1d35)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1,3,2)
    plt.scatter(Phi_I_flat[idx], Phi_R_flat[idx], s=6, alpha=0.5, color="red")
    plt.xlabel("\u03a6\u1d35"); plt.ylabel("\u03a6\u1d3f"); plt.title("Independence I(\u03a6\u1d35;\u03a6\u1d3f)"); plt.grid(True, alpha=0.3)
    plt.subplot(1,3,3)
    plt.scatter(f_flat[idx], Phi_I_flat[idx], s=6, alpha=0.5, color="green")
    f_range = np.linspace(f_flat.min(), f_flat.max(), 100)
    plt.plot(f_range, f_range, "k--", lw=2, label="Perfect")
    plt.xlabel("f (true)"); plt.ylabel("\u03a6\u1d35"); plt.title("Reconstruction Quality"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sp = os.path.join(output_dir, f"{prefix}_scatter.png")
    plt.savefig(sp, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Scatter plots saved to: {sp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(nx=100, ny=100, t=0.0,
         n_terms=16, n_bisect=60, hidden_dim=32, n_hidden=2,
         num_epochs=1000, lr=5e-3,
         lambda1=1.0, lambda2=5.0, lambda3=1.0,
         max_hsic_n=2000, output_dir="results_jcf2", seed=1, verbose=True):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    x = np.linspace(0, 1, nx); y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)
    Phi        = phi_func(X, Y, t)
    Psi_plus   = psi_plus_func(X, Y, t, deltaT=0.0, noise_std=1e-8)
    Phi_I_true = f_func(X, Y, t)
    Phi_R_true = g_func(X, Y, t)

    if verbose:
        print("=" * 60)
        print("aIND  --  Triangular Flow  (joint normal, leak-free)")
        print("z_psi=f_psi(Psi), z_phi=f_phi(Phi|z_psi), Phi_I=f_phi^{-1}(A*z_psi|z_psi)")
        print("=" * 60)
        print(f"Grid: {nx}x{ny}  n_terms={n_terms}  hidden_dim={hidden_dim}")
        print(f"Epochs={num_epochs}  lr={lr}  lambda1={lambda1}  lambda2={lambda2}  lambda3={lambda3}")
        mi_gt = mutual_info_regression(
            Phi_I_true.ravel().reshape(-1,1), Phi_R_true.ravel(), random_state=42)[0]
        print(f"\nGround truth I(f; g) = {mi_gt:.6f}  (target: ~0)\n")

    results = aind_decomposition_jcf2(
        Phi, Psi_plus,
        n_terms=n_terms, n_bisect=n_bisect, hidden_dim=hidden_dim, n_hidden=n_hidden,
        num_epochs=num_epochs, lr=lr,
        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
        max_hsic_n=max_hsic_n, verbose=verbose, seed=seed,
    )

    Phi_I = results["Phi_I"].reshape(Phi.shape)
    Phi_R = results["Phi_R"].reshape(Phi.shape)
    metrics = evaluate_decomposition(Phi.ravel(), Psi_plus.ravel(),
                                     results["Phi_I"], results["Phi_R"],
                                     Phi_I_true=Phi_I_true.ravel())

    if verbose:
        print("\n" + "=" * 60)
        print("Final Evaluation Metrics  (Triangular Flow)")
        print("=" * 60)
        print(f"  Learned rho                        : {results['rho']:+.6f}")
        print(f"  Closed-form A                      : {results['A']:+.6f}")
        print(f"  I(Phi;Psi) ~= -0.5*log(1-A^2)     : {results['mi_estimate']:.6f} nats")
        print(f"  I(Phi_I ; Phi_R)                   : {metrics['mutual_information']:.6f}")
        print(f"  Residual energy ||Phi - Phi_I||^2  : {metrics['residual_energy']:.6f}")
        print(f"  MI(Phi_I , Psi+)                   : {metrics['mi_phiI_psi']:.6f}")
        print(f"  MI(Phi_R , Psi+)                   : {metrics['mi_phiR_psi']:.6f}")
        print(f"  GT error ||Phi_I - Phi_I_true||^2  : {metrics['gt_error']:.6f}")
        print("=" * 60)

    path = os.path.join(output_dir, "aind_jcf2_metrics.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("aIND Metrics  (Triangular Flow: joint normal + leak-free)\n")
        fh.write(f"n_terms={n_terms} | hidden_dim={hidden_dim} | epochs={num_epochs}\n")
        fh.write(f"rho={results['rho']:+.6f}  A={results['A']:+.6f}  I~={results['mi_estimate']:.6f} nats\n")
        fh.write("=" * 60 + "\n")
        for name, key in [("I(Phi_I ; Phi_R)", "mutual_information"),
                          ("Residual energy",   "residual_energy"),
                          ("MI(Phi_I, Psi+)",   "mi_phiI_psi"),
                          ("MI(Phi_R, Psi+)",   "mi_phiR_psi"),
                          ("GT error",          "gt_error")]:
            fh.write(f"{name:<35}: {metrics[key]:.6f}\n")
    print(f"Metrics saved to: {path}")

    visualize_latent_space(results["z_phi"], results["z_psi"],
                           results["z_phi_I"], results["z_phi_R"],
                           A=results["A"], rho=results["rho"],
                           output_dir=output_dir, prefix="aind_jcf2")
    visualize_training(results["loss_history"], output_dir=output_dir, prefix="aind_jcf2")
    visualize_decomposition(Phi, Psi_plus, Phi_I, Phi_R, Phi_I_true, Phi_R_true,
                            output_dir=output_dir, prefix="aind_jcf2")
    return results, metrics


if __name__ == "__main__":
    main(
        nx=100, ny=100, t=0.0,
        n_terms=16, n_bisect=60,
        hidden_dim=32, n_hidden=2,
        num_epochs=2000, lr=3e-3,
        lambda1=5.0, lambda2=150.0, lambda3=0.50,
        max_hsic_n=2000,
        output_dir="results_jcf2",
        seed=1, verbose=True,
    )
