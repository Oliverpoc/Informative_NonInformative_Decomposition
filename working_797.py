"""
aind_decomposition_fields.py
------------------------------------------------------------
Improved visual version of the aIND (Eq. 2.10) validation.
Shows spatial maps of Φ, Ψ⁺, Φᴵ, and Φᴿ to illustrate the decomposition.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error

# ============================================================
# Step 1: Analytical test case (same as paper)
# ============================================================

def f_func(x, y, t):
    return 2 * np.sin(2*np.pi*x - 2*t) * np.sin(2*np.pi*y)

def g_func(x, y, t):
    return (1/5) * np.sin(7*np.sqrt(2)*np.pi*x - 0.1*t) * np.sin(8*np.sqrt(3)*np.pi*y - 0.5*t)

def phi_func(x, y, t):
    return f_func(x, y, t) + g_func(x, y, t)


# ============================================================
# Step 2: Analytical mapping (constraint Ψ⁺ = F(Φᴵ))
# ============================================================

def F(phi_I):
    return 0.5 * phi_I**2 + 3.0 * phi_I # NEW: analytic forward mapping F(Φ) = 0.5 Φ^2 + 3 Φ

def psi_plus_func(x, y, t, noise_std=0.1):
    f = f_func(x, y, t)
    eps = np.random.normal(0, noise_std, size=f.shape)
    return F(f) + eps

# Grid
nx, ny = 80, 80
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
t = 0.0

# flattened vectors
Phi = phi_func(X, Y, t)
Psi_plus = psi_plus_func(X, Y, t)
Phi_flat = Phi.ravel()
Psi_flat = Psi_plus.ravel()

# ============================================================
# Step 3: Neural network for Φᴵ = F⁻¹(Ψ⁺) using the full aIND loss (Eq. 2.10)
# ============================================================
# Implements:
#     L = I(Φᴿ; Φᴵ) + γ ||Φ − Φᴵ||²
# where Φᴿ = Φ − Φᴵ is called Residue  and  Φᴵ = NN(Ψ⁺), the ground truth of NN is F-inverse
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F_torch

# ----------------------------------------------------------------
# Device selection and NumPy → PyTorch tensors
# ----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Psi_t = torch.tensor(Psi_flat, dtype=torch.float32, device=device).view(-1, 1)   # target field Ψ⁺ (input)
Phi_t = torch.tensor(Phi_flat, dtype=torch.float32, device=device).view(-1, 1)   # source field Φ (target)

# Qi's note: This now is changed to F inverse, instead of the original F. Make sure you understand when to use which
class LearnableMapping(nn.Module):
    """
    Learnable mapping Ψ⁺ ≈ a * (Φᴵ)^2 + b * Φᴵ.
    a, b are trainable; initialized near 0.5 and -0.2.
    """

    """
    Learnable inverse quadratic:
        Φᴵ(Ψ⁺) ≈ F^{-1}(Ψ⁺)
    where F(φ) = a φ^2 + b φ.

    We learn (a, b) and use the analytic inverse formula.
    """
    def __init__(self, a_init=0.49, b_init=2.99):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))
        # This is the a minefield
        # Check if this is the right way to write computations in torch
        # self.a = self.b**2/4.0 - torch.exp(self.sigma)

    def forward(self, Psi_plus):
        # return self.a * phi_I**2 + self.b * phi_I
        # Qi's note: new version
        #return -(self.b + torch.sqrt(self.b**2 + 4*self.a*Psi_plus))/self.a/2
        a = self.a
        b = self.b
        disc = b**2 + 4.0 * a * Psi_plus
        disc = torch.clamp(disc, min=1e-8)
        phi = (-b + torch.sqrt(disc)) / (2.0 * a)
        return phi



F_net = LearnableMapping(a_init=0.49, b_init=2.99).to(device)  # for example, “guess” values

# only use this after we make sure the quadratic function can be learned

"""class MonotoneScalarNN(nn.Module):
    """
    #Simpler monotone 1D -> 1D neural network:

        #Φᴵ(Ψ⁺) = c + sum_k alpha_k * softplus(w_k * Ψ⁺ + b_k)

    #with w_k > 0, alpha_k > 0 so mapping is strictly increasing.
    #Using fewer units (e.g. 4) because the mapping here is simple.
"""
    def __init__(self, num_units=4):
        super().__init__()
        self.w_raw = nn.Parameter(torch.randn(num_units))      # unconstrained -> softplus -> >0
        self.b = nn.Parameter(torch.zeros(num_units))          # biases
        self.alpha_raw = nn.Parameter(torch.randn(num_units))  # unconstrained -> softplus -> >0
        self.c = nn.Parameter(torch.zeros(1))                  # scalar offset

    def forward(self, x):
        # x: [N, 1]
        x = x.view(-1, 1)  # ensure shape

        # enforce positivity for monotonicity
        w = F_torch.softplus(self.w_raw)          # [num_units] > 0
        alpha = F_torch.softplus(self.alpha_raw)  # [num_units] > 0

        # broadcast to [N, num_units]
        wx_plus_b = x * w.view(1, -1) + self.b.view(1, -1)

        # monotone nonlinearity
        h = F_torch.softplus(wx_plus_b)  # [N, num_units], increasing in x

        # weighted sum
        y = self.c + (h * alpha.view(1, -1)).sum(dim=1, keepdim=True)  # [N, 1]
        return y


# net = MonotoneScalarNN(num_units=4)

# class MonotoneScalarNN(nn.Module):
#     """
#     Monotone 1D -> 1D neural network:
#         Φᴵ(Ψ⁺) = c + sum_k alpha_k * softplus(w_k * Ψ⁺ + b_k)
#     with w_k > 0, alpha_k > 0 so the mapping is strictly increasing
#     and therefore invertible on its range.
#     """
#     def __init__(self, num_units=32):
#         super().__init__()
#         # unconstrained parameters
#         self.w_raw = nn.Parameter(torch.randn(num_units))      # will become >0
#         self.b = nn.Parameter(torch.zeros(num_units))          # biases
#         self.alpha_raw = nn.Parameter(torch.randn(num_units))  # will become >0
#         self.c = nn.Parameter(torch.zeros(1))                  # scalar offset

#     def forward(self, x):
#         # x: [N, 1]
#         x = x.view(-1, 1)  # ensure shape

#         # enforce positivity
#         w = F_torch.softplus(self.w_raw)          # [num_units], > 0
#         alpha = F_torch.softplus(self.alpha_raw)  # [num_units], > 0

#         # broadcast to shape [N, num_units]
#         # x: [N,1], w: [num_units] -> [N, num_units]
#         wx_plus_b = x * w.view(1, -1) + self.b.view(1, -1)  # [N, num_units]

#         # monotone nonlinearity
#         h = F_torch.softplus(wx_plus_b)           # [N, num_units], increasing in x

#         # weighted sum with positive alpha
#         y = self.c + (h * alpha.view(1, -1)).sum(dim=1, keepdim=True)  # [N,1]
#         return y



# ----------------------------------------------------------------
# Define neural network: Φᴵ = B(Ψ⁺) ≈ F⁻¹(Ψ⁺)
# Monotone scalar network -> invertible on its range
# ----------------------------------------------------------------
#net = MonotoneScalarNN(num_units=32)



# ----------------------------------------------------------------
# Differentiable Mutual Information estimator via Gaussian KDE
# for scalar Φᴿ and Φᴵ (Option B: aIND scalar setting)
# ----------------------------------------------------------------
def mi_kde(phi_r, phi_i, bandwidth=0.3, eps=1e-9):
    """
    Estimate I(Φᴿ; Φᴵ) using a Gaussian kernel density estimator.
    phi_r, phi_i: tensors of shape [N, 1]
    Returns a scalar MI estimate (PyTorch tensor) that is differentiable
    w.r.t. phi_r, phi_i (and thus w.r.t. network parameters).
    """
    # Ensure shapes [N, 1]
    x = phi_r.view(-1, 1)
    y = phi_i.view(-1, 1)
    N = x.size(0)

    # Pairwise differences (N,N)
    diff_x = x - x.t()         # Φᴿ_i - Φᴿ_j
    diff_y = y - y.t()         # Φᴵ_i - Φᴵ_j

    # Squared distances
    dist_x2 = (diff_x ** 2)    # [N, N]
    dist_y2 = (diff_y ** 2)    # [N, N]

    # Gaussian kernels (no normalization constants – they cancel in MI up to a constant)
    Kx = torch.exp(-0.5 * dist_x2 / (bandwidth ** 2))       # p(x)
    Ky = torch.exp(-0.5 * dist_y2 / (bandwidth ** 2))       # p(y)
    Kxy = torch.exp(-0.5 * (dist_x2 + dist_y2) / (bandwidth ** 2))  # p(x,y)

    # Parzen density estimates: average over j
    px = Kx.mean(dim=1)   # [N]
    py = Ky.mean(dim=1)   # [N]
    pxy = Kxy.mean(dim=1) # [N]

    # Mutual information estimate:
    # I ≈ E[ log p(x,y) - log p(x) - log p(y) ]
    mi_sample = torch.log(pxy + eps) - torch.log(px + eps) - torch.log(py + eps)
    mi = mi_sample.mean()
    return mi


#optimizer = optim.Adam(net.parameters(), lr=1e-3)
#optimizer = optim.Adam(
    #list(net.parameters()) + list(F_net.parameters()),
    #lr=1e-3
#)



gamma = 1e-3        # weight for residual energy term ||Φ - Φᴵ||²
#lambda_map = 1.0    # weight for mapping term ||Ψ⁺ - F_net(Φᴵ)||²

optimizer = optim.Adam(list(F_net.parameters()), lr=1e-3)







# ----------------------------------------------------------------
# Define Mutual Information estimator (MINE-style)
# It learns to approximate I(Φᴿ; Φᴵ)
# ----------------------------------------------------------------
#class MIEstimator(nn.Module):
    #def __init__(self, hidden_dim=64):
        #super().__init__()
        #self.net = nn.Sequential(
            #nn.Linear(2, hidden_dim), nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            #nn.Linear(hidden_dim, 1)
        #)

    #def forward(self, x, y):
        #"""
        #Estimate mutual information I(x; y) using the MINE objective.
        #x : Φᴿ (residual)
        #y : Φᴵ (informative)
        #"""
        # "Joint" samples (true paired data)
        #joint = torch.cat([x, y], dim=1)
        # "Marginal" samples (shuffled to break dependence)
        #marginal = torch.cat([x[torch.randperm(len(x))], y], dim=1)

        # Pass through small neural net
        #T_joint = self.net(joint)
        #T_marginal = self.net(marginal)

        # MINE mutual information estimate:
        # E[T_joint] - log(E[exp(T_marginal)])
        #return torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))

# Instantiate MI estimator network
#mi_est = MIEstimator()

# ----------------------------------------------------------------
# Optimizer for both networks (main net + MI estimator)
# ----------------------------------------------------------------
#optimizer = optim.Adam(list(net.parameters()) + list(mi_est.parameters()), lr=1e-3)

# Regularization coefficient γ for the reconstruction term
#gamma = 1e-3

# ----------------------------------------------------------------
# Training loop: jointly minimize aIND loss
# ----------------------------------------------------------------
num_epochs = 3000
history_total = []
history_mi = []
history_resE = []
history_map = []
for epoch in range(num_epochs):
    # ---- Forward pass ----
    # Φᴵ = NN(Ψ⁺)
    Phi_I = F_net(Psi_t)          # [N, 1]
    Phi_R = Phi_t - Phi_I       # [N, 1]

    # Mutual information term I(Φᴿ; Φᴵ)
    mi_term = mi_kde(Phi_R, Phi_I, bandwidth=0.3)

    # Residual energy term ||Φ − Φᴵ||²
    residual_energy_term = torch.mean((Phi_t - Phi_I) ** 2)

    # Mapping term ||Ψ⁺ − F_net(Φᴵ)||²
    Psi_hat = F_net(Phi_I)
    mapping_term = torch.mean((Psi_t - Psi_hat) ** 2)

    # Full aIND-style loss:
    #   I(Φᴿ; Φᴵ) + γ ||Φ − Φᴵ||² + λ_map ||Ψ⁺ − F(Φᴵ)||²
    total_loss = mi_term + gamma * residual_energy_term

    # ---- Backprop ----
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # ---- Save history ----
    history_total.append(total_loss.item())
    history_mi.append(mi_term.item())
    history_resE.append(residual_energy_term.item())
    history_map.append(mapping_term.item())

    # ---- Logging ----
    if epoch % 200 == 0:
        print(
            f"Epoch {epoch:04d} | "
            f"Total={total_loss.item():.6f} "
            f"| MI={mi_term.item():.6f} "
            f"| ResE={residual_energy_term.item():.6f} "
            f"| Map={mapping_term.item():.6f} "
            f"| a={F_net.a.item():.3f} "
            f"| b={F_net.b.item():.3f}"
        )




# ----------------------------------------------------------------
# Store learned fields
# ----------------------------------------------------------------
# Detach from computation graph and convert to NumPy
Phi_I_pred = F_net(Psi_t).detach().cpu().numpy().flatten()   # informative field Φᴵ
Phi_R_pred = Phi_flat - Phi_I_pred                   # residual field Φᴿ


# After training, with F_net already trained

Psi_sample = Psi_t.detach().cpu().numpy().flatten()
Phi_I_learned = F_net(Psi_t).detach().cpu().numpy().flatten()

# Sort by Psi so the curve looks clean
idx = np.argsort(Psi_sample)
Psi_sorted = Psi_sample[idx]
Phi_learned_sorted = Phi_I_learned[idx]

# Analytic inverse for comparison
Phi_inv_analytic = -3 + np.sqrt(9 + 2 * Psi_sorted)

plt.figure(figsize=(6, 4))
plt.plot(Psi_sorted, Phi_inv_analytic, label="Analytic F⁻¹(Ψ⁺)")
plt.plot(Psi_sorted, Phi_learned_sorted, "--", label="Learned F_net(Ψ⁺)")
plt.xlabel("Ψ⁺")
plt.ylabel("Φ")
plt.title("Learned inverse vs analytic inverse")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# ============================================================
# Step 3: Neural network for Φᴵ = F⁻¹(Ψ⁺)
# ============================================================

#nn_model = MLPRegressor(hidden_layer_sizes=(64, 64),
                         #activation='tanh',
                         #max_iter=3000,
                         #learning_rate_init=0.001,
                         #random_state=42)

# Train inverse model (Ψ⁺ → Φᴵ)
#nn_model.fit(Psi_flat.reshape(-1, 1), Phi_flat)
#Phi_I_pred = nn_model.predict(Psi_flat.reshape(-1, 1))

# Compute residual
#Phi_R_pred = Phi_flat - Phi_I_pred



# ============================================================
# Step 4: Evaluate loss components (Eq. 2.10)
# ============================================================

mi_term = mutual_info_regression(Phi_R_pred.reshape(-1, 1), Phi_I_pred, random_state=42)[0]
residual_energy_term = mean_squared_error(Phi_flat, Phi_I_pred)
map_term = mean_squared_error(Psi_flat, F(Phi_I_pred))
total_loss = mi_term +  residual_energy_term 

print("\n===== aIND Decomposition Summary (Eq. 2.10) =====")
print(f"Mutual Info I(Φᴿ;Φᴵ): {mi_term:.5f}")
print(f"Residual Energy ||Φ-Φᴿ||²: {residual_energy_term:.5f}")
print(f"Mapping ||Ψ⁺-F(Φᴵ)||²: {map_term:.5f}")
print(f"Total Loss: {total_loss:.5f}")

# Reshape for visualization
Phi_I_field = Phi_I_pred.reshape(Phi.shape)
Phi_R_field = Phi_R_pred.reshape(Phi.shape)

# ============================================================
# Step 5: Visualize fields (like Figure 2 in paper)
# ============================================================

plt.figure(figsize=(12, 4))
titles = ["Φ (Total Field)", "Ψ⁺ (Target Field)",
          "Φᴵ (Informative Component)", "Φᴿ (Residual Component)"]
fields = [Phi, Psi_plus, Phi_I_field, Phi_R_field]

for i, (F_, title) in enumerate(zip(fields, titles), 1):
    plt.subplot(1, 4, i)
    im = plt.imshow(F_, origin="lower", cmap="coolwarm")
    plt.title(title)
    plt.colorbar(im, fraction=0.046)

plt.tight_layout()
plt.show()

# ============================================================
# Step 6: Relationship scatter plots
# ============================================================

plt.figure(figsize=(12, 4))

# Ψ⁺ vs Φᴵ
plt.subplot(1, 3, 1)
plt.scatter(Phi_I_pred[::200], Psi_flat[::200], s=6, alpha=0.5)
plt.xlabel("Φᴵ (Informative)")
plt.ylabel("Ψ⁺")
plt.title("Mapping Ψ⁺ ≈ F(Φᴵ)")

# Φᴿ vs Φᴵ (Independence)
plt.subplot(1, 3, 2)
plt.scatter(Phi_I_pred[::200], Phi_R_pred[::200], s=6, alpha=0.5, color='red')
plt.xlabel("Φᴵ")
plt.ylabel("Φᴿ")
plt.title("Residual Independence")

# Φᴵ vs Φ (Residual energy term)
plt.subplot(1, 3, 3)
plt.scatter(Phi_flat[::200], Phi_I_pred[::200], s=6, alpha=0.5, color='green')
plt.xlabel("Φ (Total)")
plt.ylabel("Φᴵ (Informative)")
plt.title("Φᴵ close to Φ (Residual Energy Term)")

plt.tight_layout()
plt.show()





# ----------------------------------------------------------------
# Plot training dynamics: epoch vs loss components
# ----------------------------------------------------------------
epochs = np.arange(num_epochs)

plt.figure(figsize=(10, 6))
plt.plot(epochs, history_total, label="Total loss")
plt.plot(epochs, history_mi, label="MI(Φᴿ; Φᴵ)")
plt.plot(epochs, history_resE, label="Residual ||Φ - Φᴵ||²")
plt.plot(epochs, history_map, label="Mapping ||Ψ⁺ - F(Φᴵ)||²")
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.title("aIND training dynamics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
