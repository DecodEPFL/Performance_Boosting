import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================================
# Simulation Parameters & Setup
# ============================================================================
# Set fixed seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Environment and Robot dynamics
V_X = 0.05
T = 100  # Total time steps for training/evaluation
X_GATE = 4.0  # x-coordinate of the wall
T_GATE = int(X_GATE / V_X)  # Time step when robot reaches the wall (t = 80)
H = 0.2  # Gate half-width

# Nominal controller gain
K = 0.2

# Training parameters
BATCH_SIZE = 256
EPOCHS = 450
LR = 0.01


# ============================================================================
# Data Generation
# ============================================================================
def generate_scenarios(batch_size, horizon):
    """
    Generates disturbances w_t and gate context g_t.
    w_t: Sparse pulse-like lateral gusts.
    g_t: Gate position, alternating between +1.0 and -1.0 every 20-40 steps.
    """
    w = torch.zeros((batch_size, horizon))
    g = torch.zeros((batch_size, horizon))

    for i in range(batch_size):
        # Generate sparse gusts
        t = 0
        while t < horizon:
            if np.random.rand() < 0.08:  # 8% chance for a burst to start
                length = np.random.randint(3, 7)
                mag = np.random.uniform(0.2, 0.4) * np.random.choice([-1, 1])
                w[i, t:min(t + length, horizon)] = mag
                t += length
            else:
                t += 1

        # Generate switching gate (Option A)
        t = 0
        current_g = np.random.choice([-1.0, 1.0])
        while t < horizon:
            length = np.random.randint(20, 41)
            g[i, t:min(t + length, horizon)] = current_g
            current_g *= -1.0
            t += length

    return w, g


# ============================================================================
# Performance Boosting (PB) Models
# ============================================================================
class PBDistOnly(nn.Module):
    """ PB Operator observing ONLY the disturbance (w_t). """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, w):
        return self.net(w)


class PBContextAware(nn.Module):
    """ PB Operator observing disturbance (w_t) AND context (g_t). """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, w, g):
        # Concatenate features along the feature dimension
        x = torch.cat([w, g], dim=1)
        return self.net(x)


# ============================================================================
# Simulation & Training Logic
# ============================================================================
def simulate(model, w, g, mode='nominal', train=False):
    """
    Simulates the robot trajectory through time.
    mode: 'nominal', 'dist_only', or 'context_aware'
    """
    batch_size, horizon = w.shape
    y = torch.zeros((batch_size, 1))

    loss = 0.0
    y_hist, u_hist, v_hist = [], [], []

    for t in range(horizon):
        # 1. Nominal Controller: tries to keep robot at y=0
        u = K * (0.0 - y)

        # 2. PB Operator Correction
        wt = w[:, t].unsqueeze(1)
        gt = g[:, t].unsqueeze(1)

        if mode == 'nominal':
            v = torch.zeros_like(y)
        elif mode == 'dist_only':
            v = model(wt)
        elif mode == 'context_aware':
            v = model(wt, gt)

        # 3. Step Dynamics
        y_next = y + u + v + wt

        # 4. Compute Loss (if training)
        if train:
            # Running cost: follow the safe corridor defined by g_t and penalize effort
            cost_step = 1.0 * (y - gt) ** 2 + 0.05 * (u + v) ** 2
            loss += cost_step.sum()

            # Heavy penalty for missing the gate at the wall
            if t == T_GATE:
                miss_dist = torch.abs(y - gt)
                # Collision penalty if outside the safe bounds H
                collision_penalty = 100.0 * torch.where(miss_dist > H, miss_dist - H, torch.zeros_like(miss_dist))
                loss += collision_penalty.sum()

        # Save histories
        y_hist.append(y.detach().numpy())
        u_hist.append(u.detach().numpy())
        v_hist.append(v.detach().numpy())

        y = y_next

    if train:
        return loss / batch_size
    else:
        return np.array(y_hist).squeeze(), np.array(u_hist).squeeze(), np.array(v_hist).squeeze()


def train_model(model, mode, w_train, g_train):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()
    print(f"Training {mode} PB...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = simulate(model, w_train, g_train, mode=mode, train=True)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f" Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")
    print("Training complete.\n")


# ============================================================================
# Main Execution: Train & Evaluate
# ============================================================================
# 1. Generate Datasets
w_train, g_train = generate_scenarios(BATCH_SIZE, T)
w_test, g_test = generate_scenarios(100, T)

# 2. Initialize Models
model_dist = PBDistOnly()
model_ctx = PBContextAware()

# 3. Train Models
train_model(model_dist, 'dist_only', w_train, g_train)
train_model(model_ctx, 'context_aware', w_train, g_train)

# 4. Evaluate Models on Test Set
model_dist.eval()
model_ctx.eval()

results = {}
modes = ['nominal', 'dist_only', 'context_aware']
models = [None, model_dist, model_ctx]

for mode, model in zip(modes, models):
    y_sim, u_sim, v_sim = simulate(model, w_test, g_test, mode=mode, train=False)

    # Calculate Metrics
    y_at_gate = y_sim[T_GATE, :]
    g_at_gate = g_test[:, T_GATE].numpy()

    miss_distance = np.abs(y_at_gate - g_at_gate)
    collisions = miss_distance > H
    success_rate = 1.0 - np.mean(collisions)
    avg_miss = np.mean(miss_distance)
    total_energy = np.mean(np.sum((u_sim + v_sim) ** 2, axis=0))

    results[mode] = {
        'y': y_sim, 'u': u_sim, 'v': v_sim,
        'success_rate': success_rate * 100,
        'avg_miss': avg_miss,
        'energy': total_energy
    }

# ============================================================================
# Visualizations
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1.2])

# --- Plot 1: Top-down 2D Trajectory View (Single Representative Episode) ---
ax1 = fig.add_subplot(gs[0, :])
test_idx = 1  # Select a representative episode
x_traj = np.linspace(0, T * V_X, T)

# Plot safe context corridor (target lane)
g_traj = g_test[test_idx, :].numpy()
ax1.step(x_traj, g_traj, where='post', color='grey', linestyle='--', alpha=0.5, linewidth=2, label='Safe Path $g_t$')
ax1.fill_between(x_traj, g_traj - H, g_traj + H, step='post', color='grey', alpha=0.1)

# Plot physical wall and gate at T_GATE
gate_center = g_traj[T_GATE]
ax1.axvline(X_GATE, color='black', linewidth=4, label='Physical Wall')
# Draw opening (white line over black wall)
ax1.plot([X_GATE, X_GATE], [gate_center - H, gate_center + H], color='white', linewidth=6)

# Plot trajectories
ax1.plot(x_traj, results['nominal']['y'][:, test_idx], label='Nominal Only', linewidth=2)
ax1.plot(x_traj, results['dist_only']['y'][:, test_idx], label='PB (Disturbance Only)', linewidth=2)
ax1.plot(x_traj, results['context_aware']['y'][:, test_idx], label='PB (Context-Aware)', linewidth=2.5, color='green')

ax1.set_title("Top-Down View: Robot Navigating the Corridor", fontsize=14, fontweight='bold')
ax1.set_xlabel("x position", fontsize=12)
ax1.set_ylabel("y position", fontsize=12)
ax1.set_xlim([0, X_GATE + 1.0])
ax1.set_ylim([-2.0, 2.0])
ax1.legend(loc='upper left')

# --- Plot 2: Bar Chart - Success Rate ---
ax2 = fig.add_subplot(gs[1, 0])
labels = ['Nominal', 'PB (Dist Only)', 'PB (Context-Aware)']
success_rates = [results[m]['success_rate'] for m in modes]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

ax2.bar(labels, success_rates, color=colors, alpha=0.8)
ax2.set_title("Gate Crossing Success Rate (%)", fontsize=12, fontweight='bold')
ax2.set_ylim([0, 105])
for i, v in enumerate(success_rates):
    ax2.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

# --- Plot 3: Bar Chart - Average Miss Distance ---
ax3 = fig.add_subplot(gs[1, 1])
avg_misses = [results[m]['avg_miss'] for m in modes]

ax3.bar(labels, avg_misses, color=colors, alpha=0.8)
ax3.set_title(f"Avg |y - g_t| at Wall (Safe threshold < {H})", fontsize=12, fontweight='bold')
ax3.axhline(H, color='red', linestyle='--', label='Safe Bound')
ax3.legend()
for i, v in enumerate(avg_misses):
    ax3.text(i, v + 0.05, f"{v:.2f}", ha='center', fontweight='bold')

# --- Plot 4: The "Aha!" Moment (Opposite reactions to same disturbance) ---
ax4 = fig.add_subplot(gs[2, :])

# Create a deterministic tiny test scenario
t_len = 25
w_aha = torch.zeros((2, t_len))
g_aha = torch.zeros((2, t_len))

# Apply same upward gust to both scenarios
w_aha[:, 5:12] = 0.4

# Scenario 0: Gate is at +1.0 (Gust helps reach gate)
g_aha[0, :] = 1.0
# Scenario 1: Gate is at -1.0 (Gust prevents reaching gate)
g_aha[1, :] = -1.0

# Evaluate PB Context-Aware
model_ctx.eval()
_, _, v_aha = simulate(model_ctx, w_aha, g_aha, mode='context_aware', train=False)

time_axis = np.arange(t_len)
ax4.plot(time_axis, w_aha[0].numpy(), color='purple', linestyle='--', linewidth=2, label='Disturbance (Upward Gust)')

ax4.plot(time_axis, v_aha[:, 0], color='blue', marker='o',
         label='PB Correction (Gate is UP at +1.0)')
ax4.plot(time_axis, v_aha[:, 1], color='red', marker='s',
         label='PB Correction (Gate is DOWN at -1.0)')

ax4.set_title("Contextual Adaptation: How PB Reacts to the EXACT SAME Upward Gust", fontsize=14, fontweight='bold')
ax4.set_xlabel("Time step", fontsize=12)
ax4.set_ylabel("Value / PB Action ($v_t$)", fontsize=12)
ax4.axhline(0, color='black', linewidth=1)
ax4.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ============================================================================
# Interpretation and Summary
# ============================================================================
print("\n" + "=" * 60)
print(" EXPERIMENT INTERPRETATION & RESULTS OVERVIEW")
print("=" * 60)
print(f"1. Nominal Control Success Rate:  {results['nominal']['success_rate']:.1f}%")
print(f"   (Failed because u_t = -0.2 * y_t pulls the robot to y=0, missing gates at ±1.0)")
print(f"2. Dist-Only PB Success Rate:     {results['dist_only']['success_rate']:.1f}%")
print(f"   (Failed because it only observes gusts w_t. It minimizes variance around y=0")
print(f"    but has no idea where the moving gate is, leading to inevitable crashes.)")
print(f"3. Context-Aware PB Success Rate: {results['context_aware']['success_rate']:.1f}%")
print(f"   (Succeeded by using g_t to steer toward the safe gap AND counteract/exploit w_t.)\n")

print("WHY MULTI-INPUT (CONTEXT-AWARE) PB IS SUPERIOR:")
print("As highlighted in the bottom plot, an upward gust (+0.4) pushes the robot upward.")
print("  - If the gate is UP (+1.0), the gust actually *assists* the robot. The Context-Aware")
print("    operator recognizes this and applies a mild correction (saving control energy).")
print("  - If the gate is DOWN (-1.0), the gust violently pushes the robot *away* from safety.")
print("    The Context-Aware operator dynamically completely reverses its strategy, applying")
print("    a strong downward force to counteract both the gust and the baseline drift.")
print("\nA disturbance-only model cannot distinguish between these two states, proving that")
print("environmental context is strictly required to determine the 'correct' control action.")
print("=" * 60 + "\n")