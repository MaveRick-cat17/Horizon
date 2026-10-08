import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ================================================================
# HORIZON — FORWARD SIMULATION FROM DISCOVERED PHYSICS
# ================================================================

INPUT = "horizon_pendulum_v12.npz"

# Final physics discovered by Horizon
G_OVER_L = 28.334142
DAMPING = 0.040713


print("=" * 70)
print("HORIZON — FORWARD SIMULATION")
print("=" * 70)


# ================================================================
# LOAD EXPERIMENTAL DATA
# ================================================================

data = np.load(INPUT)

t = data["time"]
theta_deg = data["theta"]

valid = (
    np.isfinite(t)
    & np.isfinite(theta_deg)
)

t = t[valid]
theta_deg = theta_deg[valid]

theta = np.deg2rad(theta_deg)


# ================================================================
# FIND INITIAL CONDITIONS
# ================================================================

# Estimate angular velocity from measured trajectory
theta_dot = np.gradient(theta, t)

theta0 = float(theta[0])
theta_dot0 = float(theta_dot[0])

simulation_time = t.copy()


print()
print("INITIAL CONDITIONS")
print("=" * 70)

print(f"Initial time       : {t[0]:.3f} s")
print(f"Initial theta      : {np.degrees(theta0):.3f} deg")
print(f"Initial theta_dot  : {theta_dot0:.3f} rad/s")


# ================================================================
# DISCOVERED DYNAMICS
#
# θ̈ = -28.334142 sin(θ) - 0.040713 θ̇
#
# State:
#
# y[0] = θ
# y[1] = θ̇
# ================================================================

def discovered_pendulum(t, state):

    theta = state[0]
    theta_dot = state[1]

    theta_ddot = (
        -G_OVER_L * np.sin(theta)
        -DAMPING * theta_dot
    )

    return [
        theta_dot,
        theta_ddot
    ]


# ================================================================
# FORWARD SIMULATION
# ================================================================

print()
print("=" * 70)
print("SIMULATING DISCOVERED PHYSICS")
print("=" * 70)

solution = solve_ivp(
    discovered_pendulum,
    (
        simulation_time[0],
        simulation_time[-1]
    ),
    [
        theta0,
        theta_dot0
    ],
    t_eval=simulation_time,
    rtol=1e-8,
    atol=1e-10
)


if not solution.success:

    raise RuntimeError(
        "Forward simulation failed."
    )


theta_sim = solution.y[0]
theta_dot_sim = solution.y[1]


# ================================================================
# ERROR ANALYSIS
# ================================================================

error = theta - theta_sim

rmse_rad = np.sqrt(
    np.mean(error ** 2)
)

rmse_deg = np.degrees(rmse_rad)

max_error_deg = np.degrees(
    np.max(np.abs(error))
)


print()
print("=" * 70)
print("SIMULATION RESULTS")
print("=" * 70)

print(f"Simulation duration : "
      f"{simulation_time[-1] - simulation_time[0]:.2f} s")

print(f"RMSE                 : "
      f"{rmse_deg:.3f} degrees")

print(f"Maximum error        : "
      f"{max_error_deg:.3f} degrees")


# ================================================================
# PLOT — MEASURED VS SIMULATED
# ================================================================

plt.figure(figsize=(12, 6))

plt.plot(
    t,
    np.degrees(theta),
    label="Measured from video",
    linewidth=1.5
)

plt.plot(
    simulation_time,
    np.degrees(theta_sim),
    label="Forward simulation",
    linewidth=2
)

plt.xlabel("Time (s)")
plt.ylabel("θ (degrees)")

plt.title(
    "Horizon — Measured vs Forward-Simulated Pendulum Motion"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "horizon_forward_simulation.png",
    dpi=200
)

plt.show()


# ================================================================
# PHASE SPACE COMPARISON
# ================================================================

plt.figure(figsize=(9, 6))

plt.plot(
    theta,
    theta_dot,
    label="Measured"
)

plt.plot(
    theta_sim,
    theta_dot_sim,
    "--",
    label="Forward simulation"
)

plt.xlabel("θ (rad)")
plt.ylabel("θ̇ (rad/s)")

plt.title(
    "Horizon — Measured vs Simulated Phase Space"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "horizon_forward_phase_space.png",
    dpi=200
)

plt.show()


# ================================================================
# SAVE SIMULATION RESULTS
# ================================================================

np.savez(
    "horizon_forward_simulation.npz",

    time=simulation_time,

    measured_theta=theta,

    measured_theta_dot=theta_dot,

    simulated_theta=theta_sim,

    simulated_theta_dot=theta_dot_sim,

    g_over_L=G_OVER_L,

    damping=DAMPING,

    rmse_degrees=rmse_deg,

    max_error_degrees=max_error_deg
)


print()
print("=" * 70)
print("HORIZON — FORWARD SIMULATION COMPLETE")
print("=" * 70)

print()
print("Generated:")

print("  horizon_forward_simulation.png")
print("  horizon_forward_phase_space.png")
print("  horizon_forward_simulation.npz")

print()
print("DISCOVERED EQUATION:")
print()
print(
    "    θ̈ = -28.334142 sin(θ) - 0.040713 θ̇"
)

print()
print("Horizon has now gone:")
print()
print("    VIDEO")
print("      ↓")
print("    TRACKING")
print("      ↓")
print("    θ(t)")
print("      ↓")
print("    SINDy")
print("      ↓")
print("    DISCOVERED PHYSICS")
print("      ↓")
print("    FORWARD SIMULATION")
print("      ↓")
print("    PREDICTION")
print()
print("=" * 70)