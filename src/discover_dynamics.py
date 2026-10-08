import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


INPUT = "horizon_pendulum_v12.npz"


print("=" * 70)
print("HORIZON — PHYSICS-CONSTRAINED SINDy")
print("=" * 70)


# ================================================================
# LOAD V12 DATA
# ================================================================

data = np.load(INPUT)

t = data["time"]
theta_deg = data["theta"]

L_pixels = float(data["pendulum_length"])

print(f"Samples              : {len(t)}")
print(f"Pendulum length      : {L_pixels:.2f} pixels")
print(
    f"Angle range          : "
    f"{np.nanmin(theta_deg):.2f}° → "
    f"{np.nanmax(theta_deg):.2f}°"
)


# ================================================================
# DEGREES -> RADIANS
# ================================================================

theta = np.deg2rad(theta_deg)


# ================================================================
# REMOVE INVALID DATA
# ================================================================

valid = (
    np.isfinite(t)
    & np.isfinite(theta)
)

t = t[valid]
theta = theta[valid]


# ================================================================
# TIME STEP
# ================================================================

dt = np.median(np.diff(t))

print(f"Median Δt            : {dt:.6f} s")
print(f"Sampling frequency   : {1.0 / dt:.2f} Hz")


# ================================================================
# SMOOTH ANGLE
# ================================================================

window = 11

if window >= len(theta):
    window = len(theta) - 1

if window % 2 == 0:
    window -= 1

theta_smooth = savgol_filter(
    theta,
    window_length=window,
    polyorder=3
)


# ================================================================
# DERIVATIVES
# ================================================================

theta_dot = np.gradient(
    theta_smooth,
    t
)

theta_ddot = np.gradient(
    theta_dot,
    t
)


# ================================================================
# PHYSICS-CONSTRAINED LIBRARY
#
# We deliberately use ONLY the terms relevant to a damped
# pendulum:
#
#       sin(theta)
#       theta_dot
#
# Therefore SINDy searches for:
#
#       theta_ddot =
#           A sin(theta)
#           + B theta_dot
#
# This avoids the previous redundant combination:
#
#       theta + sin(theta) + cos(theta) + ...
# ================================================================

features = np.column_stack([
    np.sin(theta_smooth),
    theta_dot
])


feature_names = [
    "sin(theta)",
    "theta_dot"
]


# ================================================================
# STLSQ SPARSE REGRESSION
# ================================================================

optimizer = ps.STLSQ(
    threshold=0.01,
    alpha=0.0001
)


optimizer.fit(
    features,
    theta_ddot
)


coefficients = optimizer.coef_.flatten()


# ================================================================
# DISCOVERED EQUATION
# ================================================================

sin_theta_coeff = coefficients[0]
damping_coeff = coefficients[1]


print()
print("=" * 70)
print("DISCOVERED PENDULUM DYNAMICS")
print("=" * 70)

print()
print("Horizon discovered:")

print(
    f"    theta_ddot = "
    f"({sin_theta_coeff:.6f}) sin(theta) "
    f"+ ({damping_coeff:.6f}) theta_dot"
)


# ================================================================
# INTERPRET PHYSICAL PARAMETERS
# ================================================================

g_over_L = -sin_theta_coeff

print()
print("=" * 70)
print("PHYSICAL INTERPRETATION")
print("=" * 70)

print(
    f"Estimated g/L      : {g_over_L:.6f} s^-2"
)

print(
    f"Estimated damping   : {damping_coeff:.6f} s^-1"
)


# ================================================================
# THEORETICAL CHECK FROM MEASURED PERIOD
# ================================================================

if "estimated_period" in data.files:

    T = float(data["estimated_period"])

    if np.isfinite(T) and T > 0:

        omega0 = 2.0 * np.pi / T

        theoretical_g_over_L = omega0 ** 2

        print()
        print("=" * 70)
        print("THEORETICAL PENDULUM CHECK")
        print("=" * 70)

        print(
            f"Measured period    : {T:.4f} s"
        )

        print(
            f"Angular frequency  : "
            f"{omega0:.4f} rad/s"
        )

        print(
            f"Theoretical g/L    : "
            f"{theoretical_g_over_L:.6f} s^-2"
        )

        print(
            f"SINDy estimated g/L: "
            f"{g_over_L:.6f} s^-2"
        )

        error = (
            abs(g_over_L - theoretical_g_over_L)
            / theoretical_g_over_L
            * 100
        )

        print(
            f"Relative difference: "
            f"{error:.2f}%"
        )


# ================================================================
# PREDICT ANGULAR ACCELERATION
# ================================================================

theta_ddot_pred = (
    sin_theta_coeff * np.sin(theta_smooth)
    +
    damping_coeff * theta_dot
)


# ================================================================
# ACCELERATION COMPARISON
# ================================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    t,
    theta_ddot,
    label="Measured θ̈"
)

plt.plot(
    t,
    theta_ddot_pred,
    label="Physics-constrained SINDy θ̈"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Angular acceleration (rad/s²)"
)

plt.title(
    "Horizon — Measured vs Discovered Pendulum Dynamics"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "horizon_sindy_acceleration.png",
    dpi=150
)

plt.show()


# ================================================================
# PHASE SPACE
# ================================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    theta_smooth,
    theta_dot
)

plt.xlabel(
    "θ (rad)"
)

plt.ylabel(
    "θ̇ (rad/s)"
)

plt.title(
    "Horizon — Pendulum Phase Space"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "horizon_phase_space.png",
    dpi=150
)

plt.show()


# ================================================================
# SAVE DISCOVERY
# ================================================================

np.savez(
    "horizon_sindy_results.npz",

    time=t,

    theta=theta_smooth,

    theta_dot=theta_dot,

    theta_ddot=theta_ddot,

    theta_ddot_pred=theta_ddot_pred,

    sin_theta_coefficient=sin_theta_coeff,

    damping_coefficient=damping_coeff,

    estimated_g_over_L=g_over_L,

    pendulum_length_pixels=L_pixels
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("HORIZON — PHYSICS DISCOVERY COMPLETE")
print("=" * 70)

print()
print("FINAL DISCOVERED EQUATION:")
print()
print(
    f"    θ̈ = "
    f"{sin_theta_coeff:.6f} sin(θ) "
    f"+ {damping_coeff:.6f} θ̇"
)

print()
print("Generated:")
print("  horizon_sindy_results.npz")
print("  horizon_sindy_acceleration.png")
print("  horizon_phase_space.png")

print()
print("=" * 70)