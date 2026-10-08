import numpy as np
import matplotlib.pyplot as plt

from src.dynamics import (
    generate_pendulum_data,
    discover_dynamics
)

from src.preprocessing import (
    add_measurement_noise,
    smooth_trajectory
)


print("HORIZON — NOISY TRAJECTORY PREDICTION")
print("=" * 55)


# -------------------------------------------------
# 1. Generate clean physical data
# -------------------------------------------------

t, theta, omega = generate_pendulum_data()

X = np.column_stack([theta, omega])

print(f"Total observations: {len(t)}")


# -------------------------------------------------
# 2. Simulate imperfect measurements
# -------------------------------------------------

np.random.seed(42)

noisy_theta = add_measurement_noise(
    theta,
    noise_level=0.01
)

noisy_omega = add_measurement_noise(
    omega,
    noise_level=0.01
)

print("Artificial measurement noise added.")


# -------------------------------------------------
# 3. Smooth the noisy observations
# -------------------------------------------------

filtered_theta = smooth_trajectory(
    noisy_theta
)

filtered_omega = smooth_trajectory(
    noisy_omega
)

print("Trajectory noise filtered.")


# -------------------------------------------------
# 4. Hide the future
# -------------------------------------------------

train_ratio = 0.60

split_index = int(len(t) * train_ratio)

t_train = t[:split_index]

theta_train = filtered_theta[:split_index]
omega_train = filtered_omega[:split_index]

X_test = X[split_index:]
t_test = t[split_index:]

print(f"Observations given to Horizon: {split_index}")
print(f"Future observations hidden:    {len(t_test)}")


# -------------------------------------------------
# 5. Discover dynamics
# -------------------------------------------------

print("\nDiscovering dynamics from noisy observations...")

model = discover_dynamics(
    t_train,
    theta_train,
    omega_train
)

print("\nDISCOVERED MODEL")
print("=" * 55)

model.print()


# -------------------------------------------------
# 6. Predict the hidden future
# -------------------------------------------------

print("\nPredicting hidden future...")

initial_state = np.array([
    theta_train[-1],
    omega_train[-1]
])

predicted = model.simulate(
    initial_state,
    t_test
)


# -------------------------------------------------
# 7. Calculate prediction error
# -------------------------------------------------

actual_theta = X_test[:, 0]

predicted_theta = predicted[:, 0]

rmse = np.sqrt(
    np.mean(
        (actual_theta - predicted_theta) ** 2
    )
)

print("\nPREDICTION RESULTS")
print("=" * 55)

print(f"Prediction RMSE: {rmse:.6f} radians")


# -------------------------------------------------
# 8. Visualisation
# -------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    t_train,
    theta_train,
    label="Noisy + filtered observations"
)

plt.plot(
    t_test,
    actual_theta,
    label="Actual future"
)

plt.plot(
    t_test,
    predicted_theta,
    "--",
    label="Horizon prediction"
)

plt.axvline(
    t_train[-1],
    linestyle=":"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Angle (radians)")

plt.title(
    "Horizon — Prediction from Noisy Observations"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "horizon_noisy_prediction.png",
    dpi=150
)

plt.show()