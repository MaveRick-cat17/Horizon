from src.dynamics import (
    generate_pendulum_data,
    discover_dynamics
)


print("HORIZON")
print("=" * 40)

print("Generating motion data...")

t, theta, omega = generate_pendulum_data()

print(f"Generated {len(t)} observations.")


print("\nDiscovering governing dynamics...")

model = discover_dynamics(
    t,
    theta,
    omega
)


print("\nDISCOVERED EQUATION")
print("=" * 40)

model.print()