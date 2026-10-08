import numpy as np
import matplotlib.pyplot as plt

from src.tracker import track_ball


VIDEO_PATH = "data/Videos/Ball.mp4"


print("HORIZON — PHYSICS SEGMENT TEST")
print("=" * 50)

times, positions = track_ball(VIDEO_PATH)

print(f"Total detected positions: {len(positions)}")

if len(positions) == 0:
    print("No object detected.")
    exit()


x = positions[:, 0]
y = positions[:, 1]


# --------------------------------------------------
# FIND THE LARGEST VERTICAL MOTION
# --------------------------------------------------

dy = np.diff(y)

largest_motion_index = np.argmax(
    np.abs(dy)
)

print()
print("Largest vertical movement:")
print(
    f"Index: {largest_motion_index}"
)

print(
    f"Frame/time approximately: "
    f"{times[largest_motion_index]:.2f} s"
)


# --------------------------------------------------
# PLOT Y POSITION VS TIME
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    times,
    -y,
    marker=".",
    markersize=3
)

plt.xlabel("Time (seconds)")
plt.ylabel("Vertical position (pixels)")

plt.title(
    "Horizon — Vertical Motion vs Time"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# --------------------------------------------------
# PLOT X POSITION VS TIME
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    times,
    x,
    marker=".",
    markersize=3
)

plt.xlabel("Time (seconds)")
plt.ylabel("Horizontal position (pixels)")

plt.title(
    "Horizon — Horizontal Motion vs Time"
)

plt.grid(True)

plt.tight_layout()

plt.show()