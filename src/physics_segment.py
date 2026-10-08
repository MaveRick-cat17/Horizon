import numpy as np
import matplotlib.pyplot as plt

from tracker import track_ball


# ============================================================
# HORIZON — PHYSICS SEGMENT DETECTOR
# ============================================================

VIDEO_PATH = "data/Videos/Ball.mp4"


# ============================================================
# CONFIGURATION
# ============================================================

# Ignore the beginning of the video.
# This prevents small tracking movements from being
# interpreted as physics events.
MIN_ANALYSIS_TIME = 3.5

# Minimum vertical displacement required for a candidate.
MIN_DISPLACEMENT = 100.0

# Maximum duration allowed for a free-fall segment.
MAX_FREE_FALL_TIME = 0.7

# Minimum duration.
MIN_FREE_FALL_TIME = 0.05


# ============================================================
# LOAD TRACKED DATA
# ============================================================

print()
print("HORIZON — PHYSICS SEGMENT DETECTOR")
print("=" * 60)

print("Tracking video...")

times, positions = track_ball(VIDEO_PATH)

if len(positions) < 20:

    raise ValueError(
        "Not enough tracking data for physics analysis."
    )

print(
    f"Detected {len(positions)} positions."
)


# ============================================================
# EXTRACT POSITION
# ============================================================

x = positions[:, 0]

# OpenCV Y increases downward.
#
# Convert to conventional physics coordinates:
#
#       UP   = positive
#       DOWN = negative

y = -positions[:, 1]


# ============================================================
# CALCULATE VELOCITY
# ============================================================

velocity = np.gradient(
    y,
    times
)


# ============================================================
# LIGHT VELOCITY SMOOTHING
# ============================================================

window = 5

kernel = np.ones(window) / window

velocity_smooth = np.convolve(
    velocity,
    kernel,
    mode="same"
)


# ============================================================
# FIND DOWNWARD MOTION
# ============================================================

# Downward motion means:
#
#       velocity < 0
#
# We deliberately do NOT use acceleration here.

downward = (
    velocity_smooth < 0
)


# ============================================================
# SEARCH FOR FIRST MAJOR DOWNWARD SEGMENT
# ============================================================

candidates = []

n = len(times)

for start in range(n):

    # Ignore early motion.
    if times[start] < MIN_ANALYSIS_TIME:
        continue

    if not downward[start]:
        continue

    start_y = y[start]

    # Search forward for the first local minimum.
    #
    # This should correspond to the first impact.

    end = start + 1

    while end < n:

        elapsed = (
            times[end]
            - times[start]
        )

        if elapsed > MAX_FREE_FALL_TIME:
            break

        # Once velocity becomes positive,
        # the ball has started moving upward again.
        #
        # That means we have reached an impact/bounce.

        if velocity_smooth[end] > 0:

            break

        end += 1


    if end <= start + 1:
        continue


    # --------------------------------------------------------
    # DISPLACEMENT
    # --------------------------------------------------------

    displacement = (
        y[end - 1]
        - start_y
    )

    duration = (
        times[end - 1]
        - times[start]
    )


    # We want substantial DOWNWARD displacement.
    #
    # Since down = negative:
    #
    # displacement must be negative.

    if (
        displacement < -MIN_DISPLACEMENT
        and
        duration >= MIN_FREE_FALL_TIME
    ):

        candidates.append(
            (
                start,
                end - 1,
                displacement,
                duration
            )
        )


# ============================================================
# PRINT CANDIDATES
# ============================================================

print()
print("Downward-motion candidates:")
print("-" * 60)

for i, (
    start,
    end,
    displacement,
    duration
) in enumerate(candidates):

    print(
        f"{i + 1}. "
        f"{times[start]:.3f}s → "
        f"{times[end]:.3f}s | "
        f"Δy = {displacement:.2f} px | "
        f"duration = {duration:.3f}s"
    )


# ============================================================
# SELECT FIRST MAJOR EVENT
# ============================================================

release_index = None
impact_index = None


if candidates:

    # Sort chronologically.
    candidates.sort(
        key=lambda item: item[0]
    )

    # Select the FIRST major downward event.
    #
    # This is important:
    #
    # We do NOT select the largest event.
    #
    # We want the original release, not a later bounce.

    best = candidates[0]

    release_index = best[0]
    impact_index = best[1]


# ============================================================
# VALIDATION
# ============================================================

if (
    release_index is not None
    and
    impact_index is not None
):

    displacement = (
        y[impact_index]
        - y[release_index]
    )

    duration = (
        times[impact_index]
        - times[release_index]
    )

    print()
    print("=" * 60)

    print(
        "✅ FIRST FREE-FALL SEGMENT FOUND"
    )

    print()

    print(
        f"Release time : "
        f"{times[release_index]:.3f} s"
    )

    print(
        f"Impact time  : "
        f"{times[impact_index]:.3f} s"
    )

    print(
        f"Duration     : "
        f"{duration:.3f} s"
    )

    print(
        f"Frames       : "
        f"{impact_index - release_index + 1}"
    )

    print(
        f"Displacement : "
        f"{displacement:.2f} pixels"
    )

    print(
        f"Initial velocity : "
        f"{velocity_smooth[release_index]:.2f} pixels/s"
    )

    print(
        f"Final velocity   : "
        f"{velocity_smooth[impact_index]:.2f} pixels/s"
    )

else:

    print()
    print("=" * 60)

    print(
        "❌ FREE-FALL SEGMENT NOT FOUND"
    )


# ============================================================
# SAVE CLEAN PHYSICS SEGMENT
# ============================================================

if (
    release_index is not None
    and
    impact_index is not None
):

    segment_times = (
        times[
            release_index:
            impact_index + 1
        ]
        - times[release_index]
    )

    segment_x = x[
        release_index:
        impact_index + 1
    ]

    segment_y = y[
        release_index:
        impact_index + 1
    ]

    segment_velocity = velocity_smooth[
        release_index:
        impact_index + 1
    ]

    # Calculate acceleration ONLY on the selected segment.
    #
    # This is much better than differentiating the entire
    # noisy video first.

    segment_acceleration = np.gradient(
        segment_velocity,
        segment_times
    )


    np.savez(
        "horizon_free_fall.npz",

        time=segment_times,

        x=segment_x,

        y=segment_y,

        velocity=segment_velocity,

        acceleration=segment_acceleration
    )

    print()
    print(
        "Saved: horizon_free_fall.npz"
    )


# ============================================================
# PLOT 1 — POSITION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    times,
    y,
    label="Tracked position"
)

if release_index is not None:

    plt.axvspan(
        times[release_index],
        times[impact_index],
        alpha=0.25,
        label="Detected free fall"
    )

    plt.scatter(
        times[release_index],
        y[release_index],
        s=70,
        label="Release"
    )

    plt.scatter(
        times[impact_index],
        y[impact_index],
        s=70,
        label="First impact"
    )


plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Vertical position (pixels)"
)

plt.title(
    "Horizon — Automatic Free-Fall Detection"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2 — VELOCITY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    times,
    velocity_smooth,
    label="Smoothed velocity"
)

if release_index is not None:

    plt.axvspan(
        times[release_index],
        times[impact_index],
        alpha=0.25,
        label="Detected free fall"
    )

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Vertical velocity (pixels/s)"
)

plt.title(
    "Horizon — Vertical Velocity"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3 — ACCELERATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

acceleration_full = np.gradient(
    velocity_smooth,
    times
)

plt.plot(
    times,
    acceleration_full,
    label="Vertical acceleration"
)

if release_index is not None:

    plt.axvspan(
        times[release_index],
        times[impact_index],
        alpha=0.25,
        label="Detected free fall"
    )

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Acceleration (pixels/s²)"
)

plt.title(
    "Horizon — Vertical Acceleration"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 60)

if release_index is not None:

    print(
        "HORIZON PHYSICS SEGMENTATION COMPLETE"
    )

    print()
    print(
        "Next stage:"
    )

    print(
        "horizon_free_fall.npz → SINDy"
    )

else:

    print(
        "HORIZON COULD NOT IDENTIFY THE FREE-FALL EVENT."
    )