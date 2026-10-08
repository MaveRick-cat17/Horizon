import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from tracker import track_ball


# ============================================================
# HORIZON — DYNAMICS DATA PREPARATION
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "data/Videos/Ball.mp4"


# ------------------------------------------------------------
# POSITION SMOOTHING
# ------------------------------------------------------------

SMOOTHING_WINDOW = 7
SMOOTHING_POLYORDER = 2


# ------------------------------------------------------------
# FREE-FALL DETECTION
# ------------------------------------------------------------

# Coordinate system:
#
#       UP   = positive
#       DOWN = negative
#
# Therefore sufficiently fast downward motion has:
#
#       velocity < FREE_FALL_VELOCITY
#

FREE_FALL_VELOCITY = -500.0


# Minimum number of consecutive points required
# for a candidate.

MIN_FREE_FALL_POINTS = 3


# Minimum downward displacement required.

MIN_DISPLACEMENT = 100.0


# ------------------------------------------------------------
# TRACKING JUMP DETECTION
# ------------------------------------------------------------

JUMP_THRESHOLD = 120.0


# ------------------------------------------------------------
# IMPACT DETECTION
# ------------------------------------------------------------

# After release, the ball moves downward.
#
# At impact it reaches its lowest point and begins moving
# upward again.
#
# We therefore search for the first velocity sign change
# after the detected release.

IMPACT_SEARCH_LIMIT = 1.0


# Number of points to REMOVE around the actual impact.
#
# IMPORTANT:
#
# We do NOT give the impact point itself to SINDy because
# collision is a discontinuity and does not represent the
# continuous free-flight dynamics.
#
# 1 means:
#
#       Release ........ last pre-impact point | IMPACT
#
# The impact point and everything after it are excluded.

REMOVE_IMPACT_POINTS = 1


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("HORIZON — DYNAMICS DATA PREPARATION")
print("=" * 70)

print()
print("Tracking video...")
print()


# ============================================================
# TRACK VIDEO
# ============================================================

times, positions = track_ball(VIDEO_PATH)


if len(positions) < 10:

    raise ValueError(
        "Not enough tracking data."
    )


print(
    f"Tracked positions: {len(positions)}"
)


# ============================================================
# EXTRACT COORDINATES
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
# BASIC TIME CHECK
# ============================================================

dt = np.diff(times)

print(
    f"Time range: {times[0]:.3f}s → {times[-1]:.3f}s"
)

print(
    f"Median Δt: {np.median(dt):.5f}s"
)


# ============================================================
# TRACKING ANOMALY CHECK
# ============================================================

dx = np.diff(x)
dy = np.diff(y)

distance = np.sqrt(
    dx ** 2 + dy ** 2
)


bad_indices = (
    np.where(
        distance > JUMP_THRESHOLD
    )[0]
    + 1
)


print(
    f"Potential tracking anomalies: "
    f"{len(bad_indices)}"
)


# ============================================================
# POSITION SMOOTHING
# ============================================================

window = min(
    SMOOTHING_WINDOW,
    len(y)
)


# Window must be odd.

if window % 2 == 0:

    window -= 1


if window <= SMOOTHING_POLYORDER:

    y_smooth = y.copy()
    x_smooth = x.copy()

else:

    y_smooth = savgol_filter(
        y,
        window_length=window,
        polyorder=SMOOTHING_POLYORDER
    )

    x_smooth = savgol_filter(
        x,
        window_length=window,
        polyorder=SMOOTHING_POLYORDER
    )


# ============================================================
# VELOCITY
# ============================================================

velocity = np.gradient(
    y_smooth,
    times
)


print()
print(
    f"Maximum downward velocity: "
    f"{np.min(velocity):.2f} pixels/s"
)

print(
    f"Maximum upward velocity: "
    f"{np.max(velocity):.2f} pixels/s"
)


# ============================================================
# STEP 1 — FIND RAPID DOWNWARD MOTION
# ============================================================

rapid_downward = (
    velocity < FREE_FALL_VELOCITY
)


# ============================================================
# STEP 2 — FIND CONTINUOUS CANDIDATES
# ============================================================

candidates = []

start = None


for i, active in enumerate(
    rapid_downward
):

    if active and start is None:

        start = i

    elif not active and start is not None:

        end = i - 1

        point_count = (
            end - start + 1
        )

        displacement = (
            y_smooth[end]
            - y_smooth[start]
        )

        if (
            point_count
            >= MIN_FREE_FALL_POINTS
            and displacement
            <= -MIN_DISPLACEMENT
        ):

            candidates.append(
                (
                    start,
                    end
                )
            )

        start = None


# Handle candidate reaching the end.

if start is not None:

    end = (
        len(rapid_downward)
        - 1
    )

    point_count = (
        end - start + 1
    )

    displacement = (
        y_smooth[end]
        - y_smooth[start]
    )

    if (
        point_count
        >= MIN_FREE_FALL_POINTS
        and displacement
        <= -MIN_DISPLACEMENT
    ):

        candidates.append(
            (
                start,
                end
            )
        )


# ============================================================
# PRINT CANDIDATES
# ============================================================

print()
print(
    "Rapid downward-motion candidates:"
)

print("-" * 70)


if not candidates:

    print(
        "No rapid downward-motion candidates found."
    )

else:

    for i, (
        candidate_start,
        candidate_end
    ) in enumerate(candidates):

        displacement = (
            y_smooth[candidate_end]
            - y_smooth[candidate_start]
        )

        duration = (
            times[candidate_end]
            - times[candidate_start]
        )

        print(
            f"{i + 1}. "
            f"{times[candidate_start]:.3f}s → "
            f"{times[candidate_end]:.3f}s | "
            f"{candidate_end - candidate_start + 1} points | "
            f"Δy = {displacement:.2f} px | "
            f"duration = {duration:.3f}s"
        )


# ============================================================
# STEP 3 — SELECT MOST LIKELY RELEASE
# ============================================================

best_candidate = None


if candidates:

    # The real release should produce the largest
    # sustained downward displacement.

    best_candidate = max(
        candidates,
        key=lambda segment:
        abs(
            y_smooth[segment[1]]
            - y_smooth[segment[0]]
        )
    )


# ============================================================
# STEP 4 — FIND IMPACT
# ============================================================

release_index = None
impact_index = None


if best_candidate is not None:

    release_index = (
        best_candidate[0]
    )

    candidate_end = (
        best_candidate[1]
    )

    release_time = (
        times[release_index]
    )


    # --------------------------------------------------------
    # Search forward for impact.
    # --------------------------------------------------------
    #
    # The ball is falling:
    #
    #       velocity < 0
    #
    # At the bottom of the trajectory it begins moving upward:
    #
    #       velocity > 0
    #
    # We therefore find the first local minimum after release.
    #

    max_search_time = (
        release_time
        + IMPACT_SEARCH_LIMIT
    )


    search_end = candidate_end

    for i in range(
        release_index + 1,
        len(times)
    ):

        if times[i] > max_search_time:

            break

        search_end = i


    # Find the minimum vertical position.
    #
    # Remember:
    #
    #       smaller y = lower position
    #
    # Therefore the minimum is the impact region.

    if search_end > release_index:

        local_slice = y_smooth[
            release_index:
            search_end + 1
        ]

        local_min_index = np.argmin(
            local_slice
        )

        impact_index = (
            release_index
            + local_min_index
        )


# ============================================================
# VALIDATE IMPACT
# ============================================================

if (
    release_index is not None
    and impact_index is not None
):

    displacement = (
        y_smooth[impact_index]
        - y_smooth[release_index]
    )

    duration = (
        times[impact_index]
        - times[release_index]
    )

    if displacement > -MIN_DISPLACEMENT:

        print()
        print(
            "❌ Impact candidate failed "
            "displacement validation."
        )

        release_index = None
        impact_index = None


# ============================================================
# PRINT SELECTED EVENT
# ============================================================

print()
print("=" * 70)


if (
    release_index is not None
    and impact_index is not None
):

    print(
        "✅ FREE-FALL EVENT IDENTIFIED"
    )

    print("=" * 70)

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
        f"{times[impact_index] - times[release_index]:.3f} s"
    )

    print(
        f"Total points : "
        f"{impact_index - release_index + 1}"
    )

    print(
        f"Displacement : "
        f"{y_smooth[impact_index] - y_smooth[release_index]:.2f} px"
    )

    print(
        f"Initial velocity : "
        f"{velocity[release_index]:.2f} pixels/s"
    )

    print(
        f"Impact velocity  : "
        f"{velocity[impact_index]:.2f} pixels/s"
    )

else:

    print(
        "❌ FREE-FALL EVENT NOT FOUND"
    )

    print("=" * 70)


# ============================================================
# STEP 5 — EXTRACT ONLY PRE-IMPACT DATA
# ============================================================

if (
    release_index is not None
    and impact_index is not None
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT INCLUDE THE IMPACT POINT.
    #
    # SINDy should see only continuous free-flight motion.
    #
    # So:
    #
    #       release ............ last pre-impact sample
    #                                           |
    #                                           X impact
    #                                           |
    #                                           rebound
    #
    # --------------------------------------------------------

    end_index = (
        impact_index
        - REMOVE_IMPACT_POINTS
    )


    # Make sure at least two points remain.

    if end_index <= release_index:

        raise ValueError(
            "Free-fall segment is too short "
            "after removing impact."
        )


    # Extract trajectory.

    t_free = (
        times[
            release_index:
            end_index + 1
        ]
        - times[release_index]
    )

    x_free = (
        x_smooth[
            release_index:
            end_index + 1
        ]
    )

    y_free = (
        y_smooth[
            release_index:
            end_index + 1
        ]
    )

    velocity_free = (
        velocity[
            release_index:
            end_index + 1
        ]
    )


    # --------------------------------------------------------
    # Recalculate acceleration ONLY inside the
    # pre-impact trajectory.
    #
    # This prevents the impact from contaminating
    # acceleration.
    # --------------------------------------------------------

    if len(t_free) >= 3:

        acceleration_free = np.gradient(
            velocity_free,
            t_free
        )

    else:

        acceleration_free = np.zeros_like(
            velocity_free
        )


    # --------------------------------------------------------
    # Remove duplicate / invalid timestamps.
    # --------------------------------------------------------

    valid_time = np.ones(
        len(t_free),
        dtype=bool
    )


    if len(t_free) > 1:

        valid_time[1:] = (
            np.diff(t_free) > 0
        )


    t_free = t_free[
        valid_time
    ]

    x_free = x_free[
        valid_time
    ]

    y_free = y_free[
        valid_time
    ]

    velocity_free = velocity_free[
        valid_time
    ]

    acceleration_free = (
        acceleration_free[
            valid_time
        ]
    )


    # ========================================================
    # SAVE CLEAN PRE-IMPACT TRAJECTORY
    # ========================================================

    np.savez(
        "horizon_free_fall.npz",

        time=t_free,

        x=x_free,

        y=y_free,

        velocity=velocity_free,

        acceleration=acceleration_free,

        # Useful metadata

        release_time=np.array(
            times[release_index]
        ),

        impact_time=np.array(
            times[impact_index]
        ),

        release_index=np.array(
            release_index
        ),

        impact_index=np.array(
            impact_index
        )
    )


    # ========================================================
    # PRINT SAVED DATA
    # ========================================================

    print()
    print("=" * 70)

    print(
        "PRE-IMPACT DYNAMICS DATA"
    )

    print("=" * 70)

    print(
        f"Release : "
        f"{times[release_index]:.3f} s"
    )

    print(
        f"Impact  : "
        f"{times[impact_index]:.3f} s"
    )

    print(
        f"SINDy end : "
        f"{times[end_index]:.3f} s"
    )

    print(
        f"Saved samples : "
        f"{len(t_free)}"
    )

    print(
        f"Saved duration : "
        f"{t_free[-1]:.3f} s"
    )

    print()
    print(
        "Impact/rebound data EXCLUDED."
    )

    print()
    print(
        "Saved:"
    )

    print(
        "horizon_free_fall.npz"
    )

    print()
    print(
        "Next stage:"
    )

    print(
        "horizon_free_fall.npz → SINDy"
    )


else:

    print()
    print(
        "No dynamics file was created."
    )


# ============================================================
# VISUALISATION
# ============================================================

plt.figure(
    figsize=(11, 6)
)


# ------------------------------------------------------------
# Raw trajectory
# ------------------------------------------------------------

plt.plot(
    times,
    y,
    alpha=0.30,
    label="Raw tracked position"
)


# ------------------------------------------------------------
# Smoothed trajectory
# ------------------------------------------------------------

plt.plot(
    times,
    y_smooth,
    linewidth=2,
    label="Smoothed position"
)


# ------------------------------------------------------------
# SHOW CANDIDATES
# ------------------------------------------------------------

for i, (
    candidate_start,
    candidate_end
) in enumerate(candidates):

    plt.axvspan(
        times[candidate_start],
        times[candidate_end],
        alpha=0.10,
        label=(
            "Candidate"
            if i == 0
            else None
        )
    )


# ------------------------------------------------------------
# SHOW SELECTED PRE-IMPACT FREE FALL
# ------------------------------------------------------------

if (
    release_index is not None
    and impact_index is not None
):

    # SINDy region

    plt.axvspan(
        times[release_index],
        times[end_index],
        alpha=0.30,
        label="SINDy trajectory"
    )


    # Release marker

    plt.scatter(
        times[release_index],
        y_smooth[release_index],
        s=80,
        label="Release",
        zorder=5
    )


    # Impact marker

    plt.scatter(
        times[impact_index],
        y_smooth[impact_index],
        s=80,
        label="Impact",
        zorder=5
    )


    # Vertical impact line

    plt.axvline(
        times[impact_index],
        linestyle="--",
        alpha=0.7
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
# VELOCITY GRAPH
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    times,
    velocity,
    label="Vertical velocity"
)


plt.axhline(
    FREE_FALL_VELOCITY,
    linestyle="--",
    label="Free-fall velocity threshold"
)


if (
    release_index is not None
    and impact_index is not None
):

    # Only highlight the actual SINDy region.

    plt.axvspan(
        times[release_index],
        times[end_index],
        alpha=0.25,
        label="SINDy trajectory"
    )


    plt.axvline(
        times[release_index],
        linestyle="--",
        label="Release"
    )


    plt.axvline(
        times[impact_index],
        linestyle="--",
        alpha=0.7,
        label="Impact"
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
# ACCELERATION GRAPH
# ============================================================

acceleration = np.gradient(
    velocity,
    times
)


plt.figure(
    figsize=(11, 6)
)

plt.plot(
    times,
    acceleration,
    label="Vertical acceleration"
)


if (
    release_index is not None
    and impact_index is not None
):

    plt.axvspan(
        times[release_index],
        times[end_index],
        alpha=0.25,
        label="SINDy trajectory"
    )


    # Mark impact because acceleration after this
    # point is contaminated by the collision.

    plt.axvline(
        times[impact_index],
        linestyle="--",
        alpha=0.7,
        label="Impact"
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
print("=" * 70)

if (
    release_index is not None
    and impact_index is not None
):

    print(
        "HORIZON PHYSICS SEGMENTATION COMPLETE"
    )

    print()

    print(
        "The saved trajectory contains ONLY "
        "pre-impact free-flight data."
    )

    print()

    print(
        "Impact and rebound have been excluded "
        "from the SINDy dataset."
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
        "HORIZON COULD NOT IDENTIFY "
        "A FREE-FALL SEGMENT."
    )

print("=" * 70)