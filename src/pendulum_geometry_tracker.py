import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON — GEOMETRY-AWARE PENDULUM TRACKER
#
# Pivot + bob geometry + gold colour detection
#
# The tracker:
#   1. Finds gold candidates
#   2. Uses the manually selected pivot
#   3. Estimates pendulum length in pixels
#   4. Predicts the next angular position
#   5. Rejects candidates that violate the geometry
#   6. Processes the ENTIRE video
#
# ============================================================


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_DATA = "horizon_pendulum_geometry.npz"
OUTPUT_VIDEO = "horizon_pendulum_geometry_tracking.mp4"


# ============================================================
# GOLD COLOUR RANGE
# ============================================================

LOWER_GOLD = np.array([
    5,
    60,
    40
])

UPPER_GOLD = np.array([
    45,
    255,
    255
])


# ============================================================
# BOB DETECTION
# ============================================================

MIN_AREA = 15
MAX_AREA = 3000

MIN_RADIUS = 3
MAX_RADIUS = 60

MIN_CIRCULARITY = 0.20


# ============================================================
# GEOMETRIC TOLERANCES
# ============================================================

# Allowed percentage variation in pendulum length.
# Example: 0.18 = ±18%.
RADIUS_TOLERANCE = 0.18

# Maximum angular change per frame.
# This is deliberately generous.
MAX_ANGLE_STEP_DEG = 8.0

# Maximum distance between predicted and detected bob.
MAX_PREDICTION_ERROR = 100


# ============================================================
# FRAME BROWSER
# ============================================================

def choose_start_frame(cap, total_frames, fps):

    frame_number = 0

    print("\n" + "=" * 70)
    print("SELECT A GOOD STARTING FRAME")
    print("=" * 70)

    print("""
Controls:

  D / RIGHT ARROW : next frame
  A / LEFT ARROW  : previous frame
  ENTER           : select frame
  ESC             : cancel

Choose a frame where:

  ✓ The bob is clearly visible
  ✓ The bob is not hidden
  ✓ The complete pendulum is visible
  ✓ The pivot is visible
""")

    while True:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:
            return None

        display = frame.copy()

        cv2.putText(
            display,
            f"Frame: {frame_number}/{total_frames - 1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Time: {frame_number / fps:.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "D/Right next | A/Left previous | ENTER select",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Horizon - Frame Browser",
            display
        )

        key = cv2.waitKey(0) & 0xFF

        if key in [13, 10]:

            cv2.destroyWindow(
                "Horizon - Frame Browser"
            )

            return frame_number

        elif key == 27:

            cv2.destroyWindow(
                "Horizon - Frame Browser"
            )

            return None

        elif key in [
            ord("d"),
            ord("D"),
            83
        ]:

            frame_number += 1

            if frame_number >= total_frames:
                frame_number = total_frames - 1

        elif key in [
            ord("a"),
            ord("A"),
            81
        ]:

            frame_number -= 1

            if frame_number < 0:
                frame_number = 0


# ============================================================
# SELECT POINT USING SMALL ROI
# ============================================================

def select_point(frame, window_name, instructions):

    print("\n" + "=" * 70)
    print(instructions)
    print("=" * 70)

    roi = cv2.selectROI(
        window_name,
        frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(
        window_name
    )

    x, y, w, h = roi

    if w <= 1 or h <= 1:
        return None

    center = (
        int(x + w / 2),
        int(y + h / 2)
    )

    return center


# ============================================================
# GOLD CANDIDATE DETECTION
# ============================================================

def find_gold_candidates(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        LOWER_GOLD,
        UPPER_GOLD
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        circularity = (
            4.0
            * np.pi
            * area
            / (perimeter * perimeter)
        )

        if circularity < MIN_CIRCULARITY:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(
            contour
        )

        if radius < MIN_RADIUS:
            continue

        if radius > MAX_RADIUS:
            continue

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        center = (
            int(
                moments["m10"]
                / moments["m00"]
            ),
            int(
                moments["m01"]
                / moments["m00"]
            )
        )

        candidates.append({
            "center": center,
            "radius": radius,
            "area": area,
            "circularity": circularity
        })

    return candidates, mask


# ============================================================
# ANGLE CALCULATION
#
# Angle is measured from the downward vertical.
#
# theta = 0:
#
#          PIVOT
#            |
#            |
#            ●
#
# Positive theta = bob to the right
# Negative theta = bob to the left
# ============================================================

def calculate_theta(
    pivot,
    bob
):

    px, py = pivot
    bx, by = bob

    dx = bx - px
    dy = by - py

    theta = np.arctan2(
        dx,
        dy
    )

    return theta


# ============================================================
# PREDICT BOB FROM ANGLE
# ============================================================

def angle_to_position(
    pivot,
    radius,
    theta
):

    px, py = pivot

    x = (
        px
        + radius * np.sin(theta)
    )

    y = (
        py
        + radius * np.cos(theta)
    )

    return (
        int(round(x)),
        int(round(y))
    )


# ============================================================
# SELECT BEST GEOMETRIC CANDIDATE
# ============================================================

def choose_candidate(
    candidates,
    pivot,
    pendulum_radius,
    previous_theta,
    angular_velocity,
    dt
):

    if len(candidates) == 0:
        return None

    # --------------------------------------------------------
    # Predicted angle
    # --------------------------------------------------------

    predicted_theta = (
        previous_theta
        + angular_velocity * dt
    )

    best_candidate = None
    best_score = float("inf")

    for candidate in candidates:

        cx, cy = candidate["center"]

        # ----------------------------------------------------
        # Radius from pivot
        # ----------------------------------------------------

        dx = cx - pivot[0]
        dy = cy - pivot[1]

        radius = np.sqrt(
            dx * dx + dy * dy
        )

        radius_error = abs(
            radius - pendulum_radius
        ) / pendulum_radius

        # Reject candidates far from pendulum length.
        if radius_error > RADIUS_TOLERANCE:
            continue

        # ----------------------------------------------------
        # Angle
        # ----------------------------------------------------

        theta = calculate_theta(
            pivot,
            candidate["center"]
        )

        # Circular angular difference
        angle_difference = np.arctan2(
            np.sin(
                theta - predicted_theta
            ),
            np.cos(
                theta - predicted_theta
            )
        )

        angle_difference_deg = abs(
            np.degrees(
                angle_difference
            )
        )

        if (
            angle_difference_deg
            > MAX_ANGLE_STEP_DEG
        ):
            continue

        # ----------------------------------------------------
        # Predicted position
        # ----------------------------------------------------

        predicted_position = angle_to_position(
            pivot,
            pendulum_radius,
            predicted_theta
        )

        prediction_error = np.linalg.norm(
            np.array(
                candidate["center"],
                dtype=float
            )
            -
            np.array(
                predicted_position,
                dtype=float
            )
        )

        if prediction_error > MAX_PREDICTION_ERROR:
            continue

        # ----------------------------------------------------
        # Candidate score
        # ----------------------------------------------------
        #
        # Lower = better.
        #
        # Geometry is more important than raw colour area.

        score = (
            3.0 * radius_error
            +
            0.05 * angle_difference_deg
            +
            0.01 * prediction_error
            -
            0.20 * candidate["circularity"]
        )

        if score < best_score:

            best_score = score
            best_candidate = candidate

    return best_candidate


# ============================================================
# DRAW TRACKING
# ============================================================

def draw_overlay(
    frame,
    pivot,
    bob,
    pendulum_radius,
    trajectory,
    frame_number,
    fps,
    status,
    theta=None
):

    output = frame.copy()

    # --------------------------------------------------------
    # Draw trajectory
    # --------------------------------------------------------

    for i in range(
        1,
        len(trajectory)
    ):

        p1 = trajectory[i - 1]
        p2 = trajectory[i]

        if p1 is None or p2 is None:
            continue

        cv2.line(
            output,
            p1,
            p2,
            (255, 0, 0),
            2
        )

    # --------------------------------------------------------
    # Pivot
    # --------------------------------------------------------

    cv2.circle(
        output,
        pivot,
        6,
        (255, 0, 255),
        -1
    )

    # --------------------------------------------------------
    # Bob
    # --------------------------------------------------------

    if bob is not None:

        cv2.circle(
            output,
            bob,
            7,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            output,
            bob,
            14,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Draw pivot-to-bob line
        # ----------------------------------------------------

        cv2.line(
            output,
            pivot,
            bob,
            (0, 255, 255),
            2
        )

    # --------------------------------------------------------
    # Predicted pendulum circle
    # --------------------------------------------------------

    cv2.circle(
        output,
        pivot,
        int(pendulum_radius),
        (100, 100, 100),
        1
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"Time: {frame_number / fps:.2f}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Frame: {frame_number}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Status: {status}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    if theta is not None:

        cv2.putText(
            output,
            f"Theta: {np.degrees(theta):.2f} deg",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("HORIZON — GEOMETRY-AWARE PENDULUM TRACKER")
    print("=" * 70)

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print()
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")
    print(
        f"Resolution : "
        f"{width} × {height}"
    )

    print(
        f"Duration   : "
        f"{total_frames / fps:.2f}s"
    )

    # --------------------------------------------------------
    # Select frame
    # --------------------------------------------------------

    start_frame = choose_start_frame(
        cap,
        total_frames,
        fps
    )

    if start_frame is None:

        cap.release()

        return

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    ret, first_frame = cap.read()

    if not ret:

        cap.release()

        raise RuntimeError(
            "Could not read starting frame."
        )

    # --------------------------------------------------------
    # Select pivot
    # --------------------------------------------------------

    pivot = select_point(
        first_frame,
        "Select Pendulum Pivot",
        """
SELECT THE PENDULUM PIVOT

Draw a small box around the exact point
where the string is attached.

The centre of your selection will be
used as the pivot coordinate.
"""
    )

    if pivot is None:

        cap.release()

        print("Pivot selection cancelled.")

        return

    print(
        f"\nSelected pivot: "
        f"x={pivot[0]}, y={pivot[1]}"
    )

    # --------------------------------------------------------
    # Select bob
    # --------------------------------------------------------

    bob_initial = select_point(
        first_frame,
        "Select Pendulum Bob",
        """
SELECT THE PENDULUM BOB

Draw a tight box around the gold/brass
spherical bob.

Do NOT include the base.
Do NOT include the string.
"""
    )

    if bob_initial is None:

        cap.release()

        print("Bob selection cancelled.")

        return

    print(
        f"Selected bob: "
        f"x={bob_initial[0]}, "
        f"y={bob_initial[1]}"
    )

    # --------------------------------------------------------
    # Initial pendulum radius
    # --------------------------------------------------------

    pendulum_radius = np.linalg.norm(
        np.array(bob_initial, dtype=float)
        -
        np.array(pivot, dtype=float)
    )

    print(
        f"\nEstimated pendulum length: "
        f"{pendulum_radius:.2f} pixels"
    )

    # --------------------------------------------------------
    # Initial angle
    # --------------------------------------------------------

    theta = calculate_theta(
        pivot,
        bob_initial
    )

    angular_velocity = 0.0

    # --------------------------------------------------------
    # Output video
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    positions = np.full(
        (total_frames, 2),
        np.nan,
        dtype=np.float64
    )

    theta_array = np.full(
        total_frames,
        np.nan,
        dtype=np.float64
    )

    times = np.arange(
        total_frames,
        dtype=np.float64
    ) / fps

    trajectory = [
        None
    ] * total_frames

    # --------------------------------------------------------
    # Store initial point
    # --------------------------------------------------------

    positions[start_frame] = bob_initial

    theta_array[start_frame] = theta

    trajectory[start_frame] = bob_initial

    detected_count = 1

    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROCESSING ENTIRE VIDEO")
    print("=" * 70)

    print("""
MAGENTA = pivot
RED     = bob centre
GREEN   = bob boundary
YELLOW  = pivot-to-bob line
BLUE    = tracked trajectory
GRAY    = expected pendulum radius

Press Q to stop.
""")

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame + 1
    )

    for frame_number in range(
        start_frame + 1,
        total_frames
    ):

        ret, frame = cap.read()

        if not ret:
            break

        dt = 1.0 / fps

        # ----------------------------------------------------
        # Detect gold candidates
        # ----------------------------------------------------

        candidates, mask = (
            find_gold_candidates(frame)
        )

        # ----------------------------------------------------
        # Choose geometrically valid candidate
        # ----------------------------------------------------

        candidate = choose_candidate(
            candidates,
            pivot,
            pendulum_radius,
            theta,
            angular_velocity,
            dt
        )

        if candidate is not None:

            bob = candidate["center"]

            new_theta = calculate_theta(
                pivot,
                bob
            )

            # Angular velocity
            raw_velocity = (
                new_theta - theta
            ) / dt

            # Correct angle wrapping
            raw_velocity = np.arctan2(
                np.sin(
                    new_theta - theta
                ),
                np.cos(
                    new_theta - theta
                )
            ) / dt

            # Smooth angular velocity
            angular_velocity = (
                0.7 * angular_velocity
                +
                0.3 * raw_velocity
            )

            theta = new_theta

            positions[frame_number] = bob

            theta_array[frame_number] = theta

            trajectory[frame_number] = bob

            detected_count += 1

            status = "GEOMETRICALLY VALID"

        else:

            bob = None

            status = "NO VALID BOB"

            # IMPORTANT:
            #
            # We do NOT change theta or angular velocity.
            #
            # This prevents a false detection from corrupting
            # the future prediction.

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        annotated = draw_overlay(
            frame,
            pivot,
            bob,
            pendulum_radius,
            trajectory,
            frame_number,
            fps,
            status,
            theta
        )

        writer.write(
            annotated
        )

        cv2.imshow(
            "Horizon - Geometry Tracker",
            annotated
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print(
                "\nStopped manually."
            )

            break

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_number % 30 == 0:

            percentage = (
                frame_number
                / total_frames
                * 100
            )

            print(
                f"Frame "
                f"{frame_number:4d}/"
                f"{total_frames} "
                f"({percentage:5.1f}%) | "
                f"{status}"
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Valid data
    # --------------------------------------------------------

    valid_mask = ~np.isnan(
        positions[:, 0]
    )

    valid_positions = positions[
        valid_mask
    ]

    valid_times = times[
        valid_mask
    ]

    valid_theta = theta_array[
        valid_mask
    ]

    detection_rate = (
        detected_count
        / (total_frames - start_frame)
        * 100
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez(
        OUTPUT_DATA,
        time=valid_times,
        positions=valid_positions,
        theta=valid_theta,
        pivot=np.array(pivot),
        pendulum_radius=pendulum_radius,
        fps=fps,
        width=width,
        height=height
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GEOMETRY TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Pivot             : "
        f"{pivot}"
    )

    print(
        f"Pendulum length   : "
        f"{pendulum_radius:.2f}px"
    )

    print(
        f"Processed frames  : "
        f"{total_frames - start_frame}"
    )

    print(
        f"Valid detections  : "
        f"{detected_count}"
    )

    print(
        f"Missing detections: "
        f"{(total_frames - start_frame) - detected_count}"
    )

    print(
        f"Detection rate    : "
        f"{detection_rate:.1f}%"
    )

    print()
    print(
        f"Saved data        : "
        f"{OUTPUT_DATA}"
    )

    print(
        f"Saved video       : "
        f"{OUTPUT_VIDEO}"
    )

    # ========================================================
    # PLOT 1 — X/Y TRAJECTORY
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    plt.plot(
        valid_positions[:, 0],
        valid_positions[:, 1],
        linewidth=2,
        label="Tracked bob"
    )

    plt.scatter(
        [pivot[0]],
        [pivot[1]],
        s=80,
        label="Pivot"
    )

    plt.xlabel(
        "X position (pixels)"
    )

    plt.ylabel(
        "Y position (pixels)"
    )

    plt.title(
        "Horizon — Geometry-Validated Pendulum Trajectory"
    )

    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.show()

    # ========================================================
    # PLOT 2 — ANGLE
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        valid_times,
        np.degrees(valid_theta),
        linewidth=2
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Angle θ (degrees)"
    )

    plt.title(
        "Horizon — Pendulum Angle θ(t)"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 70)
    print("HORIZON GEOMETRY TRACKING COMPLETE")
    print("=" * 70)

    print("""
NEXT PHYSICS STAGE:

    x(t), y(t)
         ↓
       θ(t)
         ↓
    smoothing
         ↓
      θ̇(t)
         ↓
      θ̈(t)
         ↓
      SINDy
         ↓
    discovered equation
""")


if __name__ == "__main__":
    main()