import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON — PENDULUM GEOMETRY TRACKER V2
# Fixed pivot + gold bob + circular geometry
# ============================================================


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_DATA = "horizon_pendulum_geometry.npz"
OUTPUT_VIDEO = "horizon_pendulum_geometry_tracking.mp4"


# ============================================================
# IMPORTANT: FIXED PIVOT
# ============================================================
#
# Based on the video setup you showed me, the actual pivot is
# near the top of the image.
#
# Image size = 480 x 690
#
# If necessary, we can adjust these two numbers later.
#

PIVOT_X = 250
PIVOT_Y = 45

PIVOT = (
    PIVOT_X,
    PIVOT_Y
)


# ============================================================
# GOLD / BRASS DETECTION
# ============================================================

LOWER_GOLD = np.array([
    5,
    45,
    35
])

UPPER_GOLD = np.array([
    45,
    255,
    255
])


# ============================================================
# BOB SIZE
# ============================================================

MIN_AREA = 10
MAX_AREA = 2500

MIN_RADIUS = 2
MAX_RADIUS = 45

MIN_CIRCULARITY = 0.15


# ============================================================
# GEOMETRY TOLERANCES
# ============================================================

# Pendulum length variation allowed.
RADIUS_TOLERANCE = 0.20

# Maximum angular change between consecutive frames.
#
# At 30 FPS this is deliberately generous.
MAX_ANGLE_STEP_DEG = 12.0

# Maximum jump from previous bob position.
MAX_POSITION_JUMP = 100


# ============================================================
# FRAME BROWSER
# ============================================================

def choose_start_frame(
    cap,
    total_frames,
    fps
):

    frame_number = 0

    print("\n" + "=" * 70)
    print("SELECT A GOOD STARTING FRAME")
    print("=" * 70)

    print("""
Controls:

  D / RIGHT ARROW : next frame
  A / LEFT ARROW  : previous frame
  ENTER           : select
  ESC             : cancel

Choose a frame where the GOLD BOB is clearly visible.
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

        # Draw known pivot
        cv2.circle(
            display,
            PIVOT,
            7,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            display,
            "PIVOT",
            (
                PIVOT[0] + 10,
                PIVOT[1]
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

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
# SELECT BOB
# ============================================================

def select_bob(
    cap,
    frame_number
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    ret, frame = cap.read()

    if not ret:
        return None, None

    display = frame.copy()

    # Show known pivot
    cv2.circle(
        display,
        PIVOT,
        7,
        (255, 0, 255),
        -1
    )

    cv2.putText(
        display,
        "KNOWN PIVOT",
        (
            PIVOT[0] + 10,
            PIVOT[1]
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2
    )

    print("\n" + "=" * 70)
    print("SELECT THE GOLD PENDULUM BOB")
    print("=" * 70)

    print("""
ONLY select the GOLD SPHERICAL BOB.

Do NOT select:
  - the string
  - the base
  - the black stand
  - the top clamp

Make the box tight around the gold ball.

Press ENTER when finished.
""")

    roi = cv2.selectROI(
        "SELECT GOLD BOB",
        display,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(
        "SELECT GOLD BOB"
    )

    x, y, w, h = roi

    if w <= 1 or h <= 1:
        return None, None

    center = (
        int(x + w / 2),
        int(y + h / 2)
    )

    return frame, center


# ============================================================
# GOLD CANDIDATES
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
            / (perimeter ** 2)
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

        cx = int(
            moments["m10"]
            / moments["m00"]
        )

        cy = int(
            moments["m01"]
            / moments["m00"]
        )

        candidates.append({
            "center": (cx, cy),
            "radius": radius,
            "area": area,
            "circularity": circularity
        })

    return candidates


# ============================================================
# ANGLE
# ============================================================

def calculate_theta(
    bob
):

    dx = bob[0] - PIVOT[0]
    dy = bob[1] - PIVOT[1]

    return np.arctan2(
        dx,
        dy
    )


# ============================================================
# ANGLE DIFFERENCE
# ============================================================

def angular_difference(
    a,
    b
):

    return np.arctan2(
        np.sin(a - b),
        np.cos(a - b)
    )


# ============================================================
# SELECT BEST CANDIDATE
# ============================================================

def choose_candidate(
    candidates,
    pendulum_length,
    previous_bob,
    previous_theta,
    angular_velocity,
    dt
):

    if not candidates:
        return None

    # Predict next angle
    predicted_theta = (
        previous_theta
        +
        angular_velocity * dt
    )

    predicted_x = (
        PIVOT[0]
        +
        pendulum_length
        * np.sin(predicted_theta)
    )

    predicted_y = (
        PIVOT[1]
        +
        pendulum_length
        * np.cos(predicted_theta)
    )

    predicted_position = np.array([
        predicted_x,
        predicted_y
    ])

    best = None
    best_score = float("inf")

    for candidate in candidates:

        point = np.array(
            candidate["center"],
            dtype=float
        )

        # ----------------------------------------------------
        # Distance from pivot
        # ----------------------------------------------------

        radius = np.linalg.norm(
            point
            -
            np.array(
                PIVOT,
                dtype=float
            )
        )

        radius_error = abs(
            radius - pendulum_length
        ) / pendulum_length

        if radius_error > RADIUS_TOLERANCE:
            continue

        # ----------------------------------------------------
        # Angle
        # ----------------------------------------------------

        candidate_theta = calculate_theta(
            candidate["center"]
        )

        angle_error = abs(
            np.degrees(
                angular_difference(
                    candidate_theta,
                    predicted_theta
                )
            )
        )

        if angle_error > MAX_ANGLE_STEP_DEG:
            continue

        # ----------------------------------------------------
        # Position jump
        # ----------------------------------------------------

        jump = np.linalg.norm(
            point
            -
            np.array(
                previous_bob,
                dtype=float
            )
        )

        if jump > MAX_POSITION_JUMP:
            continue

        # ----------------------------------------------------
        # Distance from predicted point
        # ----------------------------------------------------

        prediction_error = np.linalg.norm(
            point
            -
            predicted_position
        )

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = (
            5.0 * radius_error
            +
            0.10 * angle_error
            +
            0.02 * prediction_error
            -
            0.15 * candidate["circularity"]
        )

        if score < best_score:

            best_score = score
            best = candidate

    return best


# ============================================================
# DRAW
# ============================================================

def draw_frame(
    frame,
    bob,
    pendulum_length,
    trajectory,
    frame_number,
    fps,
    status,
    theta
):

    output = frame.copy()

    # --------------------------------------------------------
    # Expected orbit
    # --------------------------------------------------------

    cv2.circle(
        output,
        PIVOT,
        int(pendulum_length),
        (100, 100, 100),
        1
    )

    # --------------------------------------------------------
    # Trajectory
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
        PIVOT,
        7,
        (255, 0, 255),
        -1
    )

    # --------------------------------------------------------
    # Bob
    # --------------------------------------------------------

    if bob is not None:

        cv2.line(
            output,
            PIVOT,
            bob,
            (0, 255, 255),
            2
        )

        cv2.circle(
            output,
            bob,
            14,
            (0, 255, 0),
            2
        )

        cv2.circle(
            output,
            bob,
            5,
            (0, 0, 255),
            -1
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
        0.6,
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
    print("HORIZON — PENDULUM GEOMETRY TRACKER V2")
    print("=" * 70)

    print(
        f"\nKnown pivot: {PIVOT}"
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Frames     : {total_frames}"
    )

    print(
        f"Resolution : "
        f"{width} × {height}"
    )

    print(
        f"Duration   : "
        f"{total_frames / fps:.2f}s"
    )

    # --------------------------------------------------------
    # Start frame
    # --------------------------------------------------------

    start_frame = choose_start_frame(
        cap,
        total_frames,
        fps
    )

    if start_frame is None:

        cap.release()
        return

    # --------------------------------------------------------
    # Select bob
    # --------------------------------------------------------

    first_frame, bob_initial = select_bob(
        cap,
        start_frame
    )

    if bob_initial is None:

        cap.release()
        return

    print(
        f"\nInitial bob: "
        f"{bob_initial}"
    )

    # --------------------------------------------------------
    # Calculate pendulum length
    # --------------------------------------------------------

    pendulum_length = np.linalg.norm(
        np.array(
            bob_initial,
            dtype=float
        )
        -
        np.array(
            PIVOT,
            dtype=float
        )
    )

    print(
        f"Estimated pendulum length: "
        f"{pendulum_length:.2f}px"
    )

    # Sanity check
    if pendulum_length < 200:

        print()
        print(
            "WARNING: Pendulum length is unexpectedly small."
        )

        print(
            "Check PIVOT_X and PIVOT_Y."
        )

    # --------------------------------------------------------
    # Initial angle
    # --------------------------------------------------------

    theta = calculate_theta(
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
        dtype=float
    )

    theta_array = np.full(
        total_frames,
        np.nan,
        dtype=float
    )

    times = np.arange(
        total_frames
    ) / fps

    trajectory = [
        None
    ] * total_frames

    # --------------------------------------------------------
    # Initial point
    # --------------------------------------------------------

    positions[start_frame] = bob_initial

    theta_array[start_frame] = theta

    trajectory[start_frame] = bob_initial

    previous_bob = bob_initial

    detected_count = 1

    # --------------------------------------------------------
    # Process complete video
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE VIDEO")
    print("=" * 70)

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
        # Gold candidates
        # ----------------------------------------------------

        candidates = find_gold_candidates(
            frame
        )

        # ----------------------------------------------------
        # Geometry filter
        # ----------------------------------------------------

        chosen = choose_candidate(
            candidates,
            pendulum_length,
            previous_bob,
            theta,
            angular_velocity,
            dt
        )

        if chosen is not None:

            bob = chosen["center"]

            new_theta = calculate_theta(
                bob
            )

            delta_theta = angular_difference(
                new_theta,
                theta
            )

            raw_velocity = (
                delta_theta / dt
            )

            angular_velocity = (
                0.75 * angular_velocity
                +
                0.25 * raw_velocity
            )

            theta = new_theta

            positions[
                frame_number
            ] = bob

            theta_array[
                frame_number
            ] = theta

            trajectory[
                frame_number
            ] = bob

            previous_bob = bob

            detected_count += 1

            status = "VALID BOB"

        else:

            bob = None

            status = "MISSING"

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        annotated = draw_frame(
            frame,
            bob,
            pendulum_length,
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
            "Horizon - Geometry Tracker V2",
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
                f"Frame {frame_number:4d}/"
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
    # Valid samples
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

    processed = (
        total_frames - start_frame
    )

    detection_rate = (
        detected_count
        / processed
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
        pivot=np.array(PIVOT),
        pendulum_length=pendulum_length,
        fps=fps,
        width=width,
        height=height
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Pivot              : {PIVOT}"
    )

    print(
        f"Pendulum length    : "
        f"{pendulum_length:.2f}px"
    )

    print(
        f"Processed frames   : "
        f"{processed}"
    )

    print(
        f"Valid detections   : "
        f"{detected_count}"
    )

    print(
        f"Missing detections : "
        f"{processed - detected_count}"
    )

    print(
        f"Detection rate     : "
        f"{detection_rate:.1f}%"
    )

    print()
    print(
        f"Saved data         : "
        f"{OUTPUT_DATA}"
    )

    print(
        f"Saved video        : "
        f"{OUTPUT_VIDEO}"
    )

    # ========================================================
    # PLOT TRAJECTORY
    # ========================================================

    if len(valid_positions) > 1:

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
            [PIVOT[0]],
            [PIVOT[1]],
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
            "Horizon — Pendulum Trajectory"
        )

        plt.grid(True)
        plt.legend()

        # Image coordinates: downward = positive Y
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.show()

        # ====================================================
        # ANGLE
        # ====================================================

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
            "θ (degrees)"
        )

        plt.title(
            "Horizon — Pendulum Angle θ(t)"
        )

        plt.grid(True)
        plt.tight_layout()

        plt.show()

    print("\n" + "=" * 70)
    print("HORIZON GEOMETRY TRACKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()