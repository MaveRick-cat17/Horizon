import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON — PENDULUM TRACKER V4
# BIDIRECTIONAL PHYSICS-CONSTRAINED TRACKING
# ============================================================

VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_DATA = "horizon_pendulum_v4.npz"


# ============================================================
# KNOWN PIVOT
# ============================================================

PIVOT = np.array(
    [250.0, 45.0],
    dtype=float
)


# ============================================================
# GOLD HSV RANGE
# ============================================================

GOLD_LOW = np.array(
    [5, 60, 50]
)

GOLD_HIGH = np.array(
    [40, 255, 255]
)


# ============================================================
# BOB DETECTION
# ============================================================

MIN_AREA = 15
MAX_AREA = 3000
MIN_CIRCULARITY = 0.15


# ============================================================
# TRACKING PARAMETERS
# ============================================================

# Approximate search around predicted position
SEARCH_RADIUS = 100

# Maximum physically reasonable frame-to-frame movement
MAX_STEP = 80

# Radius tolerance
RADIUS_TOLERANCE = 90


# ============================================================
# GOLD CANDIDATES
# ============================================================

def detect_gold_candidates(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        GOLD_LOW,
        GOLD_HIGH
    )

    kernel = np.ones(
        (3, 3),
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

        M = cv2.moments(
            contour
        )

        if M["m00"] == 0:
            continue

        cx = (
            M["m10"]
            /
            M["m00"]
        )

        cy = (
            M["m01"]
            /
            M["m00"]
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        candidates.append(
            {
                "x": float(cx),
                "y": float(cy),
                "area": float(area),
                "circularity": float(circularity),
                "bbox": (
                    x,
                    y,
                    w,
                    h
                )
            }
        )

    return candidates


# ============================================================
# FRAME BROWSER
# ============================================================

def choose_start_frame(cap, total, fps):

    frame_number = 39

    print()
    print("=" * 70)
    print("SELECT INITIAL BOB FRAME")
    print("=" * 70)
    print()
    print("The browser starts around frame 39.")
    print()
    print("D / RIGHT ARROW = next frame")
    print("A / LEFT ARROW  = previous frame")
    print("ENTER           = select frame")
    print("Q / ESC         = cancel")
    print()

    while True:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:
            continue

        display = frame.copy()

        cv2.putText(
            display,
            f"Frame: {frame_number}/{total - 1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Time: {frame_number / fps:.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "D=next  A=previous  ENTER=select",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Horizon - Select Bob Frame",
            display
        )

        key = cv2.waitKey(0) & 0xFF

        # D / right arrow
        if key == ord("d") or key == 83:

            frame_number = min(
                frame_number + 1,
                total - 1
            )

        # A / left arrow
        elif key == ord("a") or key == 81:

            frame_number = max(
                frame_number - 1,
                0
            )

        # ENTER
        elif key == 13:

            cv2.destroyWindow(
                "Horizon - Select Bob Frame"
            )

            return frame_number, frame

        # ESC / Q
        elif key == 27 or key == ord("q"):

            cv2.destroyAllWindows()

            raise RuntimeError(
                "Frame selection cancelled."
            )


# ============================================================
# SELECT BOB ROI
# ============================================================

def select_bob(frame):

    print()
    print("=" * 70)
    print("SELECT THE REAL GOLD PENDULUM BOB")
    print("=" * 70)
    print()
    print("Select ONLY the gold ball.")
    print("Do NOT select the string.")
    print("Do NOT select the black base.")
    print()

    roi = cv2.selectROI(
        "Select Pendulum Bob",
        frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(
        "Select Pendulum Bob"
    )

    x, y, w, h = roi

    if w <= 0 or h <= 0:

        raise RuntimeError(
            "Invalid bob ROI."
        )

    cx = x + w / 2
    cy = y + h / 2

    print(
        f"Selected ROI: "
        f"x={x}, y={y}, "
        f"w={w}, h={h}"
    )

    print(
        f"Initial bob centre: "
        f"({cx:.1f}, {cy:.1f})"
    )

    return np.array(
        [cx, cy],
        dtype=float
    )


# ============================================================
# RADIUS
# ============================================================

def distance_from_pivot(position):

    return np.linalg.norm(
        position - PIVOT
    )


# ============================================================
# CANDIDATE SELECTION
# ============================================================

def choose_candidate(
    candidates,
    predicted_position,
    expected_radius,
    previous_position,
    previous_velocity
):

    if len(candidates) == 0:
        return None

    best = None
    best_score = float("inf")

    for candidate in candidates:

        position = np.array(
            [
                candidate["x"],
                candidate["y"]
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # Distance from predicted position
        # ----------------------------------------------------

        prediction_error = np.linalg.norm(
            position - predicted_position
        )

        if prediction_error > SEARCH_RADIUS:
            continue

        # ----------------------------------------------------
        # Distance from pivot
        # ----------------------------------------------------

        radius = distance_from_pivot(
            position
        )

        radius_error = abs(
            radius - expected_radius
        )

        if radius_error > RADIUS_TOLERANCE:
            continue

        # ----------------------------------------------------
        # Frame-to-frame movement
        # ----------------------------------------------------

        movement = np.linalg.norm(
            position - previous_position
        )

        if movement > MAX_STEP:
            continue

        # ----------------------------------------------------
        # Velocity consistency
        # ----------------------------------------------------

        velocity_error = 0.0

        if previous_velocity is not None:

            predicted_velocity_position = (
                previous_position
                +
                previous_velocity
            )

            velocity_error = np.linalg.norm(
                position
                -
                predicted_velocity_position
            )

        # ----------------------------------------------------
        # Candidate score
        # Lower is better.
        # ----------------------------------------------------

        score = (
            4.0 * prediction_error
            +
            1.5 * radius_error
            +
            1.0 * movement
            +
            1.5 * velocity_error
        )

        # Slight preference for circular objects
        score -= (
            candidate["circularity"]
            * 10
        )

        if score < best_score:

            best_score = score
            best = candidate

    return best


# ============================================================
# TRACK IN ONE DIRECTION
# ============================================================

def track_direction(
    cap,
    start_frame,
    end_frame,
    step,
    fps,
    initial_position,
    expected_radius
):

    positions = {}
    angles = {}

    previous_position = (
        initial_position.copy()
    )

    previous_previous_position = None

    previous_velocity = None

    frame_number = start_frame

    while (
        frame_number >= 0
        and frame_number < int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:
            break

        candidates = detect_gold_candidates(
            frame
        )

        # ----------------------------------------------------
        # Estimate velocity
        # ----------------------------------------------------

        if (
            previous_previous_position
            is not None
        ):

            previous_velocity = (
                previous_position
                -
                previous_previous_position
            )

        # ----------------------------------------------------
        # Predict next position
        # ----------------------------------------------------

        if previous_velocity is not None:

            predicted = (
                previous_position
                +
                previous_velocity
            )

        else:

            predicted = (
                previous_position.copy()
            )

        # ----------------------------------------------------
        # Find bob
        # ----------------------------------------------------

        bob = choose_candidate(
            candidates,
            predicted,
            expected_radius,
            previous_position,
            previous_velocity
        )

        if bob is not None:

            position = np.array(
                [
                    bob["x"],
                    bob["y"]
                ],
                dtype=float
            )

            positions[frame_number] = (
                position.copy()
            )

            angle = np.arctan2(
                position[0] - PIVOT[0],
                position[1] - PIVOT[1]
            )

            angles[frame_number] = (
                angle
            )

            previous_previous_position = (
                previous_position.copy()
            )

            previous_position = (
                position
            )

        frame_number += step

    return positions, angles


# ============================================================
# VISUAL TRACKING PREVIEW
# ============================================================

def preview_tracking(
    cap,
    positions,
    angles,
    fps
):

    print()
    print("=" * 70)
    print("GENERATING TRACKING PREVIEW")
    print("=" * 70)

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    for frame_number in range(
        total
    ):

        if frame_number not in positions:
            continue

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:
            continue

        display = frame.copy()

        px, py = map(
            int,
            PIVOT
        )

        bx, by = map(
            int,
            positions[frame_number]
        )

        # Pivot
        cv2.circle(
            display,
            (px, py),
            7,
            (255, 0, 255),
            -1
        )

        # Bob
        cv2.circle(
            display,
            (bx, by),
            7,
            (0, 0, 255),
            -1
        )

        # Bob-pivot line
        cv2.line(
            display,
            (px, py),
            (bx, by),
            (0, 255, 255),
            2
        )

        theta = np.degrees(
            angles[frame_number]
        )

        cv2.putText(
            display,
            f"Frame: {frame_number}/{total-1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Time: {frame_number / fps:.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Theta: {theta:.2f} deg",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Horizon - V4 Tracking Preview",
            display
        )

        key = cv2.waitKey(
            max(
                1,
                int(
                    1000 / fps
                )
            )
        ) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyWindow(
        "Horizon - V4 Tracking Preview"
    )


# ============================================================
# MAIN
# ============================================================

def main():

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

    total = int(
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

    print()
    print("=" * 70)
    print("HORIZON — PENDULUM TRACKER V4")
    print("=" * 70)

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Frames     : {total}"
    )

    print(
        f"Resolution : {width} × {height}"
    )

    print(
        f"Duration   : {total / fps:.2f}s"
    )

    # ========================================================
    # SELECT GOOD FRAME
    # ========================================================

    start_frame, frame = choose_start_frame(
        cap,
        total,
        fps
    )

    print()
    print(
        f"Selected starting frame: "
        f"{start_frame}"
    )

    print(
        f"Selected time: "
        f"{start_frame / fps:.2f}s"
    )

    # ========================================================
    # SELECT ACTUAL BOB
    # ========================================================

    initial_position = select_bob(
        frame
    )

    expected_radius = distance_from_pivot(
        initial_position
    )

    print(
        f"Initial pendulum radius: "
        f"{expected_radius:.1f}px"
    )

    # ========================================================
    # TRACK BACKWARD
    # ========================================================

    print()
    print("=" * 70)
    print("TRACKING BACKWARD")
    print("=" * 70)

    backward_positions, backward_angles = (
        track_direction(
            cap,
            start_frame - 1,
            0,
            -1,
            fps,
            initial_position,
            expected_radius
        )
    )

    # ========================================================
    # TRACK FORWARD
    # ========================================================

    print()
    print("=" * 70)
    print("TRACKING FORWARD")
    print("=" * 70)

    forward_positions, forward_angles = (
        track_direction(
            cap,
            start_frame + 1,
            total,
            1,
            fps,
            initial_position,
            expected_radius
        )
    )

    # Add selected frame
    backward_positions[start_frame] = (
        initial_position.copy()
    )

    backward_angles[start_frame] = np.arctan2(
        initial_position[0] - PIVOT[0],
        initial_position[1] - PIVOT[1]
    )

    # ========================================================
    # MERGE
    # ========================================================

    all_positions = {}

    all_angles = {}

    all_positions.update(
        backward_positions
    )

    all_positions.update(
        forward_positions
    )

    all_angles.update(
        backward_angles
    )

    all_angles.update(
        forward_angles
    )

    # ========================================================
    # CREATE COMPLETE ARRAYS
    # ========================================================

    time = np.arange(
        total
    ) / fps

    x = np.full(
        total,
        np.nan
    )

    y = np.full(
        total,
        np.nan
    )

    theta = np.full(
        total,
        np.nan
    )

    for frame_number, position in (
        all_positions.items()
    ):

        if (
            frame_number < 0
            or frame_number >= total
        ):
            continue

        x[frame_number] = (
            position[0]
        )

        y[frame_number] = (
            position[1]
        )

        theta[frame_number] = (
            all_angles[
                frame_number
            ]
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
        &
        np.isfinite(theta)
    )

    detected = np.sum(
        valid
    )

    missing = (
        total
        -
        detected
    )

    detection_rate = (
        100
        *
        detected
        /
        total
    )

    print()
    print("=" * 70)
    print("V4 TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Total frames     : {total}"
    )

    print(
        f"Detected frames  : {detected}"
    )

    print(
        f"Missing frames   : {missing}"
    )

    print(
        f"Detection rate   : {detection_rate:.1f}%"
    )

    print(
        f"Pendulum radius  : "
        f"{expected_radius:.1f}px"
    )

    # ========================================================
    # SAVE
    # ========================================================

    np.savez(
        OUTPUT_DATA,
        time=time,
        x=x,
        y=y,
        theta=theta,
        pivot=PIVOT,
        pendulum_length=expected_radius
    )

    print()
    print(
        f"Saved: {OUTPUT_DATA}"
    )

    # ========================================================
    # PLOT ANGLE
    # ========================================================

    valid_theta = np.isfinite(
        theta
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time[valid_theta],
        np.degrees(
            theta[valid_theta]
        ),
        label="θ(t)"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Theta (degrees)"
    )

    plt.title(
        "Horizon — Pendulum Angle θ(t)"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ========================================================
    # PLOT TRAJECTORY
    # ========================================================

    valid_xy = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        x[valid_xy],
        y[valid_xy],
        label="Tracked bob"
    )

    plt.scatter(
        PIVOT[0],
        PIVOT[1],
        s=100,
        label="Pivot"
    )

    plt.xlabel(
        "X position (pixels)"
    )

    plt.ylabel(
        "Y position (pixels)"
    )

    plt.title(
        "Horizon — V4 Pendulum Trajectory"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.axis(
        "equal"
    )

    plt.tight_layout()

    plt.show()

    # ========================================================
    # OPTIONAL PREVIEW
    # ========================================================

    print()
    print(
        "Press ENTER to watch the tracking preview."
    )

    input()

    preview_tracking(
        cap,
        all_positions,
        all_angles,
        fps
    )

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()