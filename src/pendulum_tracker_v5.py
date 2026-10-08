import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON — PENDULUM TRACKER V5
#
# Gold detection
# + pendulum-circle constraint
# + motion continuity
# + STRING GEOMETRY VALIDATION
#
# The string is used to VALIDATE the bob,
# not to directly determine its position.
# ============================================================


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_DATA = "horizon_pendulum_v5.npz"


# ============================================================
# FIXED PIVOT
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
# GOLD DETECTION
# ============================================================

MIN_AREA = 15
MAX_AREA = 3000
MIN_CIRCULARITY = 0.15


# ============================================================
# TRACKING
# ============================================================

SEARCH_RADIUS = 140
MAX_STEP = 110

RADIUS_TOLERANCE = 70


# ============================================================
# STRING VALIDATION
# ============================================================

# Ignore the region immediately around the pivot because
# the metal support produces many false edges.
STRING_START = 55

# Ignore the bob itself.
STRING_END_OFFSET = 25

# Width around the theoretical string line that is searched.
STRING_WIDTH = 5


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

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        cx = (
            moments["m10"]
            /
            moments["m00"]
        )

        cy = (
            moments["m01"]
            /
            moments["m00"]
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
                "bbox": (x, y, w, h)
            }
        )

    return candidates


# ============================================================
# STRING SUPPORT
# ============================================================

def calculate_string_support(
    gray,
    edges,
    candidate
):

    bx = candidate["x"]
    by = candidate["y"]

    vx = bx - PIVOT[0]
    vy = by - PIVOT[1]

    distance = np.hypot(
        vx,
        vy
    )

    if distance < STRING_START + 30:
        return 0.0

    # Unit vector along pivot -> bob
    ux = vx / distance
    uy = vy / distance

    # Perpendicular vector
    px = -uy
    py = ux

    end_distance = (
        distance
        -
        STRING_END_OFFSET
    )

    if end_distance <= STRING_START:
        return 0.0

    samples = np.arange(
        STRING_START,
        end_distance,
        4.0
    )

    if len(samples) < 5:
        return 0.0

    edge_hits = 0
    dark_hits = 0

    valid_samples = 0

    height, width = gray.shape

    for d in samples:

        cx = (
            PIVOT[0]
            +
            ux * d
        )

        cy = (
            PIVOT[1]
            +
            uy * d
        )

        ix = int(round(cx))
        iy = int(round(cy))

        if (
            ix < 8
            or ix >= width - 8
            or iy < 8
            or iy >= height - 8
        ):
            continue

        valid_samples += 1

        # ----------------------------------------------------
        # Look for string edges across a narrow perpendicular
        # corridor.
        # ----------------------------------------------------

        found_edge = False

        for offset in range(
            -STRING_WIDTH,
            STRING_WIDTH + 1
        ):

            sx = int(
                round(
                    cx
                    +
                    px * offset
                )
            )

            sy = int(
                round(
                    cy
                    +
                    py * offset
                )
            )

            if (
                0 <= sx < width
                and
                0 <= sy < height
            ):

                if edges[sy, sx] > 0:
                    found_edge = True
                    break

        if found_edge:
            edge_hits += 1

        # ----------------------------------------------------
        # Also check whether the centre of the theoretical
        # string is darker than its surroundings.
        # ----------------------------------------------------

        center_value = float(
            gray[iy, ix]
        )

        side_values = []

        for offset in (
            -7,
            7
        ):

            sx = int(
                round(
                    cx
                    +
                    px * offset
                )
            )

            sy = int(
                round(
                    cy
                    +
                    py * offset
                )
            )

            if (
                0 <= sx < width
                and
                0 <= sy < height
            ):

                side_values.append(
                    float(
                        gray[sy, sx]
                    )
                )

        if len(side_values) == 2:

            surroundings = np.mean(
                side_values
            )

            if (
                surroundings
                -
                center_value
                >
                8
            ):
                dark_hits += 1

    if valid_samples == 0:
        return 0.0

    edge_score = (
        edge_hits
        /
        valid_samples
    )

    dark_score = (
        dark_hits
        /
        valid_samples
    )

    # String edge evidence is more reliable than darkness.
    score = (
        0.70 * edge_score
        +
        0.30 * dark_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# FRAME BROWSER
# ============================================================

def choose_start_frame(
    cap,
    total,
    fps
):

    frame_number = 39

    print()
    print("=" * 70)
    print("SELECT INITIAL BOB FRAME")
    print("=" * 70)

    print()
    print("D / RIGHT ARROW = next")
    print("A / LEFT ARROW  = previous")
    print("ENTER / SPACE   = select")
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
            "D=next A=previous ENTER=select",
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

        if key == ord("d") or key == 83:

            frame_number = min(
                frame_number + 1,
                total - 1
            )

        elif key == ord("a") or key == 81:

            frame_number = max(
                frame_number - 1,
                0
            )

        elif key == 13 or key == 32:

            cv2.destroyWindow(
                "Horizon - Select Bob Frame"
            )

            return frame_number, frame

        elif key == ord("q") or key == 27:

            cv2.destroyAllWindows()

            raise RuntimeError(
                "Frame selection cancelled."
            )


# ============================================================
# SELECT BOB
# ============================================================

def select_bob(frame):

    print()
    print("=" * 70)
    print("SELECT THE REAL GOLD BOB")
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
            "Invalid ROI."
        )

    position = np.array(
        [
            x + w / 2,
            y + h / 2
        ],
        dtype=float
    )

    print(
        f"Selected ROI: "
        f"x={x}, y={y}, "
        f"w={w}, h={h}"
    )

    print(
        f"Bob centre: "
        f"({position[0]:.1f}, "
        f"{position[1]:.1f})"
    )

    return position


# ============================================================
# CANDIDATE SCORING
# ============================================================

def choose_candidate(
    candidates,
    predicted_position,
    expected_radius,
    previous_position,
    previous_velocity,
    gray,
    edges
):

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
        # 1. Distance from predicted position
        # ----------------------------------------------------

        prediction_error = np.linalg.norm(
            position
            -
            predicted_position
        )

        if prediction_error > SEARCH_RADIUS:
            continue

        # ----------------------------------------------------
        # 2. Pendulum radius
        # ----------------------------------------------------

        radius = np.linalg.norm(
            position
            -
            PIVOT
        )

        radius_error = abs(
            radius
            -
            expected_radius
        )

        if radius_error > RADIUS_TOLERANCE:
            continue

        # ----------------------------------------------------
        # 3. Movement continuity
        # ----------------------------------------------------

        movement = np.linalg.norm(
            position
            -
            previous_position
        )

        if movement > MAX_STEP:
            continue

        # ----------------------------------------------------
        # 4. Velocity consistency
        # ----------------------------------------------------

        velocity_error = 0.0

        if previous_velocity is not None:

            velocity_prediction = (
                previous_position
                +
                previous_velocity
            )

            velocity_error = np.linalg.norm(
                position
                -
                velocity_prediction
            )

        # ----------------------------------------------------
        # 5. STRING SUPPORT
        # ----------------------------------------------------

        string_score = (
            calculate_string_support(
                gray,
                edges,
                candidate
            )
        )

        # ----------------------------------------------------
        # Reject candidates with almost no string evidence.
        # ----------------------------------------------------

        if string_score < 0.08:
            continue

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        score = (

            2.0
            *
            prediction_error
            /

            SEARCH_RADIUS

            +

            1.5
            *
            radius_error
            /
            RADIUS_TOLERANCE

            +

            1.0
            *
            movement
            /
            MAX_STEP

            +

            1.0
            *
            velocity_error
            /
            MAX_STEP

            +

            3.0
            *
            (1.0 - string_score)

            -

            0.5
            *
            candidate["circularity"]
        )

        if score < best_score:

            best_score = score
            best = candidate

    return best


# ============================================================
# TRACK ONE DIRECTION
# ============================================================

def track_direction(
    cap,
    start_frame,
    end_condition,
    step,
    total,
    expected_radius,
    initial_position
):

    positions = {}
    angles = {}
    scores = {}

    previous_position = (
        initial_position.copy()
    )

    previous_previous_position = None

    previous_velocity = None

    frame_number = start_frame

    while (
        0 <= frame_number < total
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # Image processing for string validation
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            gray,
            30,
            100
        )

        candidates = detect_gold_candidates(
            frame
        )

        # ----------------------------------------------------
        # Estimate velocity
        # ----------------------------------------------------

        if previous_previous_position is not None:

            previous_velocity = (
                previous_position
                -
                previous_previous_position
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if previous_velocity is not None:

            predicted_position = (
                previous_position
                +
                previous_velocity
            )

        else:

            predicted_position = (
                previous_position.copy()
            )

        # ----------------------------------------------------
        # Candidate selection
        # ----------------------------------------------------

        bob = choose_candidate(
            candidates,
            predicted_position,
            expected_radius,
            previous_position,
            previous_velocity,
            gray,
            edges
        )

        if bob is not None:

            position = np.array(
                [
                    bob["x"],
                    bob["y"]
                ],
                dtype=float
            )

            string_score = (
                calculate_string_support(
                    gray,
                    edges,
                    bob
                )
            )

            positions[frame_number] = (
                position.copy()
            )

            scores[frame_number] = (
                string_score
            )

            theta = np.arctan2(
                position[0] - PIVOT[0],
                position[1] - PIVOT[1]
            )

            angles[frame_number] = theta

            previous_previous_position = (
                previous_position.copy()
            )

            previous_position = (
                position.copy()
            )

        frame_number += step

    return (
        positions,
        angles,
        scores
    )


# ============================================================
# PREVIEW
# ============================================================

def preview(
    cap,
    positions,
    angles,
    scores,
    fps,
    total
):

    print()
    print("=" * 70)
    print("TRACKING PREVIEW")
    print("=" * 70)

    print("Press Q to stop preview.")

    for frame_number in range(total):

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

        # ----------------------------------------------------
        # Pivot
        # ----------------------------------------------------

        cv2.circle(
            display,
            (px, py),
            7,
            (255, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # Bob
        # ----------------------------------------------------

        cv2.circle(
            display,
            (bx, by),
            9,
            (0, 0, 255),
            2
        )

        cv2.circle(
            display,
            (bx, by),
            3,
            (0, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # Pivot -> bob
        # ----------------------------------------------------

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

        support = scores.get(
            frame_number,
            0.0
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

        cv2.putText(
            display,
            f"String support: {support:.2f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Horizon - V5 Tracking",
            display
        )

        key = cv2.waitKey(
            max(
                1,
                int(1000 / fps)
            )
        ) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


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
    print("HORIZON — PENDULUM TRACKER V5")
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
    # SELECT FRAME
    # ========================================================

    start_frame, frame = choose_start_frame(
        cap,
        total,
        fps
    )

    print()
    print(
        f"Starting frame : {start_frame}"
    )

    print(
        f"Starting time  : "
        f"{start_frame / fps:.2f}s"
    )

    # ========================================================
    # SELECT BOB
    # ========================================================

    initial_position = select_bob(
        frame
    )

    expected_radius = np.linalg.norm(
        initial_position
        -
        PIVOT
    )

    print()
    print(
        f"Initial radius : "
        f"{expected_radius:.1f}px"
    )

    # ========================================================
    # BACKWARD
    # ========================================================

    print()
    print("=" * 70)
    print("TRACKING BACKWARD")
    print("=" * 70)

    backward_positions, backward_angles, backward_scores = (
        track_direction(
            cap,
            start_frame - 1,
            0,
            -1,
            total,
            expected_radius,
            initial_position
        )
    )

    # ========================================================
    # FORWARD
    # ========================================================

    print()
    print("=" * 70)
    print("TRACKING FORWARD")
    print("=" * 70)

    forward_positions, forward_angles, forward_scores = (
        track_direction(
            cap,
            start_frame + 1,
            total,
            1,
            total,
            expected_radius,
            initial_position
        )
    )

    # ========================================================
    # ADD INITIAL FRAME
    # ========================================================

    initial_theta = np.arctan2(
        initial_position[0] - PIVOT[0],
        initial_position[1] - PIVOT[1]
    )

    backward_positions[start_frame] = (
        initial_position.copy()
    )

    backward_angles[start_frame] = (
        initial_theta
    )

    backward_scores[start_frame] = 1.0

    # ========================================================
    # MERGE
    # ========================================================

    positions = {}
    angles = {}
    scores = {}

    positions.update(
        backward_positions
    )

    positions.update(
        forward_positions
    )

    angles.update(
        backward_angles
    )

    angles.update(
        forward_angles
    )

    scores.update(
        backward_scores
    )

    scores.update(
        forward_scores
    )

    # ========================================================
    # ARRAYS
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

    string_score = np.full(
        total,
        np.nan
    )

    for frame_number in positions:

        if (
            frame_number < 0
            or frame_number >= total
        ):
            continue

        x[frame_number] = (
            positions[frame_number][0]
        )

        y[frame_number] = (
            positions[frame_number][1]
        )

        theta[frame_number] = (
            angles[frame_number]
        )

        string_score[frame_number] = (
            scores[frame_number]
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

    detected = int(
        np.sum(valid)
    )

    missing = (
        total
        -
        detected
    )

    rate = (
        100.0
        *
        detected
        /
        total
    )

    print()
    print("=" * 70)
    print("V5 TRACKING SUMMARY")
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
        f"Detection rate   : {rate:.1f}%"
    )

    print(
        f"Pendulum radius  : "
        f"{expected_radius:.1f}px"
    )

    valid_scores = string_score[
        np.isfinite(string_score)
    ]

    if len(valid_scores) > 0:

        print(
            f"Mean string score: "
            f"{np.mean(valid_scores):.3f}"
        )

        print(
            f"Min string score : "
            f"{np.min(valid_scores):.3f}"
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
        string_score=string_score,
        pivot=PIVOT,
        pendulum_length=expected_radius
    )

    print()
    print(
        f"Saved: {OUTPUT_DATA}"
    )

    # ========================================================
    # ANGLE PLOT
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
        "Horizon — V5 Pendulum Angle θ(t)"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ========================================================
    # SPATIAL TRAJECTORY
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
        "Horizon — V5 Pendulum Trajectory"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ========================================================
    # PREVIEW
    # ========================================================

    print()
    print(
        "Press ENTER for tracking preview."
    )

    input()

    preview(
        cap,
        positions,
        angles,
        scores,
        fps,
        total
    )

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()