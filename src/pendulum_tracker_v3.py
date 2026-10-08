import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON — STRING-GUIDED PENDULUM TRACKER V3
# ============================================================
#
# Strategy:
#
#   VIDEO
#      ↓
#   STRING DETECTION
#      ↓
#   PENDULUM ANGLE
#      ↓
#   PREDICT BOB LOCATION
#      ↓
#   GOLD COLOR DETECTION NEAR PREDICTION
#      ↓
#   VALIDATE WITH GEOMETRY
#      ↓
#   x(t), y(t), theta(t)
#
# ============================================================


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)


# ============================================================
# FIXED PIVOT
# ============================================================

PIVOT = np.array(
    [250.0, 45.0]
)


# ============================================================
# GOLD COLOR PARAMETERS
# ============================================================

GOLD_HSV_LOW = np.array(
    [5, 70, 60]
)

GOLD_HSV_HIGH = np.array(
    [40, 255, 255]
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

SEARCH_RADIUS = 110

MIN_AREA = 20
MAX_AREA = 2500

MIN_CIRCULARITY = 0.20


# ============================================================
# STRING DETECTOR PARAMETERS
# ============================================================

CANNY_LOW = 30
CANNY_HIGH = 100

TOP_Y = 45
BOTTOM_Y = 620


# ============================================================
# GOLD DETECTION
# ============================================================

def detect_gold_candidates(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        GOLD_HSV_LOW,
        GOLD_HSV_HIGH
    )

    # Clean the mask
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
                "x": cx,
                "y": cy,
                "area": area,
                "circularity": circularity,
                "bbox": (
                    x,
                    y,
                    w,
                    h
                )
            }
        )

    return candidates, mask


# ============================================================
# STRING DETECTION
# ============================================================

def detect_string_angle(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH
    )

    mask = np.zeros_like(
        edges
    )

    mask[
        TOP_Y:BOTTOM_Y,
        :
    ] = edges[
        TOP_Y:BOTTOM_Y,
        :
    ]

    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 360,
        threshold=20,
        minLineLength=60,
        maxLineGap=50
    )

    if lines is None:
        return None

    px, py = PIVOT

    candidates = []

    for line in lines:

        values = np.asarray(
            line
        ).reshape(-1)

        if len(values) != 4:
            continue

        x1, y1, x2, y2 = map(
            int,
            values
        )

        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(
            dx,
            dy
        )

        if length < 60:
            continue

        # Reject horizontal lines
        if abs(dy) < abs(dx) * 1.2:
            continue

        d1 = np.hypot(
            x1 - px,
            y1 - py
        )

        d2 = np.hypot(
            x2 - px,
            y2 - py
        )

        near_pivot = min(
            d1,
            d2
        )

        if near_pivot > 220:
            continue

        if d1 > d2:

            far_x = x1
            far_y = y1

        else:

            far_x = x2
            far_y = y2

        horizontal_offset = abs(
            far_x - px
        )

        # Reject central support
        if horizontal_offset < 35:
            continue

        vx = far_x - px
        vy = far_y - py

        distance = np.hypot(
            vx,
            vy
        )

        if distance < 150:
            continue

        angle = np.arctan2(
            vx,
            vy
        )

        angle_deg = abs(
            np.degrees(angle)
        )

        # Extremely small angles are unreliable
        if angle_deg < 3:
            continue

        # Score line
        score = (
            2.0 * horizontal_offset
            +
            0.5 * distance
            +
            0.25 * length
        )

        candidates.append(
            {
                "score": score,
                "angle": angle,
                "length": length,
                "distance": distance,
                "line": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# FIND GOLD BOB NEAR EXPECTED POSITION
# ============================================================

def find_best_bob(
    candidates,
    expected_position,
    previous_position=None
):

    if not candidates:
        return None

    best = None
    best_score = float("inf")

    ex, ey = expected_position

    for candidate in candidates:

        cx = candidate["x"]
        cy = candidate["y"]

        distance_to_prediction = np.hypot(
            cx - ex,
            cy - ey
        )

        if distance_to_prediction > SEARCH_RADIUS:
            continue

        # ----------------------------------------------------
        # Prefer candidates close to predicted position
        # ----------------------------------------------------

        score = distance_to_prediction

        # ----------------------------------------------------
        # If we have a previous bob position, favour smooth
        # movement as an additional constraint.
        # ----------------------------------------------------

        if previous_position is not None:

            px, py = previous_position

            movement = np.hypot(
                cx - px,
                cy - py
            )

            # Penalise extremely large jumps
            if movement > 130:
                score += (
                    movement - 130
                ) * 2.0

        if score < best_score:

            best_score = score
            best = candidate

    return best


# ============================================================
# DRAW
# ============================================================

def draw_tracking(
    frame,
    angle,
    predicted,
    bob,
    status
):

    display = frame.copy()

    px, py = map(
        int,
        PIVOT
    )

    # Pivot
    cv2.circle(
        display,
        (px, py),
        7,
        (255, 0, 255),
        -1
    )

    cv2.putText(
        display,
        "PIVOT",
        (
            px + 10,
            py
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2
    )

    # Predicted bob position
    if predicted is not None:

        pred_x, pred_y = map(
            int,
            predicted
        )

        cv2.circle(
            display,
            (
                pred_x,
                pred_y
            ),
            SEARCH_RADIUS,
            (0, 255, 255),
            1
        )

        cv2.circle(
            display,
            (
                pred_x,
                pred_y
            ),
            5,
            (0, 255, 255),
            -1
        )

        cv2.line(
            display,
            (px, py),
            (
                pred_x,
                pred_y
            ),
            (0, 255, 255),
            2
        )

    # Actual detected bob
    if bob is not None:

        bx = int(
            bob["x"]
        )

        by = int(
            bob["y"]
        )

        x, y, w, h = (
            bob["bbox"]
        )

        cv2.rectangle(
            display,
            (
                x,
                y
            ),
            (
                x + w,
                y + h
            ),
            (0, 255, 0),
            2
        )

        cv2.circle(
            display,
            (
                bx,
                by
            ),
            5,
            (0, 0, 255),
            -1
        )

    # Status
    colour = (
        (0, 255, 0)
        if status == "VALID"
        else
        (0, 0, 255)
    )

    cv2.putText(
        display,
        f"STATUS: {status}",
        (
            10,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2
    )

    if angle is not None:

        angle_deg = np.degrees(
            angle
        )

        cv2.putText(
            display,
            f"Theta: {angle_deg:.2f} deg",
            (
                10,
                60
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return display


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
    print("HORIZON — STRING-GUIDED GOLD BOB TRACKER V3")
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

    print()
    print(
        "Processing entire video..."
    )

    # ========================================================
    # FIRST PASS
    #
    # We use the gold detector + string angle to estimate
    # the pendulum length automatically.
    # ========================================================

    angles = []
    positions = []
    times = []

    previous_position = None
    previous_angle = None

    frame_number = 0

    # --------------------------------------------------------
    # We first find a useful frame where the bob is clearly
    # separated from the base.
    # --------------------------------------------------------

    calibration_found = False

    pendulum_length = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        time = frame_number / fps

        # String
        string = detect_string_angle(
            frame
        )

        # Gold
        gold_candidates, _ = (
            detect_gold_candidates(
                frame
            )
        )

        angle = None

        if string is not None:

            angle = string["angle"]

            previous_angle = angle

        elif previous_angle is not None:

            angle = previous_angle

        # ----------------------------------------------------
        # Look for an obvious gold bob during calibration
        # ----------------------------------------------------

        if angle is not None:

            # Try several plausible pendulum lengths
            # and find a gold candidate that fits one.
            #
            # This is only used to initialise L.

            for L in np.linspace(
                400,
                620,
                45
            ):

                expected_x = (
                    PIVOT[0]
                    +
                    L
                    *
                    np.sin(angle)
                )

                expected_y = (
                    PIVOT[1]
                    +
                    L
                    *
                    np.cos(angle)
                )

                for candidate in gold_candidates:

                    distance = np.hypot(
                        candidate["x"]
                        -
                        expected_x,
                        candidate["y"]
                        -
                        expected_y
                    )

                    if distance < 45:

                        pendulum_length = (
                            np.hypot(
                                candidate["x"]
                                -
                                PIVOT[0],
                                candidate["y"]
                                -
                                PIVOT[1]
                            )
                        )

                        calibration_found = True

                        print()
                        print(
                            "CALIBRATION FOUND"
                        )

                        print(
                            f"Frame    : "
                            f"{frame_number}"
                        )

                        print(
                            f"Time     : "
                            f"{time:.2f}s"
                        )

                        print(
                            f"Bob      : "
                            f"({candidate['x']:.1f}, "
                            f"{candidate['y']:.1f})"
                        )

                        print(
                            f"Length   : "
                            f"{pendulum_length:.1f}px"
                        )

                        break

                if calibration_found:
                    break

        if calibration_found:
            break

        frame_number += 1

    # ========================================================
    # CALIBRATION FAILURE
    # ========================================================

    if pendulum_length is None:

        cap.release()

        raise RuntimeError(
            "Could not automatically calibrate "
            "the pendulum length."
        )

    # ========================================================
    # RESTART VIDEO
    # ========================================================

    cap.release()

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    # ========================================================
    # TRACK ENTIRE VIDEO
    # ========================================================

    frame_number = 0

    previous_position = None
    previous_angle = None

    detected_count = 0
    missing_count = 0

    trajectory = []

    print()
    print("=" * 70)
    print("TRACKING ENTIRE VIDEO")
    print("=" * 70)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        time = frame_number / fps

        # ----------------------------------------------------
        # Detect string
        # ----------------------------------------------------

        string = detect_string_angle(
            frame
        )

        if string is not None:

            angle = string["angle"]

            previous_angle = angle

        else:

            angle = previous_angle

        # ----------------------------------------------------
        # Predict bob from pendulum geometry
        # ----------------------------------------------------

        predicted = None

        if angle is not None:

            predicted_x = (
                PIVOT[0]
                +
                pendulum_length
                *
                np.sin(angle)
            )

            predicted_y = (
                PIVOT[1]
                +
                pendulum_length
                *
                np.cos(angle)
            )

            predicted = (
                predicted_x,
                predicted_y
            )

        # ----------------------------------------------------
        # Gold candidates
        # ----------------------------------------------------

        gold_candidates, _ = (
            detect_gold_candidates(
                frame
            )
        )

        bob = None

        if predicted is not None:

            bob = find_best_bob(
                gold_candidates,
                predicted,
                previous_position
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        if bob is not None:

            bx = bob["x"]
            by = bob["y"]

            previous_position = (
                bx,
                by
            )

            detected_count += 1

            status = "VALID"

            trajectory.append(
                (
                    time,
                    bx,
                    by,
                    angle
                )
            )

        else:

            missing_count += 1

            status = "NO BOB"

            trajectory.append(
                (
                    time,
                    np.nan,
                    np.nan,
                    angle
                    if angle is not None
                    else np.nan
                )
            )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display = draw_tracking(
            frame,
            angle,
            predicted,
            bob,
            status
        )

        cv2.putText(
            display,
            f"Frame: "
            f"{frame_number}/{total - 1}",
            (
                10,
                height - 40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Time: "
            f"{time:.2f}s",
            (
                10,
                height - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Horizon - Pendulum V3",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        if frame_number % 30 == 0:

            print(
                f"Frame "
                f"{frame_number:4d}/"
                f"{total - 1} "
                f"("
                f"{100 * frame_number / total:5.1f}%"
                f") | "
                f"{status}"
            )

        frame_number += 1

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    # ========================================================
    # SAVE DATA
    # ========================================================

    trajectory = np.asarray(
        trajectory,
        dtype=float
    )

    times = trajectory[:, 0]
    x = trajectory[:, 1]
    y = trajectory[:, 2]
    theta = trajectory[:, 3]

    np.savez(
        "horizon_pendulum_v3.npz",
        time=times,
        x=x,
        y=y,
        theta=theta,
        pivot=PIVOT,
        pendulum_length=pendulum_length
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_processed = (
        detected_count
        +
        missing_count
    )

    detection_rate = (
        100
        *
        detected_count
        /
        total_processed
        if total_processed > 0
        else 0
    )

    print()
    print("=" * 70)
    print("HORIZON V3 TRACKING COMPLETE")
    print("=" * 70)

    print(
        f"Processed frames : "
        f"{total_processed}"
    )

    print(
        f"Detected frames  : "
        f"{detected_count}"
    )

    print(
        f"Missing frames   : "
        f"{missing_count}"
    )

    print(
        f"Detection rate   : "
        f"{detection_rate:.1f}%"
    )

    print(
        f"Pendulum length  : "
        f"{pendulum_length:.1f}px"
    )

    print()
    print(
        "Saved:"
    )

    print(
        "horizon_pendulum_v3.npz"
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
        times[valid_theta],
        np.degrees(
            theta[valid_theta]
        )
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
        "Horizon — Pendulum Trajectory"
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


if __name__ == "__main__":
    main()