import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON — PENDULUM TRACKER V6
#
# PRIMARY:
#     CSRT visual tracking
#
# RECOVERY:
#     Gold detection constrained by pendulum geometry
#
# The user selects the REAL bob once.
# ============================================================


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_DATA = "horizon_pendulum_v6.npz"


# ============================================================
# KNOWN PIVOT
# ============================================================

PIVOT = np.array(
    [250.0, 45.0],
    dtype=float
)


# ============================================================
# GOLD DETECTION
# ============================================================

GOLD_LOW = np.array(
    [5, 70, 60]
)

GOLD_HIGH = np.array(
    [40, 255, 255]
)


MIN_AREA = 15
MAX_AREA = 1500


# ============================================================
# TRACKING
# ============================================================

SEARCH_RADIUS = 120
RADIUS_TOLERANCE = 80
MAX_STEP = 120


# ============================================================
# CREATE CSRT
# ============================================================

def create_csrt():

    if hasattr(
        cv2,
        "TrackerCSRT_create"
    ):

        return cv2.TrackerCSRT_create()

    if hasattr(
        cv2,
        "legacy"
    ) and hasattr(
        cv2.legacy,
        "TrackerCSRT_create"
    ):

        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker is unavailable."
    )


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
            4
            * np.pi
            * area
            /
            (perimeter ** 2)
        )

        if circularity < 0.15:
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
                "bbox": (x, y, w, h)
            }
        )

    return candidates


# ============================================================
# SELECT START FRAME
# ============================================================

def choose_start_frame(
    cap,
    total,
    fps
):

    frame_number = 66

    print()
    print("=" * 70)
    print("SELECT A GOOD BOB FRAME")
    print("=" * 70)

    print()
    print("The browser starts around frame 66.")
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
                "Cancelled."
            )


# ============================================================
# SELECT BOB
# ============================================================

def select_bob(frame):

    print()
    print("=" * 70)
    print("SELECT THE ACTUAL GOLD BOB")
    print("=" * 70)

    print()
    print("Select ONLY the circular gold bob.")
    print("Do NOT include the string.")
    print("Do NOT include the black base.")
    print()

    roi = cv2.selectROI(
        "Select Gold Bob",
        frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(
        "Select Gold Bob"
    )

    x, y, w, h = roi

    if w <= 0 or h <= 0:

        raise RuntimeError(
            "Invalid ROI."
        )

    center = np.array(
        [
            x + w / 2,
            y + h / 2
        ],
        dtype=float
    )

    print(
        f"ROI: x={x}, y={y}, "
        f"w={w}, h={h}"
    )

    print(
        f"Bob centre: "
        f"({center[0]:.1f}, "
        f"{center[1]:.1f})"
    )

    return (
        x,
        y,
        w,
        h
    ), center


# ============================================================
# GEOMETRY
# ============================================================

def radius(position):

    return np.linalg.norm(
        position - PIVOT
    )


def theta_from_position(
    position
):

    return np.arctan2(
        position[0] - PIVOT[0],
        position[1] - PIVOT[1]
    )


# ============================================================
# GOLD RECOVERY
# ============================================================

def recover_bob(
    frame,
    predicted_position,
    expected_radius
):

    candidates = detect_gold_candidates(
        frame
    )

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

        # Distance from prediction
        prediction_error = np.linalg.norm(
            position
            -
            predicted_position
        )

        if prediction_error > SEARCH_RADIUS:
            continue

        # Distance from pivot
        candidate_radius = radius(
            position
        )

        radius_error = abs(
            candidate_radius
            -
            expected_radius
        )

        if radius_error > RADIUS_TOLERANCE:
            continue

        # ----------------------------------------------------
        # Prefer compact circular candidates.
        # ----------------------------------------------------

        circularity_bonus = (
            candidate["circularity"]
            *
            20
        )

        score = (
            prediction_error
            +
            radius_error
            -
            circularity_bonus
        )

        # ----------------------------------------------------
        # Strong penalty for candidates sitting very close
        # to the central support/base.
        # ----------------------------------------------------

        if (
            abs(
                position[0]
                -
                PIVOT[0]
            )
            <
            35
            and
            position[1]
            >
            500
        ):

            score += 150

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
    step,
    total,
    fps,
    initial_bbox,
    initial_center,
    expected_radius
):

    positions = {}
    angles = {}
    statuses = {}

    # --------------------------------------------------------
    # CSRT tracker
    # --------------------------------------------------------

    tracker = create_csrt()

    # --------------------------------------------------------
    # Initialise at start frame
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    ret, frame = cap.read()

    if not ret:

        return (
            positions,
            angles,
            statuses
        )

    x, y, w, h = initial_bbox

    tracker.init(
        frame,
        (
            int(x),
            int(y),
            int(w),
            int(h)
        )
    )

    previous_position = (
        initial_center.copy()
    )

    previous_previous = None

    previous_velocity = None

    frame_number = start_frame

    while (
        0 <= frame_number < total
    ):

        if frame_number != start_frame:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            ret, frame = cap.read()

            if not ret:
                break

        # ====================================================
        # CSRT UPDATE
        # ====================================================

        ok, bbox = tracker.update(
            frame
        )

        candidate_position = None

        if ok:

            bx, by, bw, bh = bbox

            candidate_position = np.array(
                [
                    bx + bw / 2,
                    by + bh / 2
                ],
                dtype=float
            )

            # ------------------------------------------------
            # Basic geometry validation
            # ------------------------------------------------

            candidate_radius = radius(
                candidate_position
            )

            radius_error = abs(
                candidate_radius
                -
                expected_radius
            )

            movement = np.linalg.norm(
                candidate_position
                -
                previous_position
            )

            if (
                radius_error
                >
                RADIUS_TOLERANCE
                or
                movement
                >
                MAX_STEP
            ):

                candidate_position = None

        # ====================================================
        # PREDICTION
        # ====================================================

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

        # ====================================================
        # RECOVERY
        # ====================================================

        if candidate_position is None:

            recovered = recover_bob(
                frame,
                predicted,
                expected_radius
            )

            if recovered is not None:

                candidate_position = np.array(
                    [
                        recovered["x"],
                        recovered["y"]
                    ],
                    dtype=float
                )

                # --------------------------------------------
                # Reinitialise CSRT around recovered bob
                # --------------------------------------------

                rx, ry, rw, rh = (
                    recovered["bbox"]
                )

                new_bbox = (
                    int(rx),
                    int(ry),
                    int(rw),
                    int(rh)
                )

                tracker = create_csrt()

                tracker.init(
                    frame,
                    new_bbox
                )

                statuses[
                    frame_number
                ] = "RECOVERED"

            else:

                statuses[
                    frame_number
                ] = "MISSING"

        else:

            statuses[
                frame_number
            ] = "CSRT"

        # ====================================================
        # ACCEPT POSITION
        # ====================================================

        if candidate_position is not None:

            if previous_previous is not None:

                previous_velocity = (
                    previous_position
                    -
                    previous_previous
                )

            previous_previous = (
                previous_position.copy()
            )

            previous_position = (
                candidate_position.copy()
            )

            positions[
                frame_number
            ] = (
                candidate_position.copy()
            )

            angles[
                frame_number
            ] = theta_from_position(
                candidate_position
            )

        # ====================================================
        # NEXT FRAME
        # ====================================================

        frame_number += step

    return (
        positions,
        angles,
        statuses
    )


# ============================================================
# PREVIEW
# ============================================================

def preview(
    cap,
    positions,
    angles,
    statuses,
    fps,
    total
):

    print()
    print("=" * 70)
    print("V6 TRACKING PREVIEW")
    print("=" * 70)

    print("Press Q to stop.")

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

        status = statuses.get(
            frame_number,
            "UNKNOWN"
        )

        status_color = (
            (0, 255, 0)
            if status == "CSRT"
            else
            (0, 255, 255)
            if status == "RECOVERED"
            else
            (0, 0, 255)
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
            f"Status: {status}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )

        cv2.imshow(
            "Horizon - V6 Pendulum Tracking",
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
    print("HORIZON — PENDULUM TRACKER V6")
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
    # START FRAME
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
    # SELECT REAL BOB
    # ========================================================

    bbox, center = select_bob(
        frame
    )

    expected_radius = radius(
        center
    )

    print()
    print(
        f"Pendulum radius: "
        f"{expected_radius:.1f}px"
    )

    # ========================================================
    # FORWARD
    # ========================================================

    print()
    print("=" * 70)
    print("TRACKING FORWARD")
    print("=" * 70)

    forward_positions, forward_angles, forward_status = (
        track_direction(
            cap,
            start_frame,
            1,
            total,
            fps,
            bbox,
            center,
            expected_radius
        )
    )

    # ========================================================
    # BACKWARD
    #
    # For backward tracking we reverse the video direction.
    # ========================================================

    print()
    print("=" * 70)
    print("TRACKING BACKWARD")
    print("=" * 70)

    backward_positions, backward_angles, backward_status = (
        track_direction(
            cap,
            start_frame,
            -1,
            total,
            fps,
            bbox,
            center,
            expected_radius
        )
    )

    # ========================================================
    # MERGE
    # ========================================================

    positions = {}
    angles = {}
    statuses = {}

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

    statuses.update(
        backward_status
    )

    statuses.update(
        forward_status
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

    status_array = np.full(
        total,
        "",
        dtype="<U12"
    )

    for frame_number in positions:

        if not (
            0 <= frame_number < total
        ):
            continue

        x[frame_number] = (
            positions[
                frame_number
            ][0]
        )

        y[frame_number] = (
            positions[
                frame_number
            ][1]
        )

        theta[frame_number] = (
            angles[
                frame_number
            ]
        )

        status_array[
            frame_number
        ] = statuses.get(
            frame_number,
            "UNKNOWN"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    detected = int(
        np.sum(valid)
    )

    missing = (
        total
        -
        detected
    )

    print()
    print("=" * 70)
    print("V6 TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Total frames     : {total}"
    )

    print(
        f"Tracked frames   : {detected}"
    )

    print(
        f"Missing frames   : {missing}"
    )

    print(
        f"Detection rate   : "
        f"{100 * detected / total:.1f}%"
    )

    csrt_count = np.sum(
        status_array == "CSRT"
    )

    recovery_count = np.sum(
        status_array == "RECOVERED"
    )

    print(
        f"CSRT frames      : {csrt_count}"
    )

    print(
        f"Recovery frames  : {recovery_count}"
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
        pendulum_length=expected_radius,
        status=status_array
    )

    print()
    print(
        f"Saved: {OUTPUT_DATA}"
    )

    # ========================================================
    # ANGLE
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
        )
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Theta (degrees)"
    )

    plt.title(
        "Horizon — V6 Pendulum Angle θ(t)"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()

    # ========================================================
    # TRAJECTORY
    # ========================================================

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        x[valid],
        y[valid],
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
        "Horizon — V6 Pendulum Trajectory"
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
        "Press ENTER for visual preview."
    )

    input()

    preview(
        cap,
        positions,
        angles,
        statuses,
        fps,
        total
    )

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()