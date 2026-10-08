import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON — PENDULUM TRACKER V7
#
# Tight gold-object initialization
# + CSRT tracking
# + geometry validation
# + motion prediction
# + gold recovery
#
# IMPORTANT:
# CSRT is NEVER trusted blindly.
# ============================================================


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_DATA = "horizon_pendulum_v7.npz"


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
    [45, 255, 255]
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

MIN_AREA = 12
MAX_AREA = 2500

RADIUS_TOLERANCE = 65.0

MAX_STEP = 100.0

SEARCH_RADIUS = 130.0

PREDICTION_WEIGHT = 1.0
RADIUS_WEIGHT = 1.5

MIN_CIRCULARITY = 0.12


# ============================================================
# CSRT
# ============================================================

def create_csrt():

    if hasattr(
        cv2,
        "TrackerCSRT_create"
    ):

        return cv2.TrackerCSRT_create()

    if (
        hasattr(cv2, "legacy")
        and
        hasattr(
            cv2.legacy,
            "TrackerCSRT_create"
        )
    ):

        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker unavailable."
    )


# ============================================================
# GOLD MASK
# ============================================================

def gold_mask(frame):

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

    return mask


# ============================================================
# CONTOUR CANDIDATES
# ============================================================

def gold_candidates(
    frame,
    roi=None
):

    mask = gold_mask(frame)

    if roi is not None:

        x, y, w, h = roi

        roi_mask = np.zeros_like(
            mask
        )

        roi_mask[
            y:y+h,
            x:x+w
        ] = mask[
            y:y+h,
            x:x+w
        ]

        mask = roi_mask

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
            /
            (perimeter * perimeter)
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

        # ----------------------------------------------------
        # Ball should roughly resemble a compact object.
        # ----------------------------------------------------

        aspect = (
            max(w, h)
            /
            max(
                1,
                min(w, h)
            )
        )

        if aspect > 3.0:
            continue

        candidates.append(
            {
                "center": np.array(
                    [cx, cy],
                    dtype=float
                ),
                "area": float(area),
                "circularity": float(
                    circularity
                ),
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
# RADIUS
# ============================================================

def get_radius(
    point
):

    return np.linalg.norm(
        point - PIVOT
    )


# ============================================================
# ANGLE
# ============================================================

def get_theta(
    point
):

    return np.arctan2(
        point[0] - PIVOT[0],
        point[1] - PIVOT[1]
    )


# ============================================================
# INITIAL BOB EXTRACTION
#
# This is the important V7 improvement.
#
# User may select a large rectangle.
# We find the actual gold object INSIDE it.
# ============================================================

def extract_bob_from_roi(
    frame,
    roi
):

    candidates = gold_candidates(
        frame,
        roi
    )

    if not candidates:

        raise RuntimeError(
            "No gold object was found "
            "inside the selected ROI."
        )

    # --------------------------------------------------------
    # The user-selected ROI defines the intended object.
    # Choose the largest compact gold contour.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda c: (
            c["area"]
            *
            (1.0 + c["circularity"])
        ),
        reverse=True
    )

    best = candidates[0]

    x, y, w, h = best["bbox"]

    # --------------------------------------------------------
    # Add a SMALL margin.
    # Do NOT include the whole selected ROI.
    # --------------------------------------------------------

    margin = 3

    x0 = max(
        0,
        x - margin
    )

    y0 = max(
        0,
        y - margin
    )

    x1 = min(
        frame.shape[1],
        x + w + margin
    )

    y1 = min(
        frame.shape[0],
        y + h + margin
    )

    tight_bbox = (
        x0,
        y0,
        x1 - x0,
        y1 - y0
    )

    return (
        tight_bbox,
        best["center"]
    )


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
            f"Frame: {frame_number}/{total-1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Time: {frame_number/fps:.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Horizon - Select Start Frame",
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
                "Horizon - Select Start Frame"
            )

            return (
                frame_number,
                frame
            )

        elif key == ord("q") or key == 27:

            cv2.destroyAllWindows()

            raise RuntimeError(
                "Cancelled."
            )


# ============================================================
# SELECT BOB
# ============================================================

def select_bob(
    frame
):

    print()
    print("=" * 70)
    print("SELECT THE REAL GOLD BOB")
    print("=" * 70)

    print()
    print(
        "Draw a rectangle around the GOLD BALL."
    )

    print(
        "It is okay if the rectangle is slightly "
        "larger than the ball."
    )

    print(
        "V7 will automatically isolate the gold "
        "object inside your rectangle."
    )

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

    return (
        x,
        y,
        w,
        h
    )


# ============================================================
# RECOVERY SEARCH
# ============================================================

def recover_bob(
    frame,
    predicted,
    expected_radius
):

    candidates = gold_candidates(
        frame
    )

    best = None
    best_score = float("inf")

    for candidate in candidates:

        point = candidate[
            "center"
        ]

        # ----------------------------------------------------
        # Distance from predicted position
        # ----------------------------------------------------

        prediction_error = np.linalg.norm(
            point - predicted
        )

        if prediction_error > SEARCH_RADIUS:
            continue

        # ----------------------------------------------------
        # Pendulum radius
        # ----------------------------------------------------

        candidate_radius = get_radius(
            point
        )

        radius_error = abs(
            candidate_radius
            -
            expected_radius
        )

        if radius_error > RADIUS_TOLERANCE:
            continue

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = (
            PREDICTION_WEIGHT
            *
            prediction_error
            +
            RADIUS_WEIGHT
            *
            radius_error
            -
            20.0
            *
            candidate["circularity"]
        )

        if score < best_score:

            best_score = score
            best = candidate

    return best


# ============================================================
# TRACK FORWARD
# ============================================================

def track_forward(
    cap,
    start_frame,
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
    # Initial tracker
    # --------------------------------------------------------

    tracker = create_csrt()

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    ret, frame = cap.read()

    if not ret:

        raise RuntimeError(
            "Could not read starting frame."
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

    previous = (
        initial_center.copy()
    )

    previous_previous = None

    velocity = np.zeros(
        2,
        dtype=float
    )

    # Save starting point
    positions[
        start_frame
    ] = previous.copy()

    angles[
        start_frame
    ] = get_theta(
        previous
    )

    statuses[
        start_frame
    ] = "INITIAL"

    # ========================================================
    # FRAME LOOP
    # ========================================================

    for frame_number in range(
        start_frame + 1,
        total
    ):

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # Motion prediction
        # ----------------------------------------------------

        predicted = (
            previous
            +
            velocity
        )

        # ----------------------------------------------------
        # CSRT
        # ----------------------------------------------------

        ok, bbox = tracker.update(
            frame
        )

        candidate = None

        if ok:

            bx, by, bw, bh = bbox

            csrt_point = np.array(
                [
                    bx + bw / 2,
                    by + bh / 2
                ],
                dtype=float
            )

            csrt_radius = get_radius(
                csrt_point
            )

            radius_error = abs(
                csrt_radius
                -
                expected_radius
            )

            movement = np.linalg.norm(
                csrt_point
                -
                previous
            )

            prediction_error = np.linalg.norm(
                csrt_point
                -
                predicted
            )

            # ------------------------------------------------
            # CSRT is valid ONLY if all constraints pass.
            # ------------------------------------------------

            if (
                radius_error
                <=
                RADIUS_TOLERANCE
                and
                movement
                <=
                MAX_STEP
                and
                prediction_error
                <=
                SEARCH_RADIUS
            ):

                candidate = csrt_point

                statuses[
                    frame_number
                ] = "CSRT"

        # ====================================================
        # RECOVERY
        # ====================================================

        if candidate is None:

            recovered = recover_bob(
                frame,
                predicted,
                expected_radius
            )

            if recovered is not None:

                candidate = recovered[
                    "center"
                ]

                rx, ry, rw, rh = (
                    recovered["bbox"]
                )

                tracker = create_csrt()

                tracker.init(
                    frame,
                    (
                        int(rx),
                        int(ry),
                        int(rw),
                        int(rh)
                    )
                )

                statuses[
                    frame_number
                ] = "RECOVERED"

            else:

                statuses[
                    frame_number
                ] = "MISSING"

        # ====================================================
        # ACCEPT
        # ====================================================

        if candidate is not None:

            if previous_previous is not None:

                raw_velocity = (
                    previous
                    -
                    previous_previous
                )

                # Smooth velocity
                velocity = (
                    0.7 * velocity
                    +
                    0.3 * raw_velocity
                )

            previous_previous = (
                previous.copy()
            )

            previous = (
                candidate.copy()
            )

            positions[
                frame_number
            ] = candidate.copy()

            angles[
                frame_number
            ] = get_theta(
                candidate
            )

        else:

            # ------------------------------------------------
            # DO NOT INVENT A POSITION.
            # ------------------------------------------------

            velocity *= 0.5

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
    print("V7 TRACKING PREVIEW")
    print("=" * 70)

    print(
        "Press Q to stop."
    )

    for frame_number in sorted(
        positions.keys()
    ):

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

        cv2.putText(
            display,
            "PIVOT",
            (
                px + 8,
                py
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

        # ----------------------------------------------------
        # Bob
        # ----------------------------------------------------

        cv2.circle(
            display,
            (bx, by),
            10,
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
        # String / geometry
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

        if status == "CSRT":

            colour = (
                0,
                255,
                0
            )

        elif status == "RECOVERED":

            colour = (
                0,
                255,
                255
            )

        else:

            colour = (
                0,
                0,
                255
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
            f"Time: {frame_number/fps:.2f}s",
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
            f"STATUS: {status}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2
        )

        cv2.imshow(
            "Horizon - V7 Pendulum Tracking",
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
    print("HORIZON — PENDULUM TRACKER V7")
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
        f"Duration   : {total/fps:.2f}s"
    )

    # ========================================================
    # START
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
        f"{start_frame/fps:.2f}s"
    )

    # ========================================================
    # USER ROI
    # ========================================================

    user_roi = select_bob(
        frame
    )

    # ========================================================
    # EXTRACT ACTUAL GOLD OBJECT
    # ========================================================

    tight_bbox, initial_center = (
        extract_bob_from_roi(
            frame,
            user_roi
        )
    )

    x, y, w, h = tight_bbox

    expected_radius = get_radius(
        initial_center
    )

    print()
    print("=" * 70)
    print("TIGHT BOB EXTRACTION")
    print("=" * 70)

    print(
        f"Tight bbox : "
        f"x={x}, y={y}, "
        f"w={w}, h={h}"
    )

    print(
        f"Bob centre : "
        f"({initial_center[0]:.1f}, "
        f"{initial_center[1]:.1f})"
    )

    print(
        f"Radius     : "
        f"{expected_radius:.1f}px"
    )

    # ========================================================
    # TRACK
    # ========================================================

    print()
    print("=" * 70)
    print("TRACKING FORWARD")
    print("=" * 70)

    positions, angles, statuses = (
        track_forward(
            cap,
            start_frame,
            total,
            fps,
            tight_bbox,
            initial_center,
            expected_radius
        )
    )

    # ========================================================
    # ARRAYS
    # ========================================================

    time = (
        np.arange(total)
        /
        fps
    )

    x_data = np.full(
        total,
        np.nan
    )

    y_data = np.full(
        total,
        np.nan
    )

    theta_data = np.full(
        total,
        np.nan
    )

    status_data = np.full(
        total,
        "",
        dtype="<U12"
    )

    for i in positions:

        x_data[i] = (
            positions[i][0]
        )

        y_data[i] = (
            positions[i][1]
        )

        theta_data[i] = (
            angles[i]
        )

        status_data[i] = (
            statuses.get(
                i,
                "UNKNOWN"
            )
        )

    valid = (
        np.isfinite(x_data)
        &
        np.isfinite(y_data)
    )

    detected = int(
        np.sum(valid)
    )

    missing = (
        total
        -
        detected
    )

    csrt_count = int(
        np.sum(
            status_data == "CSRT"
        )
    )

    recovery_count = int(
        np.sum(
            status_data == "RECOVERED"
        )
    )

    print()
    print("=" * 70)
    print("V7 TRACKING SUMMARY")
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
        f"{100*detected/total:.1f}%"
    )

    print(
        f"CSRT frames      : "
        f"{csrt_count}"
    )

    print(
        f"Recovery frames  : "
        f"{recovery_count}"
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
        x=x_data,
        y=y_data,
        theta=theta_data,
        pivot=PIVOT,
        pendulum_length=expected_radius,
        status=status_data
    )

    print()
    print(
        f"Saved: {OUTPUT_DATA}"
    )

    # ========================================================
    # ANGLE PLOT
    # ========================================================

    valid_theta = np.isfinite(
        theta_data
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time[valid_theta],
        np.degrees(
            theta_data[valid_theta]
        )
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Theta (degrees)"
    )

    plt.title(
        "Horizon — V7 Pendulum Angle θ(t)"
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
        x_data[valid],
        y_data[valid],
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
        "Horizon — V7 Pendulum Trajectory"
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
        statuses,
        fps,
        total
    )

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()