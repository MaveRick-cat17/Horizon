import cv2
import numpy as np
import matplotlib.pyplot as plt


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_NPZ = "horizon_pendulum_v8.npz"
OUTPUT_VIDEO = "horizon_pendulum_v8_tracking.mp4"


# ============================================================
# SETTINGS
# ============================================================

# Gold colour range.
# We deliberately use a fairly broad range because the bob
# changes brightness while moving.
GOLD_LOWER = np.array([5, 60, 35])
GOLD_UPPER = np.array([38, 255, 255])

# Physical constraints
RADIUS_TOLERANCE = 75
MIN_RADIUS = 8
MAX_RADIUS = 45

# Candidate quality
MIN_AREA = 40
MAX_AREA = 5000
MIN_CIRCULARITY = 0.25

# Search radius around predicted bob position
SEARCH_RADIUS = 100

# Maximum allowed frame-to-frame movement
MAX_STEP = 100


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
        GOLD_LOWER,
        GOLD_UPPER
    )

    # Clean small noise
    kernel = np.ones((3, 3), np.uint8)

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

def find_gold_candidates(
    frame,
    expected,
    search_radius,
    expected_radius
):

    mask = gold_mask(frame)

    h, w = mask.shape

    ex, ey = expected

    # Local search window
    x0 = max(
        0,
        int(ex - search_radius)
    )

    x1 = min(
        w,
        int(ex + search_radius)
    )

    y0 = max(
        0,
        int(ey - search_radius)
    )

    y1 = min(
        h,
        int(ey + search_radius)
    )

    local = np.zeros_like(mask)

    local[y0:y1, x0:x1] = mask[
        y0:y1,
        x0:x1
    ]

    contours, _ = cv2.findContours(
        local,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

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
            4 * np.pi * area
            /
            (perimeter * perimeter)
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

        distance = np.hypot(
            cx - ex,
            cy - ey
        )

        if distance > search_radius:
            continue

        # Radius consistency
        radius_error = abs(
            radius - expected_radius
        )

        # Score:
        # closer + more circular + correct radius = better
        score = (
            distance
            + radius_error * 2.0
            - circularity * 35
        )

        candidates.append(
            {
                "x": cx,
                "y": cy,
                "radius": radius,
                "area": area,
                "circularity": circularity,
                "distance": distance,
                "score": score
            }
        )

    candidates.sort(
        key=lambda c: c["score"]
    )

    return candidates


# ============================================================
# BEST CANDIDATE
# ============================================================

def choose_candidate(
    candidates,
    pivot,
    expected,
    pendulum_radius
):

    if not candidates:
        return None

    px, py = pivot

    ex, ey = expected

    best = None
    best_score = float("inf")

    for c in candidates:

        cx = c["x"]
        cy = c["y"]

        # ----------------------------------------------------
        # Physical distance from pivot
        # ----------------------------------------------------

        actual_radius = np.hypot(
            cx - px,
            cy - py
        )

        radius_error = abs(
            actual_radius
            -
            pendulum_radius
        )

        if radius_error > RADIUS_TOLERANCE:
            continue

        # ----------------------------------------------------
        # Distance from prediction
        # ----------------------------------------------------

        prediction_error = np.hypot(
            cx - ex,
            cy - ey
        )

        # ----------------------------------------------------
        # Movement constraint
        # ----------------------------------------------------

        movement = prediction_error

        if movement > MAX_STEP:
            continue

        score = (
            prediction_error * 1.5
            +
            radius_error * 1.2
            +
            abs(
                c["radius"]
                -
                18
            )
            * 0.5
            -
            c["circularity"] * 30
        )

        if score < best_score:

            best_score = score
            best = c

    return best


# ============================================================
# PREDICT NEXT POSITION
# ============================================================

def predict_position(
    pivot,
    current,
    previous
):

    px, py = pivot

    cx, cy = current

    if previous is None:

        return current

    vx = cx - previous[0]
    vy = cy - previous[1]

    predicted = (
        cx + vx,
        cy + vy
    )

    # Keep prediction close to physical pendulum circle
    dx = predicted[0] - px
    dy = predicted[1] - py

    distance = np.hypot(
        dx,
        dy
    )

    if distance < 1:

        return current

    scale = (
        np.hypot(
            current[0] - px,
            current[1] - py
        )
        /
        distance
    )

    predicted = (
        px + dx * scale,
        py + dy * scale
    )

    return predicted


# ============================================================
# DRAW
# ============================================================

def draw_frame(
    frame,
    pivot,
    bob,
    trajectory,
    status,
    radius
):

    display = frame.copy()

    # Pivot
    cv2.circle(
        display,
        (
            int(pivot[0]),
            int(pivot[1])
        ),
        7,
        (255, 0, 255),
        -1
    )

    cv2.putText(
        display,
        "PIVOT",
        (
            int(pivot[0] + 10),
            int(pivot[1])
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2
    )

    # Physical pendulum circle
    cv2.circle(
        display,
        (
            int(pivot[0]),
            int(pivot[1])
        ),
        int(radius),
        (100, 100, 100),
        1
    )

    # Trajectory
    if len(trajectory) > 1:

        pts = np.array(
            trajectory,
            dtype=np.int32
        )

        cv2.polylines(
            display,
            [pts],
            False,
            (255, 0, 0),
            2
        )

    # Bob
    if bob is not None:

        bx, by = bob

        cv2.circle(
            display,
            (
                int(bx),
                int(by)
            ),
            7,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            display,
            (
                int(bx),
                int(by)
            ),
            14,
            (0, 255, 0),
            2
        )

        # String
        cv2.line(
            display,
            (
                int(pivot[0]),
                int(pivot[1])
            ),
            (
                int(bx),
                int(by)
            ),
            (0, 255, 255),
            2
        )

    cv2.putText(
        display,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return display


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HORIZON — PENDULUM TRACKER V8")
    print("=" * 70)

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
    # SELECT PIVOT
    # ========================================================

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        40
    )

    ret, frame = cap.read()

    if not ret:

        raise RuntimeError(
            "Could not read frame."
        )

    print()
    print("=" * 70)
    print("SELECT THE PIVOT")
    print("=" * 70)

    print(
        "Select the point where the STRING ATTACHES at the TOP."
    )

    roi = cv2.selectROI(
        "Horizon — Select Pivot",
        frame,
        False,
        False
    )

    cv2.destroyWindow(
        "Horizon — Select Pivot"
    )

    px = roi[0] + roi[2] / 2
    py = roi[1] + roi[3] / 2

    pivot = (
        px,
        py
    )

    print(
        f"Pivot: ({px:.1f}, {py:.1f})"
    )

    # ========================================================
    # SELECT BOB
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT THE GOLD BOB")
    print("=" * 70)

    print(
        "Select ONLY the circular gold ball."
    )

    roi = cv2.selectROI(
        "Horizon — Select Gold Bob",
        frame,
        False,
        False
    )

    cv2.destroyWindow(
        "Horizon — Select Gold Bob"
    )

    bx = roi[0] + roi[2] / 2
    by = roi[1] + roi[3] / 2

    bob = (
        bx,
        by
    )

    # Initial pendulum radius
    pendulum_radius = np.hypot(
        bx - px,
        by - py
    )

    # Estimate bob radius
    bob_radius = (
        roi[2] + roi[3]
    ) / 4

    bob_radius = np.clip(
        bob_radius,
        MIN_RADIUS,
        MAX_RADIUS
    )

    print(
        f"Bob centre : ({bx:.1f}, {by:.1f})"
    )

    print(
        f"Pendulum radius : "
        f"{pendulum_radius:.1f}px"
    )

    print(
        f"Bob radius : "
        f"{bob_radius:.1f}px"
    )

    # ========================================================
    # PROCESS ENTIRE VIDEO
    # ========================================================

    print()
    print("=" * 70)
    print("TRACKING ENTIRE VIDEO")
    print("=" * 70)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    positions = np.full(
        (total, 2),
        np.nan,
        dtype=np.float64
    )

    times = (
        np.arange(total)
        /
        fps
    )

    trajectory = []

    previous = None
    current = bob

    detected_count = 0

    # Keep track of valid angular position
    previous_angle = None

    for frame_number in range(total):

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predicted = predict_position(
            pivot,
            current,
            previous
        )

        # ----------------------------------------------------
        # Search gold candidates
        # ----------------------------------------------------

        candidates = find_gold_candidates(
            frame,
            predicted,
            SEARCH_RADIUS,
            bob_radius
        )

        chosen = choose_candidate(
            candidates,
            pivot,
            predicted,
            pendulum_radius
        )

        # ----------------------------------------------------
        # Accept detection
        # ----------------------------------------------------

        if chosen is not None:

            current = (
                chosen["x"],
                chosen["y"]
            )

            positions[
                frame_number
            ] = current

            trajectory.append(
                current
            )

            previous = current

            detected_count += 1

            status = "GOLD BOB: VALID"

        else:

            # Don't invent data.
            positions[
                frame_number
            ] = np.nan

            status = "NO VALID BOB"

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        display = draw_frame(
            frame,
            pivot,
            current if chosen is not None else None,
            trajectory,
            status,
            pendulum_radius
        )

        cv2.putText(
            display,
            f"Frame: {frame_number}/{total - 1}",
            (
                10,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Time: {times[frame_number]:.2f}s",
            (
                10,
                height - 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        writer.write(
            display
        )

        cv2.imshow(
            "Horizon — V8 Pendulum Tracking",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print(
                "\nStopped manually."
            )

            break

        # Progress
        if frame_number % 30 == 0:

            print(
                f"Frame "
                f"{frame_number:4d}/"
                f"{total - 1} "
                f"| "
                f"{status}"
            )

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    # ========================================================
    # SAVE
    # ========================================================

    np.savez(
        OUTPUT_NPZ,
        time=times,
        x=positions[:, 0],
        y=positions[:, 1],
        pivot=np.array(pivot),
        pendulum_radius=pendulum_radius,
        fps=fps
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    valid = np.isfinite(
        positions[:, 0]
    )

    detected = np.sum(
        valid
    )

    missing = total - detected

    print()
    print("=" * 70)
    print("V8 TRACKING SUMMARY")
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
        f"Detection rate   : "
        f"{100 * detected / total:.1f}%"
    )

    print(
        f"Pendulum radius  : "
        f"{pendulum_radius:.1f}px"
    )

    print()
    print(
        f"Saved trajectory : {OUTPUT_NPZ}"
    )

    print(
        f"Saved video      : {OUTPUT_VIDEO}"
    )

    # ========================================================
    # PLOT TRAJECTORY
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        positions[:, 0],
        positions[:, 1],
        label="Tracked bob"
    )

    plt.scatter(
        pivot[0],
        pivot[1],
        s=100,
        label="Pivot"
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "X position (pixels)"
    )

    plt.ylabel(
        "Y position (pixels)"
    )

    plt.title(
        "Horizon — V8 Pendulum Trajectory"
    )

    plt.legend()
    plt.grid(True)

    plt.show()

    # ========================================================
    # ANGLE
    # ========================================================

    valid_x = positions[:, 0]
    valid_y = positions[:, 1]

    theta = np.full(
        total,
        np.nan
    )

    valid = (
        np.isfinite(valid_x)
        &
        np.isfinite(valid_y)
    )

    dx = (
        valid_x[valid]
        -
        pivot[0]
    )

    dy = (
        valid_y[valid]
        -
        pivot[1]
    )

    theta[valid] = np.degrees(
        np.arctan2(
            dx,
            dy
        )
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        times,
        theta,
        label="θ(t)"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "θ (degrees)"
    )

    plt.title(
        "Horizon — V8 Pendulum Angle θ(t)"
    )

    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()