import cv2
import numpy as np
import matplotlib.pyplot as plt


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_NPZ = "horizon_pendulum_v10.npz"
OUTPUT_VIDEO = "horizon_pendulum_v10_tracking.mp4"


# ============================================================
# SETTINGS
# ============================================================

# Gold colour
GOLD_LOWER = np.array([5, 70, 35])
GOLD_UPPER = np.array([38, 255, 255])

# Morphology
KERNEL = np.ones((5, 5), np.uint8)

# Candidate size
MIN_AREA = 25
MAX_AREA = 4000

# Circularity
MIN_CIRCULARITY = 0.20

# Physical radius tolerance
RADIUS_TOLERANCE = 90

# Maximum frame-to-frame movement
MAX_MOVEMENT = 130

# Motion threshold
MOTION_THRESHOLD = 18

# Search expansion if normal search fails
RECOVERY_RADIUS = 180


# ============================================================
# GOLD MASK
# ============================================================

def get_gold_mask(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        GOLD_LOWER,
        GOLD_UPPER
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        KERNEL
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        KERNEL
    )

    return mask


# ============================================================
# MOTION MASK
# ============================================================

def get_motion_mask(
    previous,
    current
):

    previous_gray = cv2.cvtColor(
        previous,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY
    )

    diff = cv2.absdiff(
        previous_gray,
        current_gray
    )

    _, motion = cv2.threshold(
        diff,
        MOTION_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    motion = cv2.morphologyEx(
        motion,
        cv2.MORPH_OPEN,
        KERNEL
    )

    motion = cv2.dilate(
        motion,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    return motion


# ============================================================
# CANDIDATES
# ============================================================

def get_candidates(
    frame,
    previous_frame,
    expected,
    pivot,
    pendulum_length,
    search_radius
):

    gold = get_gold_mask(frame)

    if previous_frame is None:

        combined = gold

    else:

        motion = get_motion_mask(
            previous_frame,
            frame
        )

        # Gold AND motion
        combined = cv2.bitwise_and(
            gold,
            motion
        )

        # If this becomes too restrictive,
        # retain gold as fallback.
        if cv2.countNonZero(combined) < 10:

            combined = gold

    # --------------------------------------------------------
    # Restrict to neighbourhood around expected bob
    # --------------------------------------------------------

    h, w = combined.shape

    ex, ey = expected

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

    local = np.zeros_like(combined)

    local[
        y0:y1,
        x0:x1
    ] = combined[
        y0:y1,
        x0:x1
    ]

    contours, _ = cv2.findContours(
        local,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    px, py = pivot

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
            4 * np.pi * area
            /
            (perimeter ** 2)
        )

        if circularity < MIN_CIRCULARITY:
            continue

        (cx, cy), radius = (
            cv2.minEnclosingCircle(
                contour
            )
        )

        distance_from_pivot = np.hypot(
            cx - px,
            cy - py
        )

        radius_error = abs(
            distance_from_pivot
            -
            pendulum_length
        )

        if radius_error > RADIUS_TOLERANCE:
            continue

        movement = np.hypot(
            cx - ex,
            cy - ey
        )

        if movement > search_radius:
            continue

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = (
            movement * 2.0
            +
            radius_error * 1.5
            -
            circularity * 40
            -
            min(area, 1000) * 0.01
        )

        candidates.append(
            {
                "x": cx,
                "y": cy,
                "radius": radius,
                "area": area,
                "circularity": circularity,
                "distance_from_pivot": distance_from_pivot,
                "movement": movement,
                "score": score
            }
        )

    candidates.sort(
        key=lambda c: c["score"]
    )

    return candidates


# ============================================================
# INITIAL FRAME BROWSER
# ============================================================

def browse_frame(
    cap,
    start
):

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    current = start

    while True:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            current
        )

        ret, frame = cap.read()

        if not ret:
            return None

        display = frame.copy()

        cv2.putText(
            display,
            f"Frame: {current}/{total - 1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Time: {current / fps:.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "D/RIGHT next | A/LEFT previous | ENTER select",
            (10, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Horizon — Select Bob Frame",
            display
        )

        key = cv2.waitKey(0) & 0xFF

        if key in (ord("d"), 83):

            current = min(
                current + 1,
                total - 1
            )

        elif key in (ord("a"), 81):

            current = max(
                current - 1,
                0
            )

        elif key in (13, 32):

            cv2.destroyWindow(
                "Horizon — Select Bob Frame"
            )

            return current

        elif key in (ord("q"), 27):

            cv2.destroyAllWindows()

            return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HORIZON — MOTION + GOLD + GEOMETRY TRACKER V10")
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
    # SELECT FRAME
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT A GOOD CALIBRATION FRAME")
    print("=" * 70)

    print(
        "Prefer a frame where the bob is clearly away from the base."
    )

    start_frame = browse_frame(
        cap,
        100
    )

    if start_frame is None:
        return

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    ret, frame = cap.read()

    if not ret:

        raise RuntimeError(
            "Could not read calibration frame."
        )

    print(
        f"Calibration frame : {start_frame}"
    )

    print(
        f"Calibration time  : "
        f"{start_frame / fps:.2f}s"
    )

    # ========================================================
    # PIVOT
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT THE PIVOT")
    print("=" * 70)

    roi = cv2.selectROI(
        "Horizon — Select Pivot",
        frame,
        False,
        False
    )

    cv2.destroyWindow(
        "Horizon — Select Pivot"
    )

    px = (
        roi[0]
        +
        roi[2] / 2
    )

    py = (
        roi[1]
        +
        roi[3] / 2
    )

    pivot = (
        px,
        py
    )

    print(
        f"Pivot: ({px:.1f}, {py:.1f})"
    )

    # ========================================================
    # BOB
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT THE GOLD BOB")
    print("=" * 70)

    print(
        "Select ONLY the gold ball."
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

    bx = (
        roi[0]
        +
        roi[2] / 2
    )

    by = (
        roi[1]
        +
        roi[3] / 2
    )

    bob = (
        bx,
        by
    )

    pendulum_length = np.hypot(
        bx - px,
        by - py
    )

    print(
        f"Bob: ({bx:.1f}, {by:.1f})"
    )

    print(
        f"Pendulum length: "
        f"{pendulum_length:.1f}px"
    )

    # ========================================================
    # PROCESS
    # ========================================================

    print()
    print("=" * 70)
    print("PROCESSING ENTIRE VIDEO")
    print("=" * 70)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (
            width,
            height
        )
    )

    positions = np.full(
        (total, 2),
        np.nan
    )

    theta = np.full(
        total,
        np.nan
    )

    times = (
        np.arange(total)
        /
        fps
    )

    previous_frame = None
    previous_position = bob

    detected = 0
    recovery_count = 0

    for frame_number in range(total):

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # Predict from previous motion
        # ----------------------------------------------------

        if previous_position is not None:

            if frame_number >= 2:

                dx = (
                    previous_position[0]
                    -
                    positions[
                        frame_number - 2,
                        0
                    ]
                )

                dy = (
                    previous_position[1]
                    -
                    positions[
                        frame_number - 2,
                        1
                    ]
                )

                if np.isfinite(dx) and np.isfinite(dy):

                    expected = (
                        previous_position[0] + dx,
                        previous_position[1] + dy
                    )

                else:

                    expected = previous_position

            else:

                expected = previous_position

        else:

            expected = bob

        # ----------------------------------------------------
        # Normal search
        # ----------------------------------------------------

        candidates = get_candidates(
            frame,
            previous_frame,
            expected,
            pivot,
            pendulum_length,
            MAX_MOVEMENT
        )

        chosen = (
            candidates[0]
            if candidates
            else None
        )

        # ----------------------------------------------------
        # Recovery search
        # ----------------------------------------------------

        if chosen is None:

            candidates = get_candidates(
                frame,
                previous_frame,
                expected,
                pivot,
                pendulum_length,
                RECOVERY_RADIUS
            )

            if candidates:

                chosen = candidates[0]

                recovery_count += 1

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        if chosen is not None:

            bx = chosen["x"]
            by = chosen["y"]

            previous_position = (
                bx,
                by
            )

            positions[
                frame_number
            ] = (
                bx,
                by
            )

            # Angle measured from vertical
            angle = np.degrees(
                np.arctan2(
                    bx - px,
                    by - py
                )
            )

            theta[
                frame_number
            ] = angle

            detected += 1

            status = (
                "BOB VALID"
            )

        else:

            status = (
                "NO VALID BOB"
            )

        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

        display = frame.copy()

        # Pivot
        cv2.circle(
            display,
            (
                int(px),
                int(py)
            ),
            7,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            display,
            "PIVOT",
            (
                int(px + 10),
                int(py)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2
        )

        # Pendulum circle
        cv2.circle(
            display,
            (
                int(px),
                int(py)
            ),
            int(pendulum_length),
            (100, 100, 100),
            1
        )

        # Bob
        if chosen is not None:

            cv2.circle(
                display,
                (
                    int(bx),
                    int(by)
                ),
                10,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                display,
                (
                    int(bx),
                    int(by)
                ),
                17,
                (0, 255, 0),
                2
            )

            # Pivot -> bob
            cv2.line(
                display,
                (
                    int(px),
                    int(py)
                ),
                (
                    int(bx),
                    int(by)
                ),
                (0, 255, 255),
                2
            )

            # Candidate info
            cv2.putText(
                display,
                f"Area: {chosen['area']:.0f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Circularity: "
                f"{chosen['circularity']:.2f}",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Pivot radius: "
                f"{chosen['distance_from_pivot']:.1f}",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # Status
        colour = (
            (0, 255, 0)
            if chosen is not None
            else
            (0, 0, 255)
        )

        cv2.putText(
            display,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2
        )

        cv2.putText(
            display,
            f"Frame: {frame_number}/{total - 1}",
            (
                10,
                height - 45
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
                height - 20
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
            "Horizon — V10 Tracking",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        previous_frame = frame.copy()

        if frame_number % 30 == 0:

            print(
                f"Frame "
                f"{frame_number:4d}/"
                f"{total - 1} | "
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
        theta=theta,
        pivot=np.array(pivot),
        pendulum_length=pendulum_length,
        fps=fps
    )

    valid = np.isfinite(theta)

    valid_count = int(
        np.sum(valid)
    )

    print()
    print("=" * 70)
    print("V10 TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Total frames     : {total}"
    )

    print(
        f"Detected frames  : {valid_count}"
    )

    print(
        f"Missing frames   : "
        f"{total - valid_count}"
    )

    print(
        f"Detection rate   : "
        f"{100 * valid_count / total:.1f}%"
    )

    print(
        f"Recovery frames  : "
        f"{recovery_count}"
    )

    print(
        f"Pendulum length  : "
        f"{pendulum_length:.1f}px"
    )

    print()
    print(
        f"Saved: {OUTPUT_NPZ}"
    )

    print(
        f"Saved: {OUTPUT_VIDEO}"
    )

    # ========================================================
    # ANGLE PLOT
    # ========================================================

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
        "Horizon — V10 Pendulum Angle θ(t)"
    )

    plt.grid(True)
    plt.legend()

    plt.show()

    # ========================================================
    # TRAJECTORY
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    plt.plot(
        positions[:, 0],
        positions[:, 1],
        label="Tracked bob"
    )

    plt.scatter(
        px,
        py,
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
        "Horizon — V10 Pendulum Trajectory"
    )

    plt.gca().invert_yaxis()

    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()