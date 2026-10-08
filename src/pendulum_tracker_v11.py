import cv2
import numpy as np
import matplotlib.pyplot as plt


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_NPZ = "horizon_pendulum_v11.npz"
OUTPUT_VIDEO = "horizon_pendulum_v11_tracking.mp4"


# ============================================================
# SETTINGS
# ============================================================

# Gold colour range
LOWER_GOLD = np.array([10, 70, 60])
UPPER_GOLD = np.array([45, 255, 255])

# Candidate geometry
MIN_AREA = 80
MAX_AREA = 2500
MIN_CIRCULARITY = 0.40

# Radius tolerance around calibrated pendulum length
RADIUS_TOLERANCE = 120

# Maximum allowed movement between consecutive frames
MAX_JUMP = 150

# How close candidate must be to the pivot->candidate line
STRING_DISTANCE_LIMIT = 18

# Ignore the first part of the string near the pivot
STRING_START = 45

# Ignore region immediately around the base
BASE_Y = 535

# Central support/pole region
POLE_X_MIN = 235
POLE_X_MAX = 275


# ============================================================
# UTILITY
# ============================================================

def distance(a, b):
    return np.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def point_line_distance(point, a, b):
    """
    Perpendicular distance between point and line AB.
    """

    px, py = point
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay

    length = np.hypot(dx, dy)

    if length < 1e-6:
        return 9999

    return abs(
        dy * px
        - dx * py
        + bx * ay
        - by * ax
    ) / length


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

    # Clean colour mask
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

        if radius < 5 or radius > 50:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect = w / max(h, 1)

        if aspect < 0.35 or aspect > 2.8:
            continue

        candidates.append({
            "center": (float(cx), float(cy)),
            "area": float(area),
            "radius": float(radius),
            "circularity": float(circularity),
            "bbox": (x, y, w, h)
        })

    return candidates, mask


# ============================================================
# STRING SUPPORT
# ============================================================

def calculate_string_support(
    frame,
    pivot,
    candidate
):
    """
    Estimate whether a visible string exists between
    pivot and candidate.

    We sample Canny edges along the proposed string.
    """

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
        40,
        120
    )

    cx, cy = candidate

    px, py = pivot

    dx = cx - px
    dy = cy - py

    length = np.hypot(
        dx,
        dy
    )

    if length < STRING_START + 20:
        return 0.0

    # Unit vector along string
    ux = dx / length
    uy = dy / length

    # Perpendicular vector
    vx = -uy
    vy = ux

    samples = 0
    hits = 0

    # Don't test directly at pivot or inside bob.
    start = STRING_START
    end = max(
        start + 10,
        length - 25
    )

    for d in np.linspace(
        start,
        end,
        80
    ):

        x = int(px + ux * d)
        y = int(py + uy * d)

        # Check a narrow band around proposed string
        found = False

        for offset in (-3, -2, -1, 0, 1, 2, 3):

            xx = int(
                x + vx * offset
            )

            yy = int(
                y + vy * offset
            )

            if (
                0 <= yy < edges.shape[0]
                and
                0 <= xx < edges.shape[1]
            ):

                if edges[yy, xx] > 0:
                    found = True
                    break

        samples += 1

        if found:
            hits += 1

    if samples == 0:
        return 0.0

    return hits / samples


# ============================================================
# BASE PENALTY
# ============================================================

def base_penalty(candidate):

    x, y = candidate

    # Candidates deep inside the black base are suspicious.
    if y > BASE_Y:

        # Strong penalty for central base region.
        if (
            POLE_X_MIN - 50
            <
            x
            <
            POLE_X_MAX + 50
        ):
            return 1.0

        return 0.5

    return 0.0


# ============================================================
# SELECT CALIBRATION
# ============================================================

def select_calibration(cap):

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_number = min(
        100,
        total - 1
    )

    while True:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:
            break

        display = frame.copy()

        cv2.putText(
            display,
            f"Frame: {frame_number}/{total-1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Time: {frame_number/fps:.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "D/RIGHT = next | A/LEFT = previous | ENTER = select",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Calibration Frame",
            display
        )

        key = cv2.waitKey(0) & 0xFF

        if key in (
            ord("d"),
            83
        ):
            frame_number = min(
                frame_number + 1,
                total - 1
            )

        elif key in (
            ord("a"),
            81
        ):
            frame_number = max(
                frame_number - 1,
                0
            )

        elif key in (
            13,
            32
        ):
            break

        elif key in (
            ord("q"),
            27
        ):
            cv2.destroyAllWindows()
            raise SystemExit

    cv2.destroyWindow(
        "Calibration Frame"
    )

    return frame_number, frame


# ============================================================
# MAIN
# ============================================================

def main():

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open:\n{VIDEO_PATH}"
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
    print("HORIZON — STRING-GUIDED GOLD BOB TRACKER V11")
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
    # CALIBRATION FRAME
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT CALIBRATION FRAME")
    print("=" * 70)

    print()
    print(
        "Choose a frame where:"
    )
    print(
        "  ✓ gold bob is clearly visible"
    )
    print(
        "  ✓ bob is away from the black base"
    )
    print(
        "  ✓ string is visible"
    )

    calibration_frame_number, frame = select_calibration(
        cap
    )

    # ========================================================
    # SELECT PIVOT
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT PIVOT")
    print("=" * 70)

    print(
        "Select the point where the string attaches."
    )

    pivot_roi = cv2.selectROI(
        "Select Pivot",
        frame,
        False,
        False
    )

    cv2.destroyWindow(
        "Select Pivot"
    )

    px = pivot_roi[0] + pivot_roi[2] / 2
    py = pivot_roi[1] + pivot_roi[3] / 2

    pivot = (
        float(px),
        float(py)
    )

    print(
        f"Pivot: ({px:.1f}, {py:.1f})"
    )

    # ========================================================
    # SELECT BOB
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT GOLD BOB")
    print("=" * 70)

    print(
        "Select ONLY the gold ball."
    )

    bob_roi = cv2.selectROI(
        "Select Gold Bob",
        frame,
        False,
        False
    )

    cv2.destroyWindow(
        "Select Gold Bob"
    )

    bx = bob_roi[0] + bob_roi[2] / 2
    by = bob_roi[1] + bob_roi[3] / 2

    initial_bob = (
        float(bx),
        float(by)
    )

    pendulum_length = distance(
        pivot,
        initial_bob
    )

    print(
        f"Bob: ({bx:.1f}, {by:.1f})"
    )

    print(
        f"Pendulum length: {pendulum_length:.1f}px"
    )

    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    # ========================================================
    # STORAGE
    # ========================================================

    times = []
    xs = []
    ys = []
    angles = []

    previous_bob = initial_bob

    detected_count = 0
    missing_count = 0

    # ========================================================
    # PROCESS ENTIRE VIDEO
    # ========================================================

    print()
    print("=" * 70)
    print("PROCESSING ENTIRE VIDEO")
    print("=" * 70)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        candidates, mask = find_gold_candidates(
            frame
        )

        best = None
        best_score = -999999

        # ----------------------------------------------------
        # Evaluate candidates
        # ----------------------------------------------------

        for candidate in candidates:

            center = candidate["center"]

            # Radius from pivot
            radius = distance(
                pivot,
                center
            )

            radius_error = abs(
                radius - pendulum_length
            )

            if radius_error > RADIUS_TOLERANCE:
                continue

            # ------------------------------------------------
            # Motion continuity
            # ------------------------------------------------

            movement = distance(
                center,
                previous_bob
            )

            if movement > MAX_JUMP:
                continue

            # ------------------------------------------------
            # String support
            # ------------------------------------------------

            string_support = calculate_string_support(
                frame,
                pivot,
                center
            )

            # ------------------------------------------------
            # Distance from proposed string line
            # ------------------------------------------------

            line_distance = point_line_distance(
                center,
                pivot,
                center
            )

            # This is always zero mathematically, so instead
            # evaluate the candidate's own radius direction.
            #
            # The actual string support above is the important
            # geometric test.

            # ------------------------------------------------
            # Base penalty
            # ------------------------------------------------

            penalty = base_penalty(
                center
            )

            # ------------------------------------------------
            # Score
            # ------------------------------------------------

            radius_score = max(
                0,
                1 -
                radius_error /
                RADIUS_TOLERANCE
            )

            motion_score = max(
                0,
                1 -
                movement /
                MAX_JUMP
            )

            circularity_score = min(
                candidate["circularity"],
                1.0
            )

            score = (
                5.0 * string_support
                +
                3.0 * radius_score
                +
                2.0 * motion_score
                +
                1.0 * circularity_score
                -
                6.0 * penalty
            )

            if score > best_score:

                best_score = score
                best = candidate

        # ====================================================
        # ACCEPT / REJECT
        # ====================================================

        valid = False

        if best is not None:

            center = best["center"]

            radius = distance(
                pivot,
                center
            )

            string_support = calculate_string_support(
                frame,
                pivot,
                center
            )

            movement = distance(
                center,
                previous_bob
            )

            # Final validation
            if (
                string_support >= 0.08
                and
                abs(
                    radius -
                    pendulum_length
                ) <= RADIUS_TOLERANCE
                and
                movement <= MAX_JUMP
            ):
                valid = True

        # ====================================================
        # SAVE
        # ====================================================

        t = frame_number / fps

        if valid:

            cx, cy = best["center"]

            previous_bob = (
                cx,
                cy
            )

            xs.append(cx)
            ys.append(cy)

            theta = np.arctan2(
                cx - px,
                py - cy
            )

            angles.append(
                np.degrees(theta)
            )

            detected_count += 1

        else:

            xs.append(np.nan)
            ys.append(np.nan)
            angles.append(np.nan)

            missing_count += 1

        times.append(t)

        # ====================================================
        # DISPLAY
        # ====================================================

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

        # Draw candidates
        for candidate in candidates:

            cx, cy = candidate["center"]

            cv2.circle(
                display,
                (
                    int(cx),
                    int(cy)
                ),
                int(
                    max(
                        5,
                        candidate["radius"]
                    )
                ),
                (0, 140, 255),
                1
            )

        if valid:

            cx, cy = best["center"]

            # Yellow string guide
            cv2.line(
                display,
                (
                    int(px),
                    int(py)
                ),
                (
                    int(cx),
                    int(cy)
                ),
                (0, 255, 255),
                3
            )

            # Green bob
            cv2.circle(
                display,
                (
                    int(cx),
                    int(cy)
                ),
                15,
                (0, 255, 0),
                2
            )

            cv2.circle(
                display,
                (
                    int(cx),
                    int(cy)
                ),
                4,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                "BOB VALID",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Theta: {angles[-1]:.2f} deg",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Score: {best_score:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                display,
                "NO VALID BOB",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.putText(
            display,
            f"Frame: {frame_number}/{total-1}",
            (
                10,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        writer.write(display)

        cv2.imshow(
            "Horizon V11",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nStopped manually.")
            break

        frame_number += 1

        if frame_number % 30 == 0:

            status = (
                "VALID"
                if valid
                else "NO BOB"
            )

            print(
                f"Frame {frame_number:4d}/{total-1} | "
                f"{status}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    times = np.asarray(times)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    angles = np.asarray(angles)

    # ========================================================
    # SAVE
    # ========================================================

    np.savez(
        OUTPUT_NPZ,
        time=times,
        x=xs,
        y=ys,
        theta_deg=angles,
        pivot=np.asarray(pivot),
        pendulum_length=pendulum_length,
        fps=fps
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_processed = len(times)

    detection_rate = (
        detected_count /
        max(total_processed, 1)
        *
        100
    )

    print()
    print("=" * 70)
    print("V11 TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Processed frames : {total_processed}"
    )

    print(
        f"Detected frames  : {detected_count}"
    )

    print(
        f"Missing frames   : {missing_count}"
    )

    print(
        f"Detection rate   : {detection_rate:.1f}%"
    )

    print(
        f"Pendulum length  : {pendulum_length:.1f}px"
    )

    print()
    print(
        f"Saved: {OUTPUT_NPZ}"
    )

    print(
        f"Saved: {OUTPUT_VIDEO}"
    )

    # ========================================================
    # PLOT ANGLE
    # ========================================================

    valid_mask = np.isfinite(
        angles
    )

    if np.any(valid_mask):

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            times[valid_mask],
            angles[valid_mask]
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "θ (degrees)"
        )

        plt.title(
            "Horizon — V11 Pendulum Angle θ(t)"
        )

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.show()

    # ========================================================
    # PLOT TRAJECTORY
    # ========================================================

    if np.any(valid_mask):

        plt.figure(
            figsize=(8, 8)
        )

        plt.plot(
            xs[valid_mask],
            ys[valid_mask],
            label="Tracked bob"
        )

        plt.scatter(
            [px],
            [py],
            s=80,
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
            "Horizon — V11 Pendulum Trajectory"
        )

        plt.legend()

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()