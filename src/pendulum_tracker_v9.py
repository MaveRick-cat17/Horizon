import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON V9
# STRING-GUIDED PENDULUM TRACKER
# ============================================================

VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_NPZ = "horizon_pendulum_v9.npz"
OUTPUT_VIDEO = "horizon_pendulum_v9_tracking.mp4"


# ============================================================
# STRING DETECTION SETTINGS
# ============================================================

CANNY_LOW = 30
CANNY_HIGH = 100

HOUGH_THRESHOLD = 25
MIN_LINE_LENGTH = 70
MAX_LINE_GAP = 35

# Maximum physically reasonable angle
MAX_ANGLE_DEG = 40

# How close a detected line must pass to the pivot
PIVOT_TOLERANCE = 35

# Temporal continuity
MAX_ANGLE_CHANGE = 8.0


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_edges(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH
    )

    return edges


# ============================================================
# STRING CANDIDATES
# ============================================================

def detect_string_lines(
    frame,
    pivot,
    previous_angle=None
):

    edges = prepare_edges(frame)

    h, w = edges.shape

    px, py = pivot

    # --------------------------------------------------------
    # Only examine region below pivot
    # --------------------------------------------------------

    roi = np.zeros_like(edges)

    top = max(
        0,
        int(py)
    )

    bottom = min(
        h,
        int(py + 620)
    )

    roi[
        top:bottom,
        :
    ] = edges[
        top:bottom,
        :
    ]

    # --------------------------------------------------------
    # Hough line segments
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    if lines is None:

        return None, edges, []

    candidates = []

    for line in lines:

        # OpenCV sometimes returns shape (1, 4)
        values = np.asarray(
            line
        ).reshape(-1)

        if len(values) < 4:
            continue

        x1, y1, x2, y2 = (
            map(
                float,
                values[:4]
            )
        )

        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(
            dx,
            dy
        )

        if length < MIN_LINE_LENGTH:
            continue

        # ----------------------------------------------------
        # We want a line that generally points downward
        # from the pivot.
        # ----------------------------------------------------

        # Make endpoint 1 the upper endpoint
        if y1 > y2:

            x1, x2 = x2, x1
            y1, y2 = y2, y1

            dx = x2 - x1
            dy = y2 - y1

        if dy <= 0:
            continue

        # ----------------------------------------------------
        # Angle from vertical.
        #
        # Positive = right
        # Negative = left
        # ----------------------------------------------------

        angle = np.degrees(
            np.arctan2(
                dx,
                dy
            )
        )

        if abs(angle) > MAX_ANGLE_DEG:
            continue

        # ----------------------------------------------------
        # Extrapolate line upward to pivot Y.
        # ----------------------------------------------------

        if abs(dy) < 1:
            continue

        x_at_pivot = (
            x1
            +
            (py - y1)
            *
            dx
            /
            dy
        )

        pivot_distance = abs(
            x_at_pivot - px
        )

        if pivot_distance > PIVOT_TOLERANCE:
            continue

        # ----------------------------------------------------
        # How far below pivot does the segment extend?
        # ----------------------------------------------------

        lower_y = max(
            y1,
            y2
        )

        vertical_coverage = (
            lower_y - py
        )

        if vertical_coverage < 50:
            continue

        # ----------------------------------------------------
        # Temporal consistency.
        # ----------------------------------------------------

        angle_difference = 0

        if previous_angle is not None:

            angle_difference = abs(
                angle
                -
                previous_angle
            )

            # Don't immediately reject everything here.
            # Give continuity a score instead.
            if angle_difference > 20:
                continuity_penalty = 200
            else:
                continuity_penalty = (
                    angle_difference * 8
                )

        else:

            continuity_penalty = 0

        # ----------------------------------------------------
        # Score candidate.
        #
        # Long string
        # close to pivot
        # small angle jump
        # ----------------------------------------------------

        score = (
            pivot_distance * 8
            +
            continuity_penalty
            -
            length * 0.5
            -
            vertical_coverage * 0.25
        )

        candidates.append(
            {
                "line": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ),
                "angle": angle,
                "length": length,
                "pivot_distance": pivot_distance,
                "vertical_coverage": vertical_coverage,
                "score": score
            }
        )

    if not candidates:

        return None, edges, []

    candidates.sort(
        key=lambda c: c["score"]
    )

    return (
        candidates[0],
        edges,
        candidates
    )


# ============================================================
# MANUAL FRAME SELECTION
# ============================================================

def select_start_frame(
    cap,
    start_frame
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    current = start_frame

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

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
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Time: {current / cap.get(cv2.CAP_PROP_FPS):.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "D/RIGHT = next | A/LEFT = previous | ENTER = select",
            (10, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Horizon — Select Starting Frame",
            display
        )

        key = cv2.waitKey(0) & 0xFF

        if key in (
            ord("d"),
            83
        ):

            current = min(
                current + 1,
                total - 1
            )

        elif key in (
            ord("a"),
            81
        ):

            current = max(
                current - 1,
                0
            )

        elif key in (
            13,
            32
        ):

            cv2.destroyWindow(
                "Horizon — Select Starting Frame"
            )

            return current

        elif key in (
            ord("q"),
            27
        ):

            cv2.destroyAllWindows()

            return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HORIZON — STRING-GUIDED PENDULUM TRACKER V9")
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
    # SELECT START FRAME
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT A GOOD STARTING FRAME")
    print("=" * 70)

    start_frame = select_start_frame(
        cap,
        40
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
            "Could not read selected frame."
        )

    print(
        f"Starting frame : {start_frame}"
    )

    print(
        f"Starting time  : "
        f"{start_frame / fps:.2f}s"
    )

    # ========================================================
    # SELECT PIVOT
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT THE PIVOT")
    print("=" * 70)

    print(
        "Select the exact point where the string attaches."
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
    # SELECT BOB ONLY TO CALIBRATE LENGTH
    # ========================================================

    print()
    print("=" * 70)
    print("SELECT THE GOLD BOB")
    print("=" * 70)

    print(
        "This selection is ONLY used to measure pendulum length."
    )

    roi = cv2.selectROI(
        "Horizon — Select Bob",
        frame,
        False,
        False
    )

    cv2.destroyWindow(
        "Horizon — Select Bob"
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

    pendulum_length = np.hypot(
        bx - px,
        by - py
    )

    print(
        f"Bob centre: "
        f"({bx:.1f}, {by:.1f})"
    )

    print(
        f"Pendulum length: "
        f"{pendulum_length:.1f}px"
    )

    # ========================================================
    # PROCESS ENTIRE VIDEO
    # ========================================================

    print()
    print("=" * 70)
    print("STRING TRACKING")
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

    theta = np.full(
        total,
        np.nan,
        dtype=np.float64
    )

    x = np.full(
        total,
        np.nan,
        dtype=np.float64
    )

    y = np.full(
        total,
        np.nan,
        dtype=np.float64
    )

    times = (
        np.arange(total)
        /
        fps
    )

    previous_angle = None

    detected = 0

    for frame_number in range(total):

        ret, frame = cap.read()

        if not ret:
            break

        result, edges, candidates = detect_string_lines(
            frame,
            pivot,
            previous_angle
        )

        display = frame.copy()

        # ----------------------------------------------------
        # Pivot
        # ----------------------------------------------------

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
            0.5,
            (255, 0, 255),
            2
        )

        # ----------------------------------------------------
        # Draw selected string
        # ----------------------------------------------------

        if result is not None:

            angle = result["angle"]

            # ------------------------------------------------
            # Reject huge sudden angle jumps
            # ------------------------------------------------

            valid = True

            if previous_angle is not None:

                if abs(
                    angle - previous_angle
                ) > MAX_ANGLE_CHANGE:

                    valid = False

            if valid:

                theta[
                    frame_number
                ] = angle

                previous_angle = angle

                # --------------------------------------------
                # Reconstruct bob from geometry
                # --------------------------------------------

                angle_rad = np.radians(
                    angle
                )

                bob_x = (
                    px
                    +
                    pendulum_length
                    *
                    np.sin(angle_rad)
                )

                bob_y = (
                    py
                    +
                    pendulum_length
                    *
                    np.cos(angle_rad)
                )

                x[
                    frame_number
                ] = bob_x

                y[
                    frame_number
                ] = bob_y

                detected += 1

                # --------------------------------------------
                # Draw string direction
                # --------------------------------------------

                cv2.line(
                    display,
                    (
                        int(px),
                        int(py)
                    ),
                    (
                        int(bob_x),
                        int(bob_y)
                    ),
                    (0, 255, 255),
                    3
                )

                # --------------------------------------------
                # Draw calculated bob
                # --------------------------------------------

                cv2.circle(
                    display,
                    (
                        int(bob_x),
                        int(bob_y)
                    ),
                    10,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    display,
                    (
                        int(bob_x),
                        int(bob_y)
                    ),
                    15,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"STRING VALID | "
                    f"theta={angle:.2f} deg",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    display,
                    "STRING REJECTED: ANGLE JUMP",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

        else:

            cv2.putText(
                display,
                "NO STRING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

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
            "Horizon — V9 String Tracking",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print(
                "\nStopped manually."
            )

            break

        if frame_number % 30 == 0:

            status = (
                "STRING"
                if result is not None
                else "NO STRING"
            )

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
    # SAVE DATA
    # ========================================================

    np.savez(
        OUTPUT_NPZ,
        time=times,
        theta=theta,
        x=x,
        y=y,
        pivot=np.array(pivot),
        pendulum_length=pendulum_length,
        fps=fps
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    valid = np.isfinite(theta)

    valid_count = np.sum(valid)

    print()
    print("=" * 70)
    print("V9 TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Total frames     : {total}"
    )

    print(
        f"Valid string     : {valid_count}"
    )

    print(
        f"Missing/rejected : "
        f"{total - valid_count}"
    )

    print(
        f"Detection rate   : "
        f"{100 * valid_count / total:.1f}%"
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
    # THETA PLOT
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
        "Horizon — V9 Pendulum Angle θ(t)"
    )

    plt.grid(True)
    plt.legend()

    plt.show()

    # ========================================================
    # TRAJECTORY PLOT
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    plt.plot(
        x,
        y,
        label="Calculated bob"
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
        "Horizon — V9 Pendulum Trajectory"
    )

    plt.gca().invert_yaxis()

    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()