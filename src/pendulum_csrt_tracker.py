import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON — CSRT PENDULUM TRACKER
# ============================================================

VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_DATA = "horizon_pendulum_csrt.npz"
OUTPUT_VIDEO = "horizon_pendulum_csrt_tracking.mp4"

# Fixed pivot from the actual video.
# We can refine this later if necessary.
PIVOT = (250, 45)

# Tracking sanity limits
MAX_JUMP = 100.0          # pixels/frame
MAX_RADIUS_ERROR = 0.25   # 25%

# CSRT parameters
TRACKER_PADDING = 1.5


# ============================================================
# CREATE CSRT
# ============================================================

def create_csrt():

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(
        cv2.legacy,
        "TrackerCSRT_create"
    ):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT is not available."
    )


# ============================================================
# FRAME BROWSER
# ============================================================

def choose_start_frame(cap, total_frames, fps):

    frame_number = 0

    print("\n" + "=" * 70)
    print("SELECT STARTING FRAME")
    print("=" * 70)

    print("""
D / RIGHT ARROW : next frame
A / LEFT ARROW  : previous frame
ENTER           : select
ESC             : cancel

Choose a frame where:
  ✓ the gold bob is clearly visible
  ✓ the bob is not hidden
  ✓ the bob is inside the frame
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
            (PIVOT[0] + 10, PIVOT[1]),
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

        if key == 27:

            cv2.destroyWindow(
                "Horizon - Frame Browser"
            )

            return None

        if key in [ord("d"), ord("D"), 83]:

            frame_number = min(
                frame_number + 1,
                total_frames - 1
            )

        if key in [ord("a"), ord("A"), 81]:

            frame_number = max(
                frame_number - 1,
                0
            )


# ============================================================
# SELECT BOB
# ============================================================

def select_bob(cap, frame_number):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    ret, frame = cap.read()

    if not ret:
        return None, None

    display = frame.copy()

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
        (PIVOT[0] + 10, PIVOT[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2
    )

    print("\n" + "=" * 70)
    print("SELECT THE GOLD BOB")
    print("=" * 70)

    print("""
Draw a TIGHT rectangle around the GOLD SPHERICAL BOB.

Do NOT include:
  - the string
  - the base
  - the black stand
  - large amounts of background

Press ENTER after selecting.
""")

    roi = cv2.selectROI(
        "SELECT PENDULUM BOB",
        display,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(
        "SELECT PENDULUM BOB"
    )

    x, y, w, h = roi

    if w <= 1 or h <= 1:
        return None, None

    return frame, (
        int(x),
        int(y),
        int(w),
        int(h)
    )


# ============================================================
# ROI CENTRE
# ============================================================

def roi_center(box):

    x, y, w, h = box

    return (
        x + w / 2,
        y + h / 2
    )


# ============================================================
# ANGLE FROM PIVOT
# ============================================================

def calculate_theta(point):

    dx = point[0] - PIVOT[0]
    dy = point[1] - PIVOT[1]

    return np.arctan2(
        dx,
        dy
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("HORIZON — CSRT PENDULUM TRACKER")
    print("=" * 70)

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

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(f"\nFPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")
    print(f"Resolution : {width} × {height}")
    print(
        f"Duration   : "
        f"{total_frames / fps:.2f}s"
    )

    # --------------------------------------------------------
    # Starting frame
    # --------------------------------------------------------

    start_frame = choose_start_frame(
        cap,
        total_frames,
        fps
    )

    if start_frame is None:
        cap.release()
        return

    print(
        f"\nSelected starting frame: "
        f"{start_frame}"
    )

    # --------------------------------------------------------
    # Select bob
    # --------------------------------------------------------

    first_frame, initial_roi = select_bob(
        cap,
        start_frame
    )

    if initial_roi is None:

        cap.release()

        print(
            "Bob selection cancelled."
        )

        return

    print(
        f"Initial ROI: {initial_roi}"
    )

    initial_center = roi_center(
        initial_roi
    )

    pendulum_length = np.linalg.norm(
        np.array(initial_center)
        -
        np.array(PIVOT)
    )

    print(
        f"Initial bob centre: "
        f"({initial_center[0]:.1f}, "
        f"{initial_center[1]:.1f})"
    )

    print(
        f"Estimated pendulum length: "
        f"{pendulum_length:.1f}px"
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if pendulum_length < 200:

        print()
        print(
            "ERROR: The selected bob is too close "
            "to the pivot."
        )

        print(
            "Please select the actual bob."
        )

        cap.release()

        return

    # --------------------------------------------------------
    # Create CSRT
    # --------------------------------------------------------

    tracker = create_csrt()

    tracker.init(
        first_frame,
        initial_roi
    )

    print("\nCSRT initialised.")

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
    # Data
    # --------------------------------------------------------

    positions = np.full(
        (total_frames, 2),
        np.nan
    )

    theta_array = np.full(
        total_frames,
        np.nan
    )

    times = np.arange(
        total_frames
    ) / fps

    positions[start_frame] = initial_center

    theta_array[start_frame] = (
        calculate_theta(initial_center)
    )

    previous_center = initial_center

    previous_theta = theta_array[
        start_frame
    ]

    detected_count = 1
    rejected_count = 0

    trajectory = [
        None
    ] * total_frames

    trajectory[start_frame] = (
        initial_center
    )

    # --------------------------------------------------------
    # Process from selected frame onward
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame + 1
    )

    print("\n" + "=" * 70)
    print("TRACKING ENTIRE VIDEO")
    print("=" * 70)

    for frame_number in range(
        start_frame + 1,
        total_frames
    ):

        ret, frame = cap.read()

        if not ret:
            break

        success, box = tracker.update(
            frame
        )

        valid = False

        if success:

            center = roi_center(
                box
            )

            # ------------------------------------------------
            # Check position jump
            # ------------------------------------------------

            jump = np.linalg.norm(
                np.array(center)
                -
                np.array(previous_center)
            )

            # ------------------------------------------------
            # Check distance from pivot
            # ------------------------------------------------

            radius = np.linalg.norm(
                np.array(center)
                -
                np.array(PIVOT)
            )

            radius_error = abs(
                radius - pendulum_length
            ) / pendulum_length

            # ------------------------------------------------
            # Check angle change
            # ------------------------------------------------

            current_theta = calculate_theta(
                center
            )

            angle_change = abs(
                np.degrees(
                    np.arctan2(
                        np.sin(
                            current_theta
                            -
                            previous_theta
                        ),
                        np.cos(
                            current_theta
                            -
                            previous_theta
                        )
                    )
                )
            )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if (
                jump <= MAX_JUMP
                and
                radius_error <= MAX_RADIUS_ERROR
                and
                angle_change <= 15.0
            ):

                valid = True

        if valid:

            previous_center = center
            previous_theta = current_theta

            positions[
                frame_number
            ] = center

            theta_array[
                frame_number
            ] = current_theta

            trajectory[
                frame_number
            ] = center

            detected_count += 1

            status = "CSRT VALID"

        else:

            rejected_count += 1

            center = None

            status = "REJECTED / LOST"

        # ----------------------------------------------------
        # Draw trajectory
        # ----------------------------------------------------

        display = frame.copy()

        for i in range(
            max(1, frame_number - 150),
            frame_number
        ):

            p1 = trajectory[i - 1]
            p2 = trajectory[i]

            if p1 is None or p2 is None:
                continue

            cv2.line(
                display,
                (
                    int(p1[0]),
                    int(p1[1])
                ),
                (
                    int(p2[0]),
                    int(p2[1])
                ),
                (255, 0, 0),
                2
            )

        # ----------------------------------------------------
        # Pivot
        # ----------------------------------------------------

        cv2.circle(
            display,
            PIVOT,
            7,
            (255, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # Pendulum orbit
        # ----------------------------------------------------

        cv2.circle(
            display,
            PIVOT,
            int(pendulum_length),
            (100, 100, 100),
            1
        )

        # ----------------------------------------------------
        # Tracker box
        # ----------------------------------------------------

        if success:

            x, y, w, h = map(
                int,
                box
            )

            box_colour = (
                (0, 255, 0)
                if valid
                else
                (0, 0, 255)
            )

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                box_colour,
                2
            )

        # ----------------------------------------------------
        # Valid centre
        # ----------------------------------------------------

        if center is not None:

            c = (
                int(center[0]),
                int(center[1])
            )

            cv2.circle(
                display,
                c,
                6,
                (0, 0, 255),
                -1
            )

            cv2.line(
                display,
                PIVOT,
                c,
                (0, 255, 255),
                2
            )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Time: {frame_number / fps:.2f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Frame: {frame_number}/{total_frames - 1}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Status: {status}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                (0, 255, 0)
                if valid
                else
                (0, 0, 255)
            ),
            2
        )

        # ----------------------------------------------------
        # Write
        # ----------------------------------------------------

        writer.write(
            display
        )

        cv2.imshow(
            "Horizon - CSRT Pendulum Tracker",
            display
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
                /
                total_frames
                * 100
            )

            print(
                f"Frame "
                f"{frame_number:4d}/"
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
        /
        processed
        *
        100
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
    print("CSRT TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Processed frames   : {processed}"
    )

    print(
        f"Valid frames       : {detected_count}"
    )

    print(
        f"Rejected/lost      : {rejected_count}"
    )

    print(
        f"Detection rate     : "
        f"{detection_rate:.1f}%"
    )

    print(
        f"Pendulum length    : "
        f"{pendulum_length:.1f}px"
    )

    print(
        f"Saved data         : "
        f"{OUTPUT_DATA}"
    )

    print(
        f"Saved video        : "
        f"{OUTPUT_VIDEO}"
    )

    # ========================================================
    # TRAJECTORY PLOT
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
            "Horizon — CSRT Pendulum Trajectory"
        )

        plt.grid(True)
        plt.legend()

        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.show()

        # ====================================================
        # ANGLE PLOT
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
    print("HORIZON CSRT TRACKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()