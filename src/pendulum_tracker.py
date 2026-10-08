import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HORIZON — GOLD BOB PENDULUM TRACKER
# COLOR + GEOMETRY BASED TRACKING
# ============================================================

VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

OUTPUT_DATA = "horizon_pendulum_trajectory.npz"
OUTPUT_VIDEO = "horizon_pendulum_tracking.mp4"


# ============================================================
# GOLD / BRASS COLOUR RANGE
# ============================================================
#
# HSV:
# H = hue
# S = saturation
# V = brightness
#
# The bob is gold/brass, so we search for yellow/orange
# pixels rather than using CSRT.
#
# These values are deliberately fairly broad.

LOWER_GOLD = np.array([
    10,
    70,
    50
])

UPPER_GOLD = np.array([
    45,
    255,
    255
])


# ============================================================
# DETECTION PARAMETERS
# ============================================================

MIN_AREA = 20
MAX_AREA = 3000

MIN_RADIUS = 3
MAX_RADIUS = 50

MIN_CIRCULARITY = 0.25

# Maximum movement allowed between consecutive detections.
MAX_MOVEMENT = 80


# ============================================================
# FIND GOLD BOB
# ============================================================

def detect_bob(frame, previous_center=None):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # Gold mask
    # --------------------------------------------------------

    mask = cv2.inRange(
        hsv,
        LOWER_GOLD,
        UPPER_GOLD
    )

    # --------------------------------------------------------
    # Clean mask
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

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

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
            / (perimeter ** 2)
        )

        if circularity < MIN_CIRCULARITY:
            continue

        (x, y), radius = cv2.minEnclosingCircle(
            contour
        )

        if radius < MIN_RADIUS:
            continue

        if radius > MAX_RADIUS:
            continue

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        cx = int(
            moments["m10"]
            / moments["m00"]
        )

        cy = int(
            moments["m01"]
            / moments["m00"]
        )

        candidates.append({
            "center": (cx, cy),
            "radius": radius,
            "area": area,
            "circularity": circularity,
            "contour": contour
        })

    # --------------------------------------------------------
    # No candidates
    # --------------------------------------------------------

    if len(candidates) == 0:

        return None, mask

    # --------------------------------------------------------
    # If this is the first frame:
    # choose the largest/most circular candidate.
    # --------------------------------------------------------

    if previous_center is None:

        candidates.sort(
            key=lambda c: (
                c["area"]
                * c["circularity"]
            ),
            reverse=True
        )

        return candidates[0], mask

    # --------------------------------------------------------
    # Otherwise choose candidate closest to previous bob.
    # --------------------------------------------------------

    px, py = previous_center

    for candidate in candidates:

        cx, cy = candidate["center"]

        distance = np.sqrt(
            (cx - px) ** 2
            + (cy - py) ** 2
        )

        candidate["distance"] = distance

    candidates.sort(
        key=lambda c: c["distance"]
    )

    best = candidates[0]

    # --------------------------------------------------------
    # Reject impossible jumps
    # --------------------------------------------------------

    if best["distance"] > MAX_MOVEMENT:

        return None, mask

    return best, mask


# ============================================================
# DRAW TRACKING
# ============================================================

def draw_overlay(
    frame,
    detection,
    trajectory,
    frame_number,
    fps,
    status
):

    output = frame.copy()

    # --------------------------------------------------------
    # Draw trajectory
    # --------------------------------------------------------

    for i in range(1, len(trajectory)):

        p1 = trajectory[i - 1]
        p2 = trajectory[i]

        if p1 is None or p2 is None:
            continue

        cv2.line(
            output,
            p1,
            p2,
            (255, 0, 0),
            2
        )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    if detection is not None:

        cx, cy = detection["center"]
        radius = int(
            detection["radius"]
        )

        # Circle around bob
        cv2.circle(
            output,
            (cx, cy),
            radius,
            (0, 255, 0),
            2
        )

        # Centre
        cv2.circle(
            output,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"Time: {frame_number / fps:.2f}s",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Frame: {frame_number}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Status: {status}",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("HORIZON — GOLD BOB PENDULUM TRACKER")
    print("COLOR / CONTOUR TRACKING")
    print("=" * 70)

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
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

    duration = (
        total_frames / fps
    )

    print()
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")
    print(
        f"Resolution : "
        f"{width} × {height}"
    )
    print(
        f"Duration   : "
        f"{duration:.2f}s"
    )

    # --------------------------------------------------------
    # Video writer
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
    # Tracking arrays
    # --------------------------------------------------------

    positions = []

    times = []

    trajectory = []

    previous_center = None

    detected_count = 0

    missing_count = 0

    # --------------------------------------------------------
    # Process entire video
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROCESSING ENTIRE VIDEO")
    print("=" * 70)

    print("""
Green circle = detected bob
Red dot      = bob centre
Blue line    = trajectory

The tracker will NOT stop when a frame is missed.

Press Q to stop manually.
""")

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # Detect bob
        # ----------------------------------------------------

        detection, mask = detect_bob(
            frame,
            previous_center
        )

        if detection is not None:

            center = detection["center"]

            positions.append(
                center
            )

            times.append(
                frame_number / fps
            )

            trajectory.append(
                center
            )

            previous_center = center

            detected_count += 1

            status = "DETECTED"

        else:

            positions.append(
                (np.nan, np.nan)
            )

            times.append(
                frame_number / fps
            )

            trajectory.append(
                None
            )

            missing_count += 1

            status = "MISSING"

        # ----------------------------------------------------
        # Draw overlay
        # ----------------------------------------------------

        annotated = draw_overlay(
            frame,
            detection,
            trajectory,
            frame_number,
            fps,
            status
        )

        writer.write(
            annotated
        )

        cv2.imshow(
            "Horizon - Gold Bob Tracking",
            annotated
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
                / total_frames
                * 100
            )

            print(
                f"Frame "
                f"{frame_number:4d}/"
                f"{total_frames} "
                f"({percentage:5.1f}%) | "
                f"{status}"
            )

        frame_number += 1

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    positions = np.array(
        positions,
        dtype=np.float64
    )

    times = np.array(
        times,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Valid data
    # --------------------------------------------------------

    valid_mask = (
        ~np.isnan(
            positions[:, 0]
        )
    )

    valid_positions = positions[
        valid_mask
    ]

    valid_times = times[
        valid_mask
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez(
        OUTPUT_DATA,
        time=valid_times,
        positions=valid_positions,
        fps=fps,
        width=width,
        height=height
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    processed_frames = len(
        positions
    )

    detection_rate = (
        detected_count
        / processed_frames
        * 100
    )

    print("\n" + "=" * 70)
    print("TRACKING SUMMARY")
    print("=" * 70)

    print(
        f"Processed frames : "
        f"{processed_frames}"
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

    print()

    print(
        f"Saved trajectory : "
        f"{OUTPUT_DATA}"
    )

    print(
        f"Saved video      : "
        f"{OUTPUT_VIDEO}"
    )

    # ========================================================
    # PLOT 1
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        times,
        positions[:, 1],
        linewidth=2,
        label="Bob Y"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Vertical position (pixels)"
    )

    plt.title(
        "Horizon — Pendulum Vertical Motion"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    # ========================================================
    # PLOT 2
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    plt.plot(
        positions[:, 0],
        positions[:, 1],
        linewidth=2,
        label="Pendulum bob"
    )

    plt.xlabel(
        "X position (pixels)"
    )

    plt.ylabel(
        "Y position (pixels)"
    )

    plt.title(
        "Horizon — Pendulum Spatial Trajectory"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print("HORIZON PENDULUM TRACKING COMPLETE")
    print("=" * 70)

    print("""
Pipeline:

    VIDEO
      ↓
    GOLD BOB DETECTION
      ↓
    x(t), y(t)
      ↓
    PIVOT DETECTION
      ↓
    θ(t)
      ↓
    θ̇(t), θ̈(t)
      ↓
    SINDy
      ↓
    DISCOVERED PHYSICS
""")


if __name__ == "__main__":
    main()