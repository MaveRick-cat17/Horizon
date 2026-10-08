import cv2
import numpy as np


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

# Gold/brass HSV range
LOWER_GOLD = np.array([5, 50, 35])
UPPER_GOLD = np.array([45, 255, 255])

MIN_AREA = 8
MAX_AREA = 4000

MIN_RADIUS = 2
MAX_RADIUS = 70


def get_candidates(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        LOWER_GOLD,
        UPPER_GOLD
    )

    kernel = np.ones((5, 5), np.uint8)

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

        if area < MIN_AREA or area > MAX_AREA:
            continue

        (x, y), radius = cv2.minEnclosingCircle(
            contour
        )

        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        candidates.append({
            "center": (cx, cy),
            "radius": radius,
            "area": area
        })

    return candidates


def select_point(frame, title):

    roi = cv2.selectROI(
        title,
        frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(title)

    x, y, w, h = roi

    if w <= 1 or h <= 1:
        return None

    return (
        int(x + w / 2),
        int(y + h / 2)
    )


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print("=" * 65)
    print("HORIZON — PENDULUM VISION DIAGNOSTIC")
    print("=" * 65)

    print(f"FPS: {fps}")
    print(f"Frames: {total}")
    print(f"Duration: {total / fps:.2f}s")

    # --------------------------------------------------------
    # First frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        return

    # --------------------------------------------------------
    # Select pivot
    # --------------------------------------------------------

    print()
    print("SELECT THE ACTUAL PIVOT.")
    print("Select the point where the string attaches at the TOP.")

    pivot = select_point(
        frame,
        "SELECT PIVOT"
    )

    if pivot is None:
        cap.release()
        return

    print(
        f"Pivot selected: {pivot}"
    )

    # --------------------------------------------------------
    # Select bob
    # --------------------------------------------------------

    print()
    print("SELECT THE GOLD BOB.")

    bob = select_point(
        frame,
        "SELECT BOB"
    )

    if bob is None:
        cap.release()
        return

    print(
        f"Bob selected: {bob}"
    )

    # --------------------------------------------------------
    # Estimate radius
    # --------------------------------------------------------

    pendulum_length = np.linalg.norm(
        np.array(bob, dtype=float)
        -
        np.array(pivot, dtype=float)
    )

    print(
        f"Initial pixel length: "
        f"{pendulum_length:.1f}px"
    )

    print()
    print("=" * 65)
    print("DIAGNOSTIC RUNNING")
    print("=" * 65)

    print("""
ORANGE circles = ALL gold candidates
GREEN circles  = candidates near expected pendulum radius
RED circle     = closest green candidate to previous bob

Press Q to stop.
""")

    previous_bob = bob

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        candidates = get_candidates(frame)

        display = frame.copy()

        # ----------------------------------------------------
        # Draw pivot
        # ----------------------------------------------------

        cv2.circle(
            display,
            pivot,
            7,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            display,
            "PIVOT",
            (
                pivot[0] + 10,
                pivot[1]
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

        # ----------------------------------------------------
        # Expected radius circle
        # ----------------------------------------------------

        cv2.circle(
            display,
            pivot,
            int(pendulum_length),
            (120, 120, 120),
            1
        )

        valid_candidates = []

        # ----------------------------------------------------
        # Candidates
        # ----------------------------------------------------

        for candidate in candidates:

            cx, cy = candidate["center"]

            distance = np.linalg.norm(
                np.array(
                    candidate["center"],
                    dtype=float
                )
                -
                np.array(
                    pivot,
                    dtype=float
                )
            )

            # Draw every candidate
            cv2.circle(
                display,
                (cx, cy),
                max(
                    4,
                    int(candidate["radius"])
                ),
                (0, 140, 255),
                2
            )

            # Radius tolerance
            if abs(
                distance - pendulum_length
            ) <= pendulum_length * 0.20:

                valid_candidates.append(
                    candidate
                )

                cv2.circle(
                    display,
                    (cx, cy),
                    8,
                    (0, 255, 0),
                    2
                )

        # ----------------------------------------------------
        # Choose closest geometrically valid candidate
        # ----------------------------------------------------

        chosen = None

        if valid_candidates:

            valid_candidates.sort(
                key=lambda c:
                np.linalg.norm(
                    np.array(
                        c["center"],
                        dtype=float
                    )
                    -
                    np.array(
                        previous_bob,
                        dtype=float
                    )
                )
            )

            chosen = valid_candidates[0]

            previous_bob = chosen["center"]

            cv2.circle(
                display,
                chosen["center"],
                10,
                (0, 0, 255),
                -1
            )

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        status = (
            "CANDIDATE FOUND"
            if chosen is not None
            else "NO GEOMETRIC CANDIDATE"
        )

        cv2.putText(
            display,
            f"Frame: {frame_number}",
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
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Gold candidates: {len(candidates)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Valid radius candidates: "
            f"{len(valid_candidates)}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            status,
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Horizon - Vision Diagnostic",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 65)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()