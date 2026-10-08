import cv2
import numpy as np


VIDEO_PATH = (
    "data/Videos/"
    "vidssave.com Simple Pendulum 480P.mp4"
)

# ============================================================
# FIXED PIVOT
# ============================================================

PIVOT = (250, 45)

TOP_Y = 45
BOTTOM_Y = 620


# ============================================================
# EDGE PARAMETERS
# ============================================================

CANNY_LOW = 30
CANNY_HIGH = 100


# ============================================================
# DETECT PENDULUM STRING
# ============================================================

def detect_string(frame):

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

    # --------------------------------------------------------
    # Only use the region below the pivot
    # --------------------------------------------------------

    mask = np.zeros_like(edges)

    mask[
        TOP_Y:BOTTOM_Y,
        :
    ] = edges[
        TOP_Y:BOTTOM_Y,
        :
    ]

    # --------------------------------------------------------
    # Hough lines
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 360,
        threshold=20,
        minLineLength=60,
        maxLineGap=50
    )

    if lines is None:
        return None, edges, []

    px, py = PIVOT

    candidates = []

    # ========================================================
    # EXAMINE EVERY LINE
    # ========================================================

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

        # ----------------------------------------------------
        # Ignore almost-horizontal lines
        # ----------------------------------------------------

        if abs(dy) < abs(dx) * 1.2:
            continue

        # ----------------------------------------------------
        # Endpoint distances from pivot
        # ----------------------------------------------------

        d1 = np.hypot(
            x1 - px,
            y1 - py
        )

        d2 = np.hypot(
            x2 - px,
            y2 - py
        )

        # One end should be reasonably close to pivot
        near_pivot = min(
            d1,
            d2
        )

        if near_pivot > 220:
            continue

        # ----------------------------------------------------
        # Determine which endpoint is farther from pivot
        # ----------------------------------------------------

        if d1 > d2:

            far_x = x1
            far_y = y1

        else:

            far_x = x2
            far_y = y2

        # ----------------------------------------------------
        # Horizontal displacement from pivot
        #
        # THIS IS THE IMPORTANT PART.
        #
        # The black stand stays around x = 250.
        # The pendulum string moves strongly left/right.
        # ----------------------------------------------------

        horizontal_displacement = abs(
            far_x - px
        )

        # ----------------------------------------------------
        # Reject the central support
        #
        # Anything that remains within ~35 px of the pivot
        # vertical axis is very likely the stand.
        # ----------------------------------------------------

        if horizontal_displacement < 35:
            continue

        # ----------------------------------------------------
        # Calculate pendulum angle
        #
        # 0° = vertical downward
        # + = right
        # - = left
        # ----------------------------------------------------

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

        angle_degrees = abs(
            np.degrees(angle)
        )

        # ----------------------------------------------------
        # Extremely small angles are probably the stand.
        # ----------------------------------------------------

        if angle_degrees < 4:
            continue

        # ----------------------------------------------------
        # Score candidate
        #
        # We WANT:
        #   - long lines
        #   - large horizontal displacement
        #   - endpoint far from pivot
        #
        # We DON'T want:
        #   - central vertical lines
        # ----------------------------------------------------

        score = (
            2.0 * horizontal_displacement
            +
            0.5 * distance
            +
            0.25 * length
        )

        candidates.append(
            {
                "score": score,
                "length": length,
                "angle": angle,
                "distance": distance,
                "horizontal_displacement":
                    horizontal_displacement,
                "line": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }
        )

    # --------------------------------------------------------
    # Sort by score
    # --------------------------------------------------------

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    if not candidates:

        return None, edges, []

    return (
        candidates[0],
        edges,
        candidates
    )


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
    print("HORIZON — PENDULUM STRING DIAGNOSTIC V2")
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
    print("Purple = pivot")
    print("Blue   = selected string candidate")
    print("Red    = inferred pendulum direction")
    print("Orange = other possible string candidates")
    print()
    print("The central black stand is explicitly rejected.")
    print()
    print("Press Q to stop.")

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        display = frame.copy()

        result, edges, candidates = detect_string(
            frame
        )

        # ====================================================
        # PIVOT
        # ====================================================

        cv2.circle(
            display,
            PIVOT,
            7,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            display,
            "PIVOT",
            (
                PIVOT[0] + 10,
                PIVOT[1]
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

        # ====================================================
        # SHOW OTHER CANDIDATES
        # ====================================================

        for candidate in candidates[:8]:

            x1, y1, x2, y2 = (
                candidate["line"]
            )

            cv2.line(
                display,
                (x1, y1),
                (x2, y2),
                (0, 140, 255),
                1
            )

        # ====================================================
        # BEST STRING
        # ====================================================

        if result is not None:

            length = result["length"]
            angle = result["angle"]
            distance = result["distance"]
            displacement = (
                result[
                    "horizontal_displacement"
                ]
            )

            x1, y1, x2, y2 = (
                result["line"]
            )

            # ------------------------------------------------
            # Blue detected string
            # ------------------------------------------------

            cv2.line(
                display,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                4
            )

            # ------------------------------------------------
            # Extend from pivot
            # ------------------------------------------------

            ray_length = 650

            predicted_x = int(
                PIVOT[0]
                +
                ray_length
                *
                np.sin(angle)
            )

            predicted_y = int(
                PIVOT[1]
                +
                ray_length
                *
                np.cos(angle)
            )

            cv2.line(
                display,
                PIVOT,
                (
                    predicted_x,
                    predicted_y
                ),
                (0, 0, 255),
                2
            )

            # ------------------------------------------------
            # Information
            # ------------------------------------------------

            angle_deg = np.degrees(
                angle
            )

            cv2.putText(
                display,
                f"Angle: "
                f"{angle_deg:.2f} deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"String length: "
                f"{length:.1f}px",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Pivot distance: "
                f"{distance:.1f}px",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Horizontal offset: "
                f"{displacement:.1f}px",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                "STRING CANDIDATE",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display,
                "NO STRING CANDIDATE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        # ====================================================
        # FRAME / TIME
        # ====================================================

        cv2.putText(
            display,
            f"Frame: "
            f"{frame_number}/{total - 1}",
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
            f"Time: "
            f"{frame_number / fps:.2f}s",
            (
                10,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Horizon - String Diagnostic V2",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        frame_number += 1

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("STRING DIAGNOSTIC V2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()