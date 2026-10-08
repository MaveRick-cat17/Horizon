import cv2
import numpy as np


VIDEO_PATH = "data/Videos/Ball.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise ValueError("Could not open video.")


# ==========================================================
# TRACKING STATE
# ==========================================================

previous_position = None
previous_velocity = np.array([0.0, 0.0])

previous_radius = None

trajectory = []

frame_index = 0

NORMAL_SEARCH = 120
MAX_JUMP = 140


while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ======================================================
    # HSV
    # ======================================================

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    # ======================================================
    # YELLOW MASK
    # ======================================================

    lower = np.array(
        [25, 140, 120]
    )

    upper = np.array(
        [40, 255, 255]
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )


    # ======================================================
    # CLEAN MASK
    # ======================================================

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


    # ======================================================
    # FIND CONTOURS
    # ======================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []


    # ======================================================
    # GENERATE CANDIDATES
    # ======================================================

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 200:
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


        if circularity < 0.45:
            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        if h == 0:
            continue


        aspect_ratio = w / h


        if aspect_ratio < 0.60:
            continue

        if aspect_ratio > 1.40:
            continue


        # Centre
        cx = x + w / 2
        cy = y + h / 2


        # Radius
        radius = (w + h) / 4


        candidates.append({
            "x": cx,
            "y": cy,
            "radius": radius,
            "area": area,
            "circularity": circularity,
            "box": (x, y, w, h)
        })


    # ======================================================
    # PREDICT NEXT POSITION
    # ======================================================

    selected = None

    predicted_position = None


    if previous_position is not None:

        predicted_position = (
            previous_position
            + previous_velocity
        )


    # ======================================================
    # FIRST FRAME
    # ======================================================

    if previous_position is None:

        if candidates:

            selected = max(
                candidates,
                key=lambda c:
                c["area"] * c["circularity"]
            )


    # ======================================================
    # TRACKING
    # ======================================================

    else:

        scored_candidates = []


        for candidate in candidates:

            candidate_position = np.array([
                candidate["x"],
                candidate["y"]
            ])


            # ----------------------------------------------
            # Distance from predicted position
            # ----------------------------------------------

            distance = np.linalg.norm(
                candidate_position
                - predicted_position
            )


            # ----------------------------------------------
            # Reject distant candidates
            # ----------------------------------------------

            if distance > NORMAL_SEARCH:
                continue


            # ----------------------------------------------
            # Actual jump from previous position
            # ----------------------------------------------

            actual_jump = np.linalg.norm(
                candidate_position
                - previous_position
            )


            if actual_jump > MAX_JUMP:
                continue


            # ----------------------------------------------
            # Radius consistency
            # ----------------------------------------------

            size_error = 0.0

            if previous_radius is not None:

                radius_ratio = (
                    candidate["radius"]
                    / previous_radius
                )


                if radius_ratio < 0.60:
                    continue


                if radius_ratio > 1.70:
                    continue


                size_error = abs(
                    radius_ratio - 1.0
                )


            # ----------------------------------------------
            # Score
            # ----------------------------------------------

            score = (
                distance
                + size_error * 80
                - candidate["circularity"] * 60
                - np.sqrt(candidate["area"]) * 0.08
            )


            scored_candidates.append(
                (score, candidate)
            )


        # ----------------------------------------------
        # Select best candidate
        # ----------------------------------------------

        if scored_candidates:

            scored_candidates.sort(
                key=lambda item: item[0]
            )

            selected = scored_candidates[0][1]


    # ======================================================
    # DRAW ALL CANDIDATES
    # ======================================================

    for candidate in candidates:

        x, y, w, h = candidate["box"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 120, 255),
            2
        )


        cv2.putText(
            frame,
            "candidate",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 120, 255),
            1
        )


    # ======================================================
    # DRAW PREDICTED POSITION
    # ======================================================

    if predicted_position is not None:

        px = int(predicted_position[0])
        py = int(predicted_position[1])


        cv2.circle(
            frame,
            (px, py),
            10,
            (255, 0, 255),
            2
        )


        cv2.putText(
            frame,
            "PREDICTED",
            (px + 12, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )


    # ======================================================
    # DRAW SELECTED BALL
    # ======================================================

    if selected is not None:

        x, y, w, h = selected["box"]

        cx = int(selected["x"])
        cy = int(selected["y"])


        # Bright green = selected object
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )


        cv2.circle(
            frame,
            (cx, cy),
            7,
            (0, 255, 0),
            -1
        )


        cv2.putText(
            frame,
            "SELECTED BALL",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # ----------------------------------------------
        # Update motion
        # ----------------------------------------------

        current_position = np.array([
            selected["x"],
            selected["y"]
        ])


        if previous_position is not None:

            velocity = (
                current_position
                - previous_position
            )


            previous_velocity = (
                0.7 * previous_velocity
                + 0.3 * velocity
            )


        previous_position = current_position

        previous_radius = selected["radius"]


        trajectory.append(
            current_position.copy()
        )


    # ======================================================
    # DRAW TRAJECTORY
    # ======================================================

    for i in range(
        1,
        len(trajectory)
    ):

        p1 = tuple(
            trajectory[i - 1].astype(int)
        )

        p2 = tuple(
            trajectory[i].astype(int)
        )


        cv2.line(
            frame,
            p1,
            p2,
            (255, 0, 0),
            2
        )


    # ======================================================
    # INFORMATION PANEL
    # ======================================================

    cv2.putText(
        frame,
        f"Frame: {frame_index}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Candidates: {len(candidates)}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    if selected is not None:

        cv2.putText(
            frame,
            f"Selected: ({int(selected['x'])}, {int(selected['y'])})",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ======================================================
    # DISPLAY
    # ======================================================

    cv2.imshow(
        "Horizon - Tracking Debug",
        frame
    )


    cv2.imshow(
        "Horizon - Yellow Mask",
        mask
    )


    # ======================================================
    # CONTROLS
    # ======================================================

    key = cv2.waitKey(30) & 0xFF


    if key == ord("q"):
        break


    if key == ord(" "):

        while True:

            pause_key = cv2.waitKey(0) & 0xFF

            if pause_key == ord(" "):
                break

            if pause_key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()


    frame_index += 1


cap.release()

cv2.destroyAllWindows()