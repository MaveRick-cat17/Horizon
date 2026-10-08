import cv2
import numpy as np


def track_ball(video_path):

    # ==========================================================
    # OPEN VIDEO
    # ==========================================================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0


    # ==========================================================
    # STORAGE
    # ==========================================================

    detected_positions = []
    detected_frames = []

    all_times = []

    previous_position = None

    previous_velocity = np.array(
        [0.0, 0.0],
        dtype=float
    )

    previous_radius = None

    missing_frames = 0

    frame_index = 0


    # ==========================================================
    # TRACKING PARAMETERS
    # ==========================================================

    # Normal search radius.
    NORMAL_SEARCH = 130

    # Search radius during rapid motion.
    FAST_SEARCH = 280

    # Search radius when recovering from a miss.
    RECOVERY_SEARCH = 400

    # Maximum consecutive missed detections
    # before we stop trusting the previous velocity.
    MAX_MISSING = 10

    # Only short gaps are interpolated.
    MAX_INTERPOLATION_GAP = 3


    # ==========================================================
    # DETECTION PARAMETERS
    # ==========================================================

    # Yellow colour range.
    #
    # We make this slightly more tolerant because
    # the ball becomes blurred during rapid motion.

    LOWER_YELLOW = np.array(
        [20, 90, 70]
    )

    UPPER_YELLOW = np.array(
        [45, 255, 255]
    )


    # Morphological kernel.
    #
    # Slightly larger than before to reconnect
    # motion-blurred yellow regions.

    kernel = np.ones(
        (5, 5),
        np.uint8
    )


    # ==========================================================
    # PROCESS VIDEO
    # ==========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        current_time = (
            frame_index / fps
        )

        all_times.append(
            current_time
        )


        # ======================================================
        # HSV CONVERSION
        # ======================================================

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )


        # ======================================================
        # YELLOW MASK
        # ======================================================

        mask = cv2.inRange(
            hsv,
            LOWER_YELLOW,
            UPPER_YELLOW
        )


        # ======================================================
        # CLEAN MASK
        # ======================================================

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
        # ANALYSE CANDIDATES
        # ======================================================

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            # Lowered from 400.
            #
            # Motion blur can make the visible
            # yellow region smaller.

            if area < 200:
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
                / (perimeter ** 2)
            )


            # More tolerant than before.
            if circularity < 0.25:
                continue


            # ==================================================
            # ENCLOSING CIRCLE
            # ==================================================

            (cx, cy), radius = (
                cv2.minEnclosingCircle(
                    contour
                )
            )


            # Wider radius range to tolerate
            # motion blur / partial detection.

            if radius < 8 or radius > 180:
                continue


            # ==================================================
            # BOUNDING BOX
            # ==================================================

            x, y, w, h = (
                cv2.boundingRect(
                    contour
                )
            )


            if h == 0:
                continue


            aspect_ratio = (
                w / h
            )


            # More tolerant because motion blur
            # can stretch the ball.

            if aspect_ratio < 0.45:
                continue

            if aspect_ratio > 2.2:
                continue


            candidates.append(
                {
                    "x": float(cx),
                    "y": float(cy),
                    "radius": float(radius),
                    "area": float(area),
                    "circularity": float(circularity)
                }
            )


        # ======================================================
        # SELECTED CANDIDATE
        # ======================================================

        selected = None


        # ======================================================
        # FIRST DETECTION
        # ======================================================

        if previous_position is None:

            if candidates:

                # Largest/highest-quality yellow object.
                selected = max(
                    candidates,
                    key=lambda c:
                    (
                        c["area"]
                        * c["circularity"]
                    )
                )


        # ======================================================
        # TRACK EXISTING BALL
        # ======================================================

        else:

            # --------------------------------------------------
            # PREDICT NEXT POSITION
            # --------------------------------------------------

            predicted_position = (
                previous_position
                + previous_velocity
            )


            # --------------------------------------------------
            # ESTIMATE CURRENT SPEED
            # --------------------------------------------------

            speed = np.linalg.norm(
                previous_velocity
            )


            # --------------------------------------------------
            # ADAPTIVE SEARCH RADIUS
            # --------------------------------------------------

            if missing_frames > 0:

                # We already lost the ball.
                search_distance = (
                    RECOVERY_SEARCH
                )

            elif speed > 60:

                # Ball is moving quickly.
                search_distance = (
                    FAST_SEARCH
                )

            else:

                search_distance = (
                    NORMAL_SEARCH
                )


            # --------------------------------------------------
            # SCORE CANDIDATES
            # --------------------------------------------------

            scored_candidates = []


            for candidate in candidates:

                candidate_position = np.array(
                    [
                        candidate["x"],
                        candidate["y"]
                    ],
                    dtype=float
                )


                distance = np.linalg.norm(
                    candidate_position
                    - predicted_position
                )


                # Reject candidates too far away.

                if distance > search_distance:
                    continue


                # ==============================================
                # SIZE CONSISTENCY
                # ==============================================

                size_error = 0.0


                if previous_radius is not None:

                    radius_ratio = (
                        candidate["radius"]
                        / previous_radius
                    )


                    # Very extreme size changes
                    # are probably false detections.

                    if radius_ratio < 0.35:
                        continue

                    if radius_ratio > 2.8:
                        continue


                    size_error = abs(
                        np.log(
                            max(
                                radius_ratio,
                                1e-6
                            )
                        )
                    )


                # ==============================================
                # MOTION SCORE
                # ==============================================

                # Distance is the most important factor.
                #
                # Circularity and size are secondary.

                score = (
                    distance
                    + size_error * 50
                    - candidate["circularity"] * 25
                    - np.sqrt(
                        candidate["area"]
                    ) * 0.05
                )


                scored_candidates.append(
                    (
                        score,
                        candidate
                    )
                )


            # --------------------------------------------------
            # SELECT BEST CANDIDATE
            # --------------------------------------------------

            if scored_candidates:

                scored_candidates.sort(
                    key=lambda item:
                    item[0]
                )

                selected = (
                    scored_candidates[0][1]
                )


        # ======================================================
        # REAL DETECTION
        # ======================================================

        if selected is not None:

            current_position = np.array(
                [
                    selected["x"],
                    selected["y"]
                ],
                dtype=float
            )


            # --------------------------------------------------
            # UPDATE VELOCITY
            # --------------------------------------------------

            if previous_position is not None:

                new_velocity = (
                    current_position
                    - previous_position
                )


                # ------------------------------------------------
                # ADAPTIVE VELOCITY SMOOTHING
                # ------------------------------------------------

                # During rapid motion we want the prediction
                # to react quickly.
                #
                # During slow motion we smooth more strongly.

                new_speed = np.linalg.norm(
                    new_velocity
                )


                if new_speed > 50:

                    previous_velocity = (
                        0.45
                        * previous_velocity
                        + 0.55
                        * new_velocity
                    )

                else:

                    previous_velocity = (
                        0.70
                        * previous_velocity
                        + 0.30
                        * new_velocity
                    )


            # --------------------------------------------------
            # UPDATE TRACK STATE
            # --------------------------------------------------

            previous_position = (
                current_position.copy()
            )

            previous_radius = (
                selected["radius"]
            )


            # --------------------------------------------------
            # STORE REAL OBSERVATION
            # --------------------------------------------------

            detected_positions.append(
                current_position
            )

            detected_frames.append(
                frame_index
            )


            # Detection recovered.
            missing_frames = 0


        # ======================================================
        # MISSING DETECTION
        # ======================================================

        else:

            missing_frames += 1


            # Do NOT insert fake position here.
            #
            # We want to preserve the distinction between
            # real and interpolated observations.


            if missing_frames > MAX_MISSING:

                previous_velocity = np.array(
                    [0.0, 0.0],
                    dtype=float
                )


        frame_index += 1


    # ==========================================================
    # RELEASE VIDEO
    # ==========================================================

    cap.release()


    # ==========================================================
    # NO DETECTIONS
    # ==========================================================

    if len(detected_positions) == 0:

        return (
            np.array([]),
            np.empty(
                (0, 2)
            )
        )


    detected_positions = np.array(
        detected_positions,
        dtype=float
    )

    detected_frames = np.array(
        detected_frames,
        dtype=int
    )

    all_times = np.array(
        all_times,
        dtype=float
    )


    # ==========================================================
    # BUILD FRAME-BY-FRAME TRAJECTORY
    # ==========================================================

    full_positions = np.full(
        (
            len(all_times),
            2
        ),
        np.nan,
        dtype=float
    )


    full_positions[
        detected_frames
    ] = detected_positions


    # ==========================================================
    # INTERPOLATE ONLY VERY SHORT GAPS
    # ==========================================================

    interpolated = np.zeros(
        len(all_times),
        dtype=bool
    )


    for i in range(
        len(detected_frames) - 1
    ):

        frame_a = (
            detected_frames[i]
        )

        frame_b = (
            detected_frames[i + 1]
        )

        gap = (
            frame_b
            - frame_a
            - 1
        )


        # ------------------------------------------------------
        # SHORT GAP
        # ------------------------------------------------------

        if (
            gap > 0
            and gap
            <= MAX_INTERPOLATION_GAP
        ):

            start_position = (
                full_positions[
                    frame_a
                ]
            )

            end_position = (
                full_positions[
                    frame_b
                ]
            )


            for frame in range(
                frame_a + 1,
                frame_b
            ):

                alpha = (
                    frame - frame_a
                ) / (
                    frame_b - frame_a
                )


                full_positions[
                    frame
                ] = (
                    start_position
                    + alpha
                    * (
                        end_position
                        - start_position
                    )
                )


                interpolated[
                    frame
                ] = True


    # ==========================================================
    # KEEP ONLY VALID DATA
    # ==========================================================

    valid = ~np.isnan(
        full_positions[:, 0]
    )


    times = all_times[
        valid
    ]

    positions = full_positions[
        valid
    ]


    # ==========================================================
    # SUMMARY
    # ==========================================================

    real_count = np.sum(
        ~interpolated[valid]
    )

    interpolated_count = np.sum(
        interpolated[valid]
    )


    print()
    print(
        "TRACKER DATA SUMMARY"
    )

    print(
        "-" * 45
    )

    print(
        f"Real detections      : "
        f"{real_count}"
    )

    print(
        f"Interpolated points  : "
        f"{interpolated_count}"
    )

    print(
        f"Total usable points  : "
        f"{len(positions)}"
    )

    print(
        f"Video frames         : "
        f"{len(all_times)}"
    )

    print(
        "-" * 45
    )


    return (
        times,
        positions
    )