import cv2

VIDEO = "data/Videos/vidssave.com Simple Pendulum 480P.mp4"

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Frames:", frames)

# Frames we want to inspect
check_frames = [
    0,
    60,
    120,
    180,
    240,
    300,
    360,
    420,
    480,
    540,
    600,
    660,
    720
]

for frame_number in check_frames:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        continue

    display = frame.copy()

    time = frame_number / fps

    cv2.putText(
        display,
        f"Frame: {frame_number}  Time: {time:.2f}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Horizon - Pendulum Preview",
        display
    )

    print(
        f"Showing frame {frame_number} "
        f"({time:.2f}s)"
    )

    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()