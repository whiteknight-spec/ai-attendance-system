
import cv2
import numpy as np
from pathlib import Path

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load face detector")

dataset_dir = Path(__file__).resolve().parent / "dataset"
dataset_dir.mkdir(exist_ok=True)


def make_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32))
    gray = cv2.equalizeHist(gray)

    dct = cv2.dct(np.float32(gray))
    low = dct[:8, :8].flatten()[1:]
    return low > np.median(low)


def find_duplicate(face):
    new_hash = make_hash(face)

    for path in dataset_dir.glob("face_*.jpg"):
        saved = cv2.imread(str(path))

        if saved is None or saved.size == 0:
            continue

        if saved.shape[0] < 80 or saved.shape[1] < 80:
            continue

        old_hash = make_hash(saved)
        distance = np.count_nonzero(new_hash != old_hash)

        # Less strict than the previous threshold.
        if distance <= 16:
            return path.name

    return None


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

message = "S: Save | Q: Quit"
message_color = (255, 255, 255)
message_frames = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read webcam frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80)
        )

        crops = []

        for x, y, w, h in faces:
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            crop = frame[y:y + h, x:x + w].copy()

            if crop.size > 0:
                crops.append(crop)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if not crops:
                message = "No face detected - not saved"
                message_color = (0, 0, 255)
            else:
                duplicate_name = find_duplicate(crops[0])

                if duplicate_name:
                    message = f"Already saved: {duplicate_name}"
                    message_color = (0, 165, 255)
                else:
                    number = 1

                    while (dataset_dir / f"face_{number}.jpg").exists():
                        number += 1

                    path = dataset_dir / f"face_{number}.jpg"

                    if cv2.imwrite(str(path), crops[0]):
                        message = f"Saved: {path.name}"
                        message_color = (0, 255, 0)
                    else:
                        message = "Save failed"
                        message_color = (0, 0, 255)

            message_frames = 120

        if key == ord("q"):
            break

        # Display current status on the main camera frame.
        display_message = (
            message if message_frames > 0
            else "S: Save | Q: Quit"
        )

        cv2.putText(
            frame,
            display_message[:55],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            message_color if message_frames > 0 else (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Face Detection", frame)

        # Display the same status below the face crop.
        if crops:
            crop = crops[0]
            bar = np.zeros((45, crop.shape[1], 3), dtype=np.uint8)

            cv2.putText(
                bar,
                display_message[:35],
                (5, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                message_color if message_frames > 0 else (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow("Cropped Face", np.vstack((crop, bar)))

        if message_frames > 0:
            message_frames -= 1

        if cv2.getWindowProperty(
            "Face Detection", cv2.WND_PROP_VISIBLE
        ) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Camera stopped.")
