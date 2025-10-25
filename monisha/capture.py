import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam
import mss

# Setup MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Get screen capture region (full screen here; adjust 'monitor' for window region)
with mss.mss() as sct:
    monitor = sct.monitors[1]  # Primary monitor

    width = monitor['width']
    height = monitor['height']

    # Start virtual camera with screen resolution
    with pyvirtualcam.Camera(width=width, height=height, fps=20) as cam:
        print(f'Virtual camera started: {cam.device}')

        while True:
            # Capture screen frame
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Convert BGR to RGB for mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                    x1 = int(x * width)
                    y1 = int(y * height)
                    w1 = int(w * width)
                    h1 = int(h * height)

                    # Clamp box coordinates
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    w1 = min(w1, width - x1)
                    h1 = min(h1, height - y1)

                    # Blur face region
                    face_roi = frame[y1:y1 + h1, x1:x1 + w1]
                    if face_roi.size > 0:
                        blur = cv2.GaussianBlur(face_roi, (99, 99), 30)
                        frame[y1:y1 + h1, x1:x1 + w1] = blur

            # Show optional preview window
            cv2.imshow('Blurred Screen Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Send frame to virtual cam
            cam.send(frame)
            cam.sleep_until_next_frame()

cv2.destroyAllWindows()
