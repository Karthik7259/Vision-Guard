import cv2
import numpy as np
import mss
import time

# -------- SCREEN SETTINGS --------
MONITOR = {"top": 0, "left": 0, "width": 1280, "height": 720}

# -------- DNN FACE DETECTOR --------
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT = "deploy.prototxt"

# PROTOTXT = "deploy.prototxt"
# MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

def blur_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # filter weak detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Clip values to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                face = cv2.GaussianBlur(face, (99, 99), 30)
                frame[y1:y2, x1:x2] = face

    return frame

def main():
    with mss.mss() as sct:
        monitor = MONITOR
        width, height = monitor["width"], monitor["height"]

        out = cv2.VideoWriter('blurred_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        print("[INFO] Recording with DNN face blurring. Press 'q' to stop.")

        while True:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frame = blur_faces(frame)

            cv2.imshow("Blurred Screen", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()
        print("[INFO] Saved as blurred_output.mp4")

if __name__ == "__main__":
    main()
