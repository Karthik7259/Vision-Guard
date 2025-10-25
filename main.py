import cv2
import numpy as np
import mss

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with mss.mss() as sct:
    # Pick the monitor where WhatsApp is open
    # Replace 1 with the correct monitor index if needed
    monitor = sct.monitors[1]

    while True:
        # Capture the monitor
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Blur detected faces
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_blur = cv2.GaussianBlur(face, (25, 25), 30)
            img[y:y+h, x:x+w] = face_blur

        # Show the blurred monitor in a single window
        cv2.imshow("Blurred WhatsApp Monitor", img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
