import cv2

# load pretrained OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

# open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Press 'b' to toggle blur, 'q' to quit.")
blur_on = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        if blur_on:  # apply Gaussian blur to each face region
            roi = frame[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(roi, (99, 99), 30)
            frame[y:y+h, x:x+w] = blur
        color = (0, 0, 255) if blur_on else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Live Face Blur", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('b'):  # toggle blur
        blur_on = not blur_on
        print(f"Blur {'ON' if blur_on else 'OFF'}")
    elif key == ord('q'):  # quit
        break

cap.release()
cv2.destroyAllWindows()
