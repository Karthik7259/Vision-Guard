import cv2
import numpy as np

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Haar cascade for full coverage of faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def load_and_prepare(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    faces = detect_faces(img)
    print(f"Detected {len(faces)} faces in the image.")
    blurred_rois = [cv2.GaussianBlur(img[y:y+h, x:x+w], (99, 99), 30) for (x, y, w, h) in faces]
    blur_states = [False] * len(faces)
    return img, faces, blurred_rois, blur_states

def render_frame(original, faces, blurred_rois, blur_states):
    frame = original.copy()
    for i, (x, y, w, h) in enumerate(faces):
        if blur_states[i]:
            frame[y:y+h, x:x+w] = blurred_rois[i]
        color = (0, 0, 255) if blur_states[i] else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    return frame

def interactive_face_blur(image_path, output_path="final_blurred.png"):
    img_original, faces, blurred_rois, blur_states = load_and_prepare(image_path)
    if len(faces) == 0:
        print("No faces detected, exiting.")
        return

    window_name = "Face Selector | click toggles blur | s = save | q = quit"
    cv2.namedWindow(window_name)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (fx, fy, fw, fh) in enumerate(faces):
                if fx <= x <= fx+fw and fy <= y <= fy+fh:
                    blur_states[i] = not blur_states[i]
                    print(f"Face {i}: {'Blurred' if blur_states[i] else 'Unblurred'}")

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        frame = render_frame(img_original, faces, blurred_rois, blur_states)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            result = render_frame(img_original, faces, blurred_rois, blur_states)
            cv2.imwrite(output_path, result)
            print(f"Saved final image as {output_path}")
            break
        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    interactive_face_blur(r"C:\Documents\hashcode\images\image2.jpeg", "output_blurred.png")
