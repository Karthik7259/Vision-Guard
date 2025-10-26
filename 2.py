import cv2
import numpy as np
import mss
import pyvirtualcam
import pytesseract
import re
from PIL import Image

# -------- SCREEN SETTINGS --------
MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# -------- DNN FACE DETECTOR --------
PROTOTXT = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# -------- REGEX PATTERNS FOR SENSITIVE INFO --------
SENSITIVE_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
    'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'password': r'\b(password|pwd|pass|secret|key)[:=]\s*\S+\b',
    'api_key': r'\b(api[_-]?key|access[_-]?token|secret[_-]?key)[:=]\s*\S+\b',
    'bank_account': r'\b\d{8,17}\b',
    'routing_number': r'\b\d{9}\b',
    'dob': r'\b(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b',
    'passport': r"\b[A-Z]{1,2}\d{6,9}\b",
    'aadhaar': r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    'pan': r"\b[A-Z]{5}\d{4}[A-Z]\b",
    'driving_license': r"\b[A-Z]{2}\d{2}[0-9A-Z]{11}\b",
    'vin': r"\b[A-HJ-NPR-Z0-9]{17}\b"
}

KEYWORDS_NEXT_VALUE = ["password", "pwd", "secret", "api_key", "token", "access_key"]

def blackout_region(frame, x1, y1, x2, y2):
    frame[y1:y2, x1:x2] = 0
    return frame

def blur_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            frame = blackout_region(frame, max(0,x1), max(0,y1), min(w,x2), min(h,y2))
    return frame

def is_sensitive_text(text):
    text = text.strip()
    if len(text) < 3:
        return False
    for pattern in SENSITIVE_PATTERNS.values():
        if re.search(pattern, text, re.IGNORECASE):
            return True
    # Also check if keyword + value pattern exists
    if any(k in text.lower() for k in KEYWORDS_NEXT_VALUE):
        return True
    return False

def blackout_sensitive_text_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)
    ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    n_boxes = len(ocr_data['text'])
    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])
        if conf > 30 and is_sensitive_text(text):
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            padding = 10
            frame = blackout_region(frame, max(0,x-padding), max(0,y-padding), min(frame.shape[1],x+w+padding), min(frame.shape[0],y+h+padding))
        # Optional: blur next token if keyword detected
        if any(k in text.lower() for k in KEYWORDS_NEXT_VALUE) and i+1<n_boxes:
            nx, ny, nw, nh = ocr_data['left'][i+1], ocr_data['top'][i+1], ocr_data['width'][i+1], ocr_data['height'][i+1]
            padding = 10
            frame = blackout_region(frame, max(0,nx-padding), max(0,ny-padding), min(frame.shape[1],nx+nw+padding), min(frame.shape[0],ny+nh+padding))
    return frame

def process_frame(frame):
    frame = blur_faces(frame)
    frame = blackout_sensitive_text_ocr(frame)
    return frame

def main():
    print("🎥 Vision Guard Virtual Cam Active")
    try:
        with mss.mss() as sct, pyvirtualcam.Camera(width=1920, height=1080, fps=20) as cam:
            print(f"✅ Virtual camera: {cam.device}")
            while True:
                img = np.array(sct.grab(MONITOR))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                processed = process_frame(frame)
                cam.send(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()
    except KeyboardInterrupt:
        print("🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
