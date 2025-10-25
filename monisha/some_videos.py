from deepface import DeepFace
import cv2
import numpy as np

import glob, os
from deepface import DeepFace

targets = []
for ext in ["jpg", "jpeg", "png", "bmp"]:
    for path in glob.glob(f"faces_to_blur/*.{ext}"):
        emb = DeepFace.represent(img_path=path, model_name="Facenet")[0]['embedding']
        targets.append(emb)

cap = cv2.VideoCapture("video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter("C:\\Documents\\hashcode\\blur_selective.avi", fourcc, fps, (w, h))

# fourcc = cv2.VideoWriter_fourcc(*'avc1')
# out = cv2.VideoWriter("C:\\Documents\\hashcode\\blur_selective.mp4", fourcc, fps, (w, h))

out = cv2.VideoWriter("blur_selective.mp4", cv2.VideoWriter_fourcc(*'X264'), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret: break

    # detect faces
    detections = DeepFace.extract_faces(img_path=frame, detector_backend='opencv', enforce_detection=False)
    for d in detections:
        (x, y, w, h) = (d["facial_area"]["x"], d["facial_area"]["y"], d["facial_area"]["w"], d["facial_area"]["h"])
        face_crop = frame[y:y+h, x:x+w]
        try:
            rep = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]['embedding']
            # cosine similarity to our target encodings
            def cosine_sim(a,b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
            sims = [cosine_sim(rep, t) for t in targets]
            if max(sims) > 0.4:   # threshold (~0.3–0.5)
                frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_crop, (55,55), 30)
        except: pass

    out.write(frame)
cap.release()
out.release()
