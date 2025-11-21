import cv2
import numpy as np
import mss
import os
from flask import Flask, render_template, Response, request, jsonify
import base64

app = Flask(__name__)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_image(img, blur=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if blur:
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_blur = cv2.GaussianBlur(face, (25, 25), 30)
            img[y:y+h, x:x+w] = face_blur

    return img

def capture_tab(tab_id):
    with mss.mss() as sct:
        monitor = sct.monitors[tab_id]
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_tab', methods=['POST'])
def capture_tab_route():
    tab_id = int(request.form['tab_id'])
    blur = request.form.get('blur', False) == 'on'
    img = capture_tab(tab_id)
    processed_img = process_image(img, blur)

    # Convert image to base64 for sharing
    _, encoded = cv2.imencode('.jpg', processed_img)
    base64_image = base64.b64encode(encoded).decode('utf-8')

    return jsonify({
        'image': base64_image
    })

if __name__ == '__main__':
    app.run(debug=True)