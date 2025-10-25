import cv2
import numpy as np
import mss
from flask import Flask, Response
import threading
import socket

# -------- SCREEN SETTINGS --------
MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# -------- DNN FACE DETECTOR --------
PROTOTXT = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
BLUR_KERNEL_SIZE = (99, 99)
BLUR_SIGMA = 30

# Initialize Flask app
app = Flask(__name__)

# Global variables
current_frame = None
frame_lock = threading.Lock()
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

def blur_faces(frame):
    """
    Detect and blur faces in the given frame
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        1.0,
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                blurred = cv2.GaussianBlur(face, BLUR_KERNEL_SIZE, BLUR_SIGMA)
                frame[y1:y2, x1:x2] = blurred

    return frame

def get_local_ip():
    """Get local IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def capture_screen():
    """Capture screen and process frames"""
    global current_frame
    
    with mss.mss() as sct:
        while True:
            # Capture screen
            img = np.array(sct.grab(MONITOR))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Process frame
            processed = blur_faces(frame)
            
            # Update global frame
            with frame_lock:
                current_frame = processed.copy()
            
            # Show preview
            cv2.imshow("Blurred Screen (Press 'q' to quit)", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

def generate_frames():
    """Generate frames for streaming"""
    global current_frame
    while True:
        with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Return the streaming page"""
    return """
    <html>
    <head>
        <title>Live Blurred Stream</title>
        <style>
            body { margin: 0; background: #000; }
            .video-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            img { max-width: 100%; max-height: 100vh; }
        </style>
    </head>
    <body>
        <div class="video-container">
            <img src="/video_feed">
        </div>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    local_ip = get_local_ip()
    port = 5000
    
    print("\n=== Vision Guard Streaming Server ===")
    print(f"🌐 Local stream URL: http://{local_ip}:{port}")
    print("📱 Share this URL with devices on your network")
    print("🎥 Preview window will open on this computer")
    print("❌ Press 'q' in the preview window to quit\n")
    
    # Start screen capture in a separate thread
    capture_thread = threading.Thread(target=capture_screen)
    capture_thread.daemon = True
    capture_thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    main()