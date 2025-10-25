import uvicorn
import webbrowser
import threading
import time
import os

def start_frontend():
    # Wait for backend to start
    time.sleep(2)
    # Open the web browser
    webbrowser.open('http://127.0.0.1:8000/static/index.html')

if __name__ == "__main__":
    # Start frontend in a separate thread
    frontend_thread = threading.Thread(target=start_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    # Start backend
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)