"""
Quick test script to verify backend compatibility without starting the full server
"""
import os
# Set environment variables BEFORE importing transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import sys
print("Loading dependencies...")

try:
    import cv2
    print("✓ OpenCV loaded")
except Exception as e:
    print(f"✗ OpenCV failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy loaded (version: {np.__version__})")
except Exception as e:
    print(f"✗ NumPy failed: {e}")
    sys.exit(1)

try:
    import easyocr
    print("✓ EasyOCR loaded")
except Exception as e:
    print(f"✗ EasyOCR failed: {e}")
    sys.exit(1)

try:
    from transformers import pipeline
    print("✓ Transformers loaded")
except Exception as e:
    print(f"✗ Transformers failed: {e}")
    sys.exit(1)

try:
    from vision_guard import process_image
    print("✓ vision_guard.process_image loaded")
except Exception as e:
    print(f"✗ vision_guard failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("Backend Compatibility Check: PASSED")
print("="*50)
print("\nAll dependencies loaded successfully!")
print("The FastAPI backend should work correctly.")
print("\nTo start the server, run:")
print("  python app.py")
