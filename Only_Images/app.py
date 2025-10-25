import os
# Set environment variables BEFORE importing transformers to prevent NumPy/TensorFlow conflicts
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import traceback
from PIL import Image
import io
import base64
import easyocr
from transformers import pipeline
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static frontend files from ./static (single mount)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: Avoid mounting /static twice to prevent route conflicts

# Initialize models once at startup
reader = None
ner_pipeline = None
pii_ner_pipeline = None
policy = None

@app.on_event("startup")
async def startup_event():
    global reader, ner_pipeline, pii_ner_pipeline, policy
    logger.info("Loading models...")
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        # Explicitly use PyTorch framework to avoid TensorFlow/Keras issues
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", framework="pt")
        # PII-specific NER model (token-classification)
        # Try multiple models in order of preference
        pii_models_to_try = [
            "Jean-Baptiste/roberta-large-ner-english",  # Publicly available, good for PII
            "dslim/bert-base-NER",  # Fallback to same as general NER
        ]
        pii_ner_pipeline = None
        for model_name in pii_models_to_try:
            try:
                logger.info(f"Attempting to load PII NER model: {model_name}")
                pii_ner_pipeline = pipeline("token-classification", model=model_name, aggregation_strategy="simple", framework="pt")
                logger.info(f"✓ PII NER model loaded successfully: {model_name}")
                break
            except Exception as e:
                logger.warning(f"✗ Failed to load {model_name}: {str(e)[:100]}")
                continue
        
        if pii_ner_pipeline is None:
            logger.warning("No PII NER model loaded, will rely on regex + context + general NER only")
        # Load policy config
        try:
            import json
            with open(os.path.join("config", "policy.json"), "r", encoding="utf-8") as f:
                policy = json.load(f)
            logger.info("Policy loaded")
        except Exception as e:
            policy = None
            logger.warning(f"Policy not found or invalid, using defaults: {e}")
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

# Import the processing function
from vision_guard import process_image

@app.get("/model-status")
async def model_status():
    """Check which models are loaded"""
    return {
        "easyocr": reader is not None,
        "ner_pipeline": ner_pipeline is not None,
        "pii_ner_pipeline": pii_ner_pipeline is not None,
        "policy_loaded": policy is not None,
    }

@app.post("/detect-pii/")
async def detect_pii(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return JSONResponse({
                "error": "Invalid file type. Please upload an image."
            }, status_code=400)

        # Read uploaded image
        input_bytes = await file.read()
        img = Image.open(io.BytesIO(input_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Save to temporary file
        temp_input_path = "temp_input.png"
        img.save(temp_input_path)
        
        temp_output_path = "temp_output.png"

        # Process image (detect sensitive info & blur regions)
        summary = process_image(
            temp_input_path,
            temp_output_path,
            reader,
            ner_pipeline,
            debug=True,  # Enable debug to see detailed detection info
            pii_ner_pipeline=pii_ner_pipeline,
            policy=policy,
        )

        boxes = summary["sensitive_boxes"]
        box_colors = []
        
        # Log detection scores for debugging
        logger.info(f"Detected {len(boxes)} sensitive regions:")
        for i, d in enumerate(summary.get("debug_flags", [])):
            score = d.get("score", "N/A")
            reasons = d.get("reasons", [])
            text = d.get("text", "")
            conf = d.get("conf", 0.0)
            logger.info(f"  [{i+1}] '{text}' - Reasons: {reasons}, Score: {score}, OCR Conf: {conf:.3f}")

        # Determine box colors based on type of sensitive data
        for d in summary.get("debug_flags", []):
            reasons = d.get("reasons", [])
            if any(r in ["EMAIL", "AADHAAR", "PAN", "CREDIT_CARD", "BANK_ACCOUNT", "ACCOUNT_ID", "PHONE"] for r in reasons):
                box_colors.append("red")
            else:
                box_colors.append("yellow")  # uncertain

        # Encode original image as base64
        buf = io.BytesIO()
        # Use the original image from summary
        if 'img_original' in summary:
            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(summary["img_original"], cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(buf, format="PNG")
        else:
            # Fallback to saved image
            img.save(buf, format="PNG")
        original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Encode blurred regions as base64 PNGs
        blurred_regions_b64 = []
        for region in summary.get("blurred_regions", []):
            if region is not None and region.size > 0:
                buf_region = io.BytesIO()
                # Convert BGR to RGB for PIL
                if len(region.shape) == 3 and region.shape[2] == 3:
                    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                    Image.fromarray(region_rgb).save(buf_region, format="PNG")
                else:
                    Image.fromarray(region).save(buf_region, format="PNG")
                blurred_regions_b64.append(base64.b64encode(buf_region.getvalue()).decode("utf-8"))

        # Clean up temporary files
        try:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except:
            pass

        return JSONResponse({
            "original_image": original_b64,
            "sensitive_boxes": boxes,
            "blurred_regions": blurred_regions_b64,
            "box_colors": box_colors,
            "debug_flags": summary.get("debug_flags", []),
            "model_status": {
                "pii_ner_loaded": pii_ner_pipeline is not None,
                "policy_loaded": policy is not None,
                "total_detections": len(boxes),
            }
        })
        
    except Exception as exc:
        # Log full traceback to server logs for debugging
        logger.error("Error in /detect-pii/: %s", exc)
        logger.error(traceback.format_exc())
        # Return structured error so front-end can display a helpful message
        return JSONResponse({
            "error": "Failed to process image. Check backend logs.",
            "detail": str(exc)
        }, status_code=500)

# Keep root redirect to frontend index
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/status")
async def get_status():
    """Check model loading status"""
    return {
        "status": "running",
        "models": {
            "easyocr": reader is not None,
            "general_ner": ner_pipeline is not None,
            "pii_ner": pii_ner_pipeline is not None,
            "policy": policy is not None,
        },
        "policy_config": policy if policy else "using defaults",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)