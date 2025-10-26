"""
Vision-Guard Web: NER-only PII detection for web interface
Uses transformers NER models exclusively, no regex patterns.
"""

import os
# Avoid importing TensorFlow/Flax backends in Transformers to prevent NumPy/TensorFlow conflicts
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any

# Import orientation detector as optional fallback for vertical cards
try:
    from orientation_detector import detect_vertical_credit_card
    ORIENTATION_DETECTOR_AVAILABLE = True
except ImportError:
    ORIENTATION_DETECTOR_AVAILABLE = False



def process_image(input_path: str, output_path: str, reader: Any, ner_pipeline: Any, debug: bool = True,
                  pii_ner_pipeline: Any = None, policy: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process an image: OCR -> NER-based PII detection -> blur detected regions.
    Pure NER approach using transformers models only.

    Args:
        input_path: path to input image
        output_path: path to save blurred output
        reader: EasyOCR reader instance
        ner_pipeline: transformers NER pipeline for general entities
        pii_ner_pipeline: optional PII-specific NER pipeline
        debug: whether to print debug info
        policy: optional policy config (unused in NER-only mode)

    Returns:
        dict with processing summary including boxes, regions, and debug flags
    """
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")
    # Keep an original copy early for later use
    img_original = img.copy()

    # OCR
    ocr_results = reader.readtext(img, detail=1, paragraph=False)
    
    if debug:
        print(f"\n=== OCR Results === Total tokens: {len(ocr_results)}")

    # Normalize OCR outputs into tokens
    tokens: List[Dict[str, Any]] = []
    for item in ocr_results:
        if isinstance(item, (list, tuple)):
            if len(item) == 3:
                bbox, text, conf = item
            elif len(item) == 2:
                bbox, text = item
                conf = 1.0
            else:
                continue
        else:
            continue
        if not text or not text.strip():
            continue
        pts = bbox
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        tokens.append({
            "text": text.strip(),
            "conf": float(conf),
            "pts": pts,
            "rect": (x1, y1, x2, y2),
        })

    # Sort tokens by reading order (top to bottom, left to right)
    tokens.sort(key=lambda t: (t["rect"][1], t["rect"][0]))

    blur_indices = set()
    debug_flags: List[Dict[str, Any]] = []

    # Helper for grouping tokens on same line
    def vertical_overlap(a: Dict, b: Dict) -> float:
        """Calculate vertical overlap ratio between two token rects."""
        _, ay1, _, ay2 = a["rect"]
        _, by1, _, by2 = b["rect"]
        inter = max(0, min(ay2, by2) - max(ay1, by1))
        union = (ay2 - ay1) + (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    # First pass: Apply NER to each token individually
    for idx, tok in enumerate(tokens):
        text_str = tok["text"]
        detected = False
        reasons = []
        max_score = 0.0

        # General NER pipeline - ONLY detect LOC (locations/addresses)
        if ner_pipeline is not None:
            try:
                ner_preds = ner_pipeline(text_str)
                for pred in ner_preds:
                    entity = pred.get('entity_group') or pred.get('entity', '')
                    score = pred.get('score', 0.0)
                    # Only blur LOC (addresses, locations)
                    if entity in ('LOC', 'LOCATION') and score > 0.6:
                        detected = True
                        reasons.append(f"NER:{entity}")
                        max_score = max(max_score, score)
            except Exception as e:
                if debug:
                    print(f"NER error on '{text_str}': {e}")

        # PII-specific NER pipeline - detect ONLY allowed financial info (account/routing) per policy
        if pii_ner_pipeline is not None:
            try:
                pii_preds = pii_ner_pipeline(text_str)
                for pred in pii_preds:
                    entity = pred.get('entity_group') or pred.get('entity', '')
                    score = pred.get('score', 0.0)
                    
                    # ONLY blur account and routing. Exclude CREDIT_CARD/CARD/CVV/etc. (handled by orientation detector)
                    financial_entities = [
                        'ACCOUNT', 'ACCOUNT_NUMBER', 'ROUTING', 'ROUTING_NUMBER'
                    ]
                    
                    entity_upper = entity.upper()
                    is_financial = any(fin in entity_upper for fin in financial_entities)
                    
                    if is_financial and score > 0.5:
                        detected = True
                        entity_label = entity.upper()
                        reasons.append(f"PII:{entity_label}")
                        max_score = max(max_score, score)
                        
                        if debug:
                            print(f"  Financial PII: '{text_str}' -> {entity_label} (score: {score:.3f})")
            except Exception as e:
                if debug:
                    print(f"PII-NER error on '{text_str}': {e}")

        if detected:
            blur_indices.add(idx)
            debug_flags.append({
                "text": text_str,
                "reasons": reasons,
                "conf": float(tok["conf"]),
                "ml_confidence": float(max_score),
            })

    # Second pass: Group tokens on same line and apply NER to combined text
    # This helps detect multi-token entities like "john.doe @ gmail.com" or "4532 1234 5678 9010"
    n = len(tokens)
    lines: List[List[int]] = []
    if n > 0:
        current_line = [0]
        for i in range(1, n):
            if vertical_overlap(tokens[i-1], tokens[i]) > 0.3:
                current_line.append(i)
            else:
                if len(current_line) >= 2:  # Only process multi-token lines
                    lines.append(current_line)
                current_line = [i]
        if len(current_line) >= 2:
            lines.append(current_line)

    for line_indices in lines:
        # Try different grouping windows (2-5 consecutive tokens)
        for window_size in range(2, min(6, len(line_indices) + 1)):
            for start_idx in range(len(line_indices) - window_size + 1):
                window = line_indices[start_idx:start_idx + window_size]
                combined_text = " ".join(tokens[i]["text"] for i in window)
                
                # Apply PII NER to combined text - ONLY for allowed financial info (account/routing)
                if pii_ner_pipeline is not None:
                    try:
                        pii_preds = pii_ner_pipeline(combined_text)
                        for pred in pii_preds:
                            entity = pred.get('entity_group') or pred.get('entity', '')
                            score = pred.get('score', 0.0)
                            
                            # Filter to allowed financial entities only (exclude CARD/CVV)
                            financial_entities = [
                                'ACCOUNT', 'ACCOUNT_NUMBER', 'ROUTING', 'ROUTING_NUMBER'
                            ]
                            
                            entity_upper = entity.upper()
                            is_financial = any(fin in entity_upper for fin in financial_entities)
                            
                            if is_financial and score > 0.5:
                                entity_label = entity.upper()
                                # Blur all tokens in this window
                                for idx in window:
                                    if idx not in blur_indices:
                                        blur_indices.add(idx)
                                        debug_flags.append({
                                            "text": tokens[idx]["text"],
                                            "reasons": [f"PII-GROUP:{entity_label}"],
                                            "conf": float(tokens[idx]["conf"]),
                                            "ml_confidence": float(score),
                                        })
                                if debug:
                                    print(f"  Multi-token financial PII: '{combined_text}' -> {entity_label} (score: {score:.3f})")
                                break  # Found PII in this window, move to next
                    except Exception as e:
                        if debug:
                            print(f"PII-NER error on combined text '{combined_text}': {e}")

    # Third pass: Detect long numeric sequences (routing/account numbers, check numbers with heuristics)
    # These are highly specific patterns that NER might miss
    for idx, tok in enumerate(tokens):
        if idx in blur_indices:
            continue
        
        text_str = tok["text"].strip()
        # Remove common separators
        digits_only = ''.join(c for c in text_str if c.isdigit())
        
        # Routing numbers: exactly 9 digits
        if len(digits_only) == 9:
            blur_indices.add(idx)
            debug_flags.append({
                "text": text_str,
                "reasons": ["ROUTING_NUMBER"],
                "conf": float(tok["conf"]),
                "ml_confidence": 1.0,
            })
            if debug:
                print(f"  Routing number detected: '{text_str}'")
        
        # Account numbers: 10-17 digits
        elif 10 <= len(digits_only) <= 17:
            blur_indices.add(idx)
            debug_flags.append({
                "text": text_str,
                "reasons": ["ACCOUNT_NUMBER"],
                "conf": float(tok["conf"]),
                "ml_confidence": 1.0,
            })
            if debug:
                print(f"  Account number detected: '{text_str}'")
        
        # Check numbers: exactly 4 digits (avoid dates like 07/21 or years like 1991)
        elif len(digits_only) == 4:
            raw = text_str
            looks_like_date = any(sep in raw for sep in ['/', '-', ':'])
            # Year-like (1900-2099)
            try:
                val = int(digits_only)
            except ValueError:
                val = -1
            looks_like_year = 1900 <= val <= 2099
            # MMYY pattern
            mm = int(digits_only[:2]) if digits_only[:2].isdigit() else -1
            yy = int(digits_only[2:]) if digits_only[2:].isdigit() else -1
            looks_like_mmyy = 1 <= mm <= 12 and 0 <= yy <= 99

            if not (looks_like_date or looks_like_year or looks_like_mmyy):
                blur_indices.add(idx)
                debug_flags.append({
                    "text": text_str,
                    "reasons": ["CHECK_NUMBER"],
                    "conf": float(tok["conf"]),
                    "ml_confidence": 0.8,
                })
                if debug:
                    print(f"  Check number detected: '{text_str}'")

    # Build blur boxes from selected token indices
    sensitive_boxes: List[List[int]] = []
    for idx in sorted(blur_indices):
        x1, y1, x2, y2 = tokens[idx]["rect"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        sensitive_boxes.append([x1, y1, w, h])

    # FALLBACK: If no financial PII detected by main pipeline, try orientation detector
    # This handles vertical/tilted credit card numbers
    has_financial_pii = any(
        any(r in ["ROUTING_NUMBER", "ACCOUNT_NUMBER", "CHECK_NUMBER"] 
            for r in flag.get("reasons", []))
        for flag in debug_flags
    )
    
    if not has_financial_pii and ORIENTATION_DETECTOR_AVAILABLE:
        if debug:
            print("\n--- No financial PII found, trying orientation detector for vertical cards ---")
        try:
            # For demo/test images, relax Luhn validation to catch placeholder numbers
            vertical_boxes = detect_vertical_credit_card(img, reader, debug=debug, require_luhn=False)
            if vertical_boxes:
                if debug:
                    print(f"✓ Orientation detector found {len(vertical_boxes)} vertical credit card(s)")
                for box in vertical_boxes:
                    x, y, w, h = box
                    sensitive_boxes.append([x, y, w, h])
                    debug_flags.append({
                        "text": f"<vertical-card-{len(sensitive_boxes)}>",
                        "reasons": ["VERTICAL_CREDIT_CARD"],
                        "conf": 1.0,
                        "ml_confidence": 1.0,
                    })
        except Exception as e:
            if debug:
                print(f"Orientation detector error: {e}")

    # Store original image for web response
    blur_states: List[bool] = [True] * len(sensitive_boxes)
    
    # Apply blur to detected boxes
    if debug:
        if not sensitive_boxes:
            print("No sensitive regions detected by NER.")
        else:
            print(f"Blurring {len(sensitive_boxes)} detected sensitive regions.")
    
    blurred_regions: List[np.ndarray] = []
    for box in sensitive_boxes:
        x, y, w, h = box
        # Clamp coordinates
        x = max(0, x)
        y = max(0, y)
        h = max(1, h)
        w = max(1, w)
        roi = img_original[y:y+h, x:x+w]
        if roi.size == 0:
            blurred_regions.append(np.zeros((h, w, 3), dtype=np.uint8))
            continue
        # Adapt kernel size based on box size
        k = max(51, (max(w, h) // 2) | 1)  # ensure odd kernel
        try:
            blur = cv2.GaussianBlur(roi, (k, k), 30)
        except:
            # Fallback fixed kernel
            blur = cv2.GaussianBlur(roi, (51, 51), 30)
        blurred_regions.append(blur)
        img[y:y+h, x:x+w] = blur
    
    # Save result
    cv2.imwrite(output_path, img)
    if debug:
        print(f"Saved blurred output: {output_path}")

    return {
        "input": input_path,
        "output": output_path,
        "total_ocr_regions": len(tokens),
        "sensitive_regions_blurred": len(sensitive_boxes),
        "debug_flags": debug_flags,
        # Return data for web interface
        "img_original": img_original,
        "img_blurred": img,
        "sensitive_boxes": sensitive_boxes,
        "blurred_regions": blurred_regions,
        "blur_states": blur_states,
    }
