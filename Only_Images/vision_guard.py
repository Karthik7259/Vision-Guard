"""
Vision-Guard Web: Advanced PII detection for web interface
Adapted from root vision_guard.py with structured detectors + transformers NER.
"""

import os
# Avoid importing TensorFlow/Flax backends in Transformers to prevent NumPy/TensorFlow conflicts
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import cv2
import numpy as np
from PIL import Image
import re
from typing import List, Tuple, Dict, Any, Set


# ----------------------------
# Helper detectors (top-level)
# ----------------------------

def luhn_check(card_number: str) -> bool:
    """Luhn algorithm for credit card validation."""
    digits = [int(d) for d in re.sub(r'\D', '', card_number)]
    if len(digits) < 12:  # not a valid card length
        return False
    checksum = 0
    alt = False
    for d in reversed(digits):
        if alt:
            d = d * 2
            if d > 9:
                d -= 9
        checksum += d
        alt = not alt
    return checksum % 10 == 0


# Common patterns (India-focused examples included)
PATTERNS = {
    # Emails
    "email": re.compile(r"\b[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"),
    "aadhaar": re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b'),  # 12 digits grouped or ungrouped
    "pan": re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b', re.IGNORECASE),  # PAN format
    "credit_card": re.compile(r'\b(?:\d[ -]*?){13,19}\b'),  # 13-19 digits with optional spaces/hyphens
    # Bank account numbers vary widely; this is a heuristic for long numeric sequences
    "bank_account": re.compile(r'\b\d{9,17}\b'),
    # Money amounts
    "amount": re.compile(r'\$\s*[\d,]+(?:\.\d{2})?'),
    # Check numbers (typically 4 digits)
    "check_number": re.compile(r'\b\d{4}\b'),
    # Indian vehicle number plate common pattern
    "veh_plate_india": re.compile(r'\b[A-Z]{2}\s?-?\s?\d{1,2}\s?-?\s?[A-Z]{1,3}\s?-?\s?\d{1,4}\b', re.IGNORECASE),
    # Generic phone numbers (simple)
    "phone": re.compile(r'(\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){6,12}\d'),
    # Passport (simple heuristic)
    "passport": re.compile(r'\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b'),
}

# Labelled account/id (e.g., "Account ID: 456789")
ACCOUNT_LABEL_REGEX = re.compile(
    r"\b(?:account|acct|a/c|acc(?:ount)?)\s*(?:id|no|number|#)?\s*[:#-]?\s*([A-Za-z0-9-]{4,18})\b",
    re.IGNORECASE,
)
GENERIC_ID_WITH_LABEL = re.compile(
    r"\b(?:id|user\s*id|emp\s*id|ref(?:erence)?\s*id)\s*[:#-]?\s*([A-Za-z0-9-]{4,18})\b",
    re.IGNORECASE,
)


def matches_structured_patterns(text: str) -> Tuple[bool, str]:
    """
    Check if text matches structured PII patterns.
    Returns (is_match, reason_string)
    """
    t = text.strip()

    # Email
    if PATTERNS["email"].search(t):
        return True, "EMAIL"

    # Aadhaar
    if PATTERNS["aadhaar"].search(t):
        digits = re.sub(r'\D', '', t)
        if len(digits) == 12:
            return True, "AADHAAR"

    # PAN
    if PATTERNS["pan"].search(t):
        return True, "PAN"

    # Money amounts
    if PATTERNS["amount"].search(t):
        return True, "AMOUNT"

    # Credit card
    cc_match = PATTERNS["credit_card"].search(t)
    if cc_match:
        digits = re.sub(r'\D', '', cc_match.group())
        if luhn_check(digits):
            return True, "CREDIT_CARD"
        if len(digits) >= 13:
            return True, "POSSIBLE_CARD_OR_ACCOUNT"

    # Bank account/routing heuristic (long digit sequences)
    ba_match = PATTERNS["bank_account"].search(t)
    if ba_match:
        digits = ba_match.group()
        digits_clean = re.sub(r'\D', '', digits)
        if len(digits_clean) >= 9 and len(digits_clean) != 12:  # not Aadhaar
            return True, "BANK_ACCOUNT"

    # Account/ID labels with shorter values
    m = ACCOUNT_LABEL_REGEX.search(t) or GENERIC_ID_WITH_LABEL.search(t)
    if m:
        val = m.group(1)
        if 4 <= len(val) <= 18:
            digits_only = re.sub(r'\D', '', val)
            if len(digits_only) >= 4:
                return True, "ACCOUNT_ID"

    # Vehicle plate (India-ish)
    if PATTERNS["veh_plate_india"].search(t):
        return True, "VEHICLE_PLATE"

    # Phone
    if PATTERNS["phone"].search(t):
        digits = re.sub(r'\D', '', t)
        if 7 <= len(digits) <= 15:
            return True, "PHONE"

    # Passport (optional heuristic)
    if PATTERNS.get("passport") and PATTERNS["passport"].search(t):
        return True, "PASSPORT"

    return False, None


# ----------------------------
# Main processing function
# ----------------------------

def process_image(input_path: str, output_path: str, reader: Any, ner_pipeline: Any, debug: bool = True,
                  pii_ner_pipeline: Any = None, policy: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process an image: OCR -> classify sensitive text -> blur detected regions.
    This is the robust hybrid version: structured detectors (regex + validators) + token-level NER for names.

    Extra params (pii_ner_pipeline, policy) are accepted for compatibility but ignored here.

    Args:
        input_path: path to input image
        output_path: path to save blurred output
        reader: EasyOCR reader instance
        ner_pipeline: transformers NER pipeline instance (for PERSON)
        debug: whether to print debug info

    Returns:
        dict with processing summary including boxes, regions, and debug flags
    """
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")

    # OCR -> tokenize (word-level) to allow selective span blurring
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

    # Sort tokens roughly reading order (by y, then x)
    tokens.sort(key=lambda t: (t["rect"][1], t["rect"][0]))

    blur_indices: Set[int] = set()
    debug_flags: List[Dict[str, Any]] = []
    n = len(tokens)
    
    # Helper fns for line grouping
    def vertical_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
        _, ay1, _, ay2 = a
        _, by1, _, by2 = b
        inter = max(0, min(ay2, by2) - max(ay1, by1))
        union = (ay2 - ay1) + (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def same_line(i: int, j: int) -> bool:
        return vertical_iou(tokens[i]["rect"], tokens[j]["rect"]) > 0.3

    def is_email_piece(s: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9._+-]+|[._-]", s)) or ("@" in s)

    # 1) Emails: expand around token with '@' and validate
    for i in range(n):
        t = tokens[i]["text"]
        if "@" not in t:
            continue
        L = i - 1
        while L >= 0 and same_line(i, L) and is_email_piece(tokens[L]["text"]):
            L -= 1
        L += 1
        R = i + 1
        while R < n and same_line(i, R) and is_email_piece(tokens[R]["text"]):
            R += 1
        R -= 1
        group = list(range(L, R + 1))
        joined_raw = "".join(tokens[k]["text"] for k in group)
        normalized = joined_raw.replace(" ", "").replace("_", ".").replace("•", ".").replace("·", ".")
        if PATTERNS["email"].search(normalized):
            for k in group:
                if k not in blur_indices:
                    blur_indices.add(k)
                    debug_flags.append({"text": tokens[k]["text"], "reasons": ["EMAIL"], "conf": tokens[k]["conf"]})

    # 2) Account ID labels: blur only the value, not the label
    lines: List[List[int]] = []
    if n:
        current = [0]
        for i in range(1, n):
            if same_line(i - 1, i):
                current.append(i)
            else:
                lines.append(current)
                current = [i]
        if current:
            lines.append(current)

    for line in lines:
        # Build a space-joined string (for regex search)
        line_texts = [tokens[idx]["text"] for idx in line]
        line_str = " ".join(line_texts)

        for rx in (ACCOUNT_LABEL_REGEX, GENERIC_ID_WITH_LABEL):
            for m in rx.finditer(line_str):
                g1 = m.group(1)
                if not g1:
                    continue
                # Prefer exact match to a single token equal to the captured value
                target_idx = None
                for idx in line:
                    tok_val = tokens[idx]["text"].strip()
                    tok_core = re.sub(r'^[^A-Za-z0-9-]+|[^A-Za-z0-9-]+$', '', tok_val)
                    if tok_val == g1 or tok_core == g1:
                        target_idx = idx
                        break
                if target_idx is not None and target_idx not in blur_indices:
                    blur_indices.add(target_idx)
                    debug_flags.append({"text": tokens[target_idx]["text"], "reasons": ["ACCOUNT_ID"], "conf": tokens[target_idx]["conf"]})

        # Minimal rule: if a pure digit token appears on a line that contains an 'account' label, blur it
        has_account_label = any(("account" in tokens[i]["text"].lower()) for i in line)
        if has_account_label:
            for idx in line:
                if re.fullmatch(r"\d{4,18}", tokens[idx]["text"].strip()):
                    if idx not in blur_indices:
                        blur_indices.add(idx)
                        debug_flags.append({"text": tokens[idx]["text"], "reasons": ["ACCOUNT_ID"], "conf": tokens[idx]["conf"]})

    # 3) Phone numbers: group sequences of phone-like tokens; blur digits but keep country code token
    i = 0
    while i < n:
        def phone_like(s: str) -> bool:
            return bool(re.fullmatch(r"[\+\d\s\-()]+", s))

        if not phone_like(tokens[i]["text"]):
            i += 1
            continue
        group = [i]
        j = i + 1
        while j < n and same_line(i, j) and phone_like(tokens[j]["text"]):
            group.append(j)
            j += 1
        digits = re.sub(r"\D", "", "".join(tokens[k]["text"] for k in group))
        if 7 <= len(digits) <= 15:
            cc_idx = None
            for k in group:
                tok_norm = tokens[k]["text"].strip().replace(" ", "")
                if re.fullmatch(r"\+?\d{1,3}", tok_norm) or re.fullmatch(r"\(\+?\d{1,3}\)", tok_norm):
                    cc_idx = k
                    break
            for k in group:
                if k == cc_idx:
                    continue
                if re.search(r"\d", tokens[k]["text"]):
                    if k not in blur_indices:
                        blur_indices.add(k)
                        debug_flags.append({"text": tokens[k]["text"], "reasons": ["PHONE"], "conf": tokens[k]["conf"]})
        i = j

    # 4) Other structured patterns per token (Aadhaar, PAN, CC, bank acct, veh plate, passport)
    for idx, tok in enumerate(tokens):
        text_str = tok["text"]
        matched, reason = matches_structured_patterns(text_str)
        if matched and reason not in {"EMAIL", "PHONE"}:  # avoid double-report
            if idx not in blur_indices:
                blur_indices.add(idx)
                debug_flags.append({"text": text_str, "reasons": [reason], "conf": tok["conf"]})

        # catch long numeric strings (routing/account numbers)
        core_digits = re.sub(r'\D', '', text_str)
        if 9 <= len(core_digits) <= 17:
            if idx not in blur_indices:
                blur_indices.add(idx)
                debug_flags.append({"text": text_str, "reasons": ["BANK_ACCOUNT"], "conf": tok["conf"]})

        # Check numbers (exactly 4 digits) – careful to avoid addresses
        if len(core_digits) == 4 and re.fullmatch(r'\d{4}', text_str.strip()):
            is_likely_address = False
            window_start = max(0, idx - 2)
            window_end = min(len(tokens), idx + 3)
            for j in range(window_start, window_end):
                if j == idx:
                    continue
                nearby = tokens[j]["text"].lower()
                if any(addr_word in nearby for addr_word in ["ny", "ca", "lane", "street", "st", "blvd", "ave", "road", "zip"]):
                    is_likely_address = True
                    break
            if not is_likely_address and idx not in blur_indices:
                blur_indices.add(idx)
                debug_flags.append({"text": text_str, "reasons": ["CHECK_NUMBER"], "conf": tok["conf"]})

    # 5) NER (names/orgs). Apply per-token to avoid blurring labels like 'Account'. Restrict to PERSON.
    for idx, tok in enumerate(tokens):
        if idx in blur_indices:
            continue
        text_str = tok["text"]
        try:
            ner_preds = ner_pipeline(text_str) if ner_pipeline is not None else []
        except Exception:
            ner_preds = []
        if ner_preds:
            entities = [p.get('entity_group') or p.get('entity') for p in ner_preds]
            if any(g in ("PER", "PERSON") for g in entities):
                blur_indices.add(idx)
                debug_flags.append({"text": text_str, "reasons": ["NER:PER"], "conf": tok["conf"]})

    # ---------------------------------------------
    # Re-introduce credit card detection (regex+Luhn)
    # ---------------------------------------------
    def luhn_check(number: str) -> bool:
        digits = [int(d) for d in re.sub(r"\D", "", number)]
        if len(digits) < 13 or len(digits) > 19:
            return False
        checksum = 0
        alt = False
        for d in reversed(digits):
            if alt:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
            alt = not alt
        return checksum % 10 == 0

    def vertical_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
        _, ay1, _, ay2 = a
        _, by1, _, by2 = b
        inter = max(0, min(ay2, by2) - max(ay1, by1))
        union = (ay2 - ay1) + (by2 - by1) - inter
        return inter/union if union > 0 else 0.0

    def same_line(i: int, j: int) -> bool:
        return vertical_iou(tokens[i]["rect"], tokens[j]["rect"]) > 0.3

    # 1) Per-token CC check
    for i, tok in enumerate(tokens):
        s = tok["text"]
        digits_only = re.sub(r"\D", "", s)
        if 13 <= len(digits_only) <= 19 and luhn_check(digits_only):
            if i not in blur_indices:
                blur_indices.add(i)
                debug_flags.append({
                    "text": s,
                    "reasons": ["CREDIT_CARD"],
                    "conf": tok.get("conf", 0.0),
                    "ml_confidence": 1.0,
                    "method": "single-token"
                })

    # 2) Line-based grouping of digit chunks with spaces/dashes
    # Build lines of tokens using vertical overlap
    lines: List[List[int]] = []
    if n:
        current = [0]
        for i in range(1, n):
            if same_line(i - 1, i):
                current.append(i)
            else:
                lines.append(current)
                current = [i]
        lines.append(current)

    def is_cc_piece(s: str) -> bool:
        return bool(re.fullmatch(r"[0-9\-\s]{1,}", s))

    for line in lines:
        # scan contiguous windows of cc-like pieces
        i = 0
        L = len(line)
        while i < L:
            if not is_cc_piece(tokens[line[i]]["text"]):
                i += 1
                continue
            j = i
            while j < L and is_cc_piece(tokens[line[j]]["text"]):
                j += 1
            # window = line[i:j]
            if j - i >= 2:  # at least two parts
                joined = "".join(tokens[k]["text"] for k in line[i:j])
                digits = re.sub(r"\D", "", joined)
                if 13 <= len(digits) <= 19 and luhn_check(digits):
                    for k in line[i:j]:
                        if k not in blur_indices:
                            blur_indices.add(k)
                            debug_flags.append({
                                "text": tokens[k]["text"],
                                "reasons": ["CREDIT_CARD"],
                                "conf": tokens[k].get("conf", 0.0),
                                "ml_confidence": 1.0,
                                "method": "line-group"
                            })
            i = j

    # Build blur boxes from selected token indices
    sensitive_boxes: List[List[int]] = []
    for idx in sorted(blur_indices):
        x1, y1, x2, y2 = tokens[idx]["rect"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        sensitive_boxes.append([x1, y1, w, h])

    # NER-only mode: no extra credit-card merging
    
    # Store original image and blurred regions
    img_original = img.copy()
    blur_states: List[bool] = [True] * len(sensitive_boxes)
    
    # APPLY BLUR to detected boxes
    if debug:
        if not sensitive_boxes:
            print("No sensitive boxes detected by the pipeline.")
        else:
            print(f"Blurring {len(sensitive_boxes)} detected sensitive regions.")
    
    blurred_regions: List[np.ndarray] = []
    for box in sensitive_boxes:
        x, y, w, h = box
        # clamp coordinates
        x = max(0, x)
        y = max(0, y)
        h = max(1, h)
        w = max(1, w)
        roi = img_original[y:y+h, x:x+w]
        if roi.size == 0:
            blurred_regions.append(np.zeros((h, w, 3), dtype=np.uint8))
            continue
        # adapt kernel size based on box size
        k = max(51, (max(w, h) // 2) | 1)  # ensure odd kernel
        try:
            blur = cv2.GaussianBlur(roi, (k, k), 30)
        except:
            # fallback fixed kernel
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