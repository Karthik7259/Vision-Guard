"""
Orientation-agnostic credit card detector for vertical/tilted cards.
Uses multi-angle OCR sweeps + spatial clustering + Luhn validation.
Designed as a standalone module - does NOT modify existing detection logic.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.cluster import DBSCAN
import re


def luhn_check(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in re.sub(r'\D', '', card_number)]
    if len(digits) < 13 or len(digits) > 19:
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


def rotate_image(img: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image by given angle (degrees) around center.
    Returns rotated image and rotation matrix.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, M


def inverse_transform_box(bbox: Tuple[int, int, int, int], M: np.ndarray, 
                         original_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Transform bounding box back to original image coordinates.
    bbox: (x, y, w, h) in rotated space
    M: rotation matrix
    original_shape: (height, width) of original image
    """
    x, y, w, h = bbox
    # Get 4 corners of rotated box
    corners = np.array([
        [x, y, 1],
        [x + w, y, 1],
        [x + w, y + h, 1],
        [x, y + h, 1]
    ], dtype=np.float32).T
    
    # Invert the transformation
    M_inv = cv2.invertAffineTransform(M)
    transformed = M_inv @ corners
    
    # Get bounding box in original space
    xs = transformed[0, :]
    ys = transformed[1, :]
    x1, y1 = int(np.min(xs)), int(np.min(ys))
    x2, y2 = int(np.max(xs)), int(np.max(ys))
    
    # Clamp to image bounds
    h_orig, w_orig = original_shape
    x1 = max(0, min(x1, w_orig - 1))
    y1 = max(0, min(y1, h_orig - 1))
    x2 = max(0, min(x2, w_orig - 1))
    y2 = max(0, min(y2, h_orig - 1))
    
    return (x1, y1, x2 - x1, y2 - y1)


def estimate_skew_angle(img: np.ndarray, debug: bool = False) -> float:
    """
    Estimate text skew angle in degrees to deskew the image (make text horizontal).
    Positive angle means text is rotated clockwise.
    Returns angle in degrees in approximately [-45, 45].
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        # If already grayscale
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement for better edge detection
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    angles: List[float] = []
    if lines is not None:
        for rho_theta in lines:
            rho, theta = rho_theta[0]
            # Convert to angle relative to horizontal; theta=90deg implies horizontal line -> angle 0
            angle = float(np.degrees(theta) - 90.0)
            # Normalize to [-90, 90]
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # Prefer small tilts, map to [-45, 45]
            if angle > 45:
                angle -= 90
            if angle < -45:
                angle += 90
            angles.append(angle)

    # Fallback using minAreaRect on binarized mask if Hough failed
    if not angles:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = 255 - th  # text white
        coords = cv2.findNonZero(th)
        if coords is not None:
            rect = cv2.minAreaRect(coords)
            angle = float(rect[2])  # in (-90, 0]
            if angle < -45:
                angle += 90
            angles = [angle]

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return 0.0
    if debug:
        print(f"[deskew] estimated skew angle: {median_angle:.2f}°")
    return median_angle


def extract_digit_tokens(ocr_results: List[Any]) -> List[Dict[str, Any]]:
    """Extract tokens that are single digits or digit sequences."""
    digit_tokens = []
    for item in ocr_results:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            bbox, text = item[0], item[1]
            conf = item[2] if len(item) > 2 else 1.0
            
            # Keep only tokens with digits
            if re.search(r'\d', text):
                pts = bbox
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                digit_tokens.append({
                    'text': text,
                    'conf': float(conf),
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'center': (cx, cy),
                })
    return digit_tokens


def cluster_nearby_digits(tokens: List[Dict[str, Any]], eps: float = 80) -> List[List[Dict[str, Any]]]:
    """
    Use DBSCAN to cluster digit tokens that are spatially close.
    Returns list of clusters (each cluster is a list of tokens).
    """
    if len(tokens) < 2:  # Lowered from 3
        return []
    
    # Use token centers as features
    centers = np.array([t['center'] for t in tokens])
    
    # DBSCAN clustering with more lenient parameters
    clustering = DBSCAN(eps=eps, min_samples=2).fit(centers)  # Lowered from min_samples=3
    labels = clustering.labels_
    
    # Group tokens by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:  # noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tokens[idx])
    
    return list(clusters.values())


def order_tokens_in_cluster(cluster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Order tokens within a cluster using PCA to find the main axis,
    then sort along that axis.
    """
    if len(cluster) <= 1:
        return cluster
    
    centers = np.array([t['center'] for t in cluster])
    
    # PCA to find principal axis
    mean = centers.mean(axis=0)
    centered = centers - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Principal component (direction of max variance)
    pc1 = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Project centers onto principal axis
    projections = centered @ pc1
    
    # Sort by projection
    sorted_indices = np.argsort(projections)
    return [cluster[i] for i in sorted_indices]


def extract_cc_number_from_cluster(cluster: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract credit card number from ordered cluster of tokens.
    Returns number if valid (13-19 digits), else None.
    """
    ordered = order_tokens_in_cluster(cluster)
    combined = ''.join(t['text'] for t in ordered)
    digits = re.sub(r'\D', '', combined)
    
    # First try: Look for 4-digit groups (typical CC format: XXXX XXXX XXXX XXXX)
    # This handles cases like ['1234', '5678', '9012', '3456']
    four_digit_groups = []
    for t in ordered:
        # Extract clean digit sequences
        seq = re.sub(r'\D', '', t['text'])
        if len(seq) == 4:  # Exact 4-digit match
            # Skip if it looks like a date (0125, 0422, etc. - month/year patterns)
            if seq[:2] in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] and seq[2:4] <= '31':
                continue
            four_digit_groups.append(seq)
    
    if len(four_digit_groups) >= 4:
        # Combine the first 4 groups to make a 16-digit card
        candidate = ''.join(four_digit_groups[:4])
        return candidate
    
    # Second try: Extract all digit sequences and filter out short ones (likely dates)
    digit_sequences = []
    for t in ordered:
        seq = re.sub(r'\D', '', t['text'])
        if len(seq) >= 3:  # Skip very short sequences
            digit_sequences.append(seq)
    
    # Filter out common date patterns (4 digits that might be MM/YY or MM/DD)
    filtered_sequences = []
    for seq in digit_sequences:
        # Skip if it looks like a date (0125, 0422, etc. - month/year patterns)
        if len(seq) == 4 and seq[:2] in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            continue
        filtered_sequences.append(seq)
    
    if filtered_sequences:
        combined_filtered = ''.join(filtered_sequences)
        if 13 <= len(combined_filtered) <= 19:
            return combined_filtered
    
    # Third try: Extract 13-19 digit sequences from raw digit string
    if len(digits) >= 13:
        # Try to find a valid 13-19 digit substring
        for start in range(len(digits)):
            for length in range(19, 12, -1):  # Try longest first
                if start + length <= len(digits):
                    candidate = digits[start:start+length]
                    if 13 <= len(candidate) <= 19:
                        return candidate
    
    return None


def detect_vertical_credit_card(img: np.ndarray, reader: Any, debug: bool = False, 
                               require_luhn: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Multi-angle OCR sweep to detect vertical/tilted credit card numbers.
    
    Strategy:
    1. Rotate image at 0°, 90°, 180°, 270° (and optionally ±45°)
    2. Run OCR on each rotation
    3. Extract digit tokens and cluster spatially nearby ones
    4. For each cluster, order tokens and validate as credit card (Luhn)
    5. Transform valid boxes back to original orientation
    
    Args:
        img: Input image
        reader: EasyOCR reader instance
        debug: Print debug information
        require_luhn: If True, only accept numbers that pass Luhn validation.
                     If False, accept any 13-19 digit sequence (useful for demo/test cards)
    
    Returns list of bounding boxes in original image coordinates.
    """
    original_img = img
    original_shape = original_img.shape[:2]
    base_angles = [0, 90, 180, 270, 45, -45, 135, -135]

    # Build candidate pre-rotations: original + optional deskewed
    candidates: List[Tuple[np.ndarray, Optional[np.ndarray], str]] = [(original_img, None, "original")]
    try:
        skew = estimate_skew_angle(original_img, debug=debug)
    except Exception:
        skew = 0.0
    if abs(skew) >= 0.5:
        deskewed_img, pre_M = rotate_image(original_img, -skew)
        candidates.append((deskewed_img, pre_M, f"deskewed({-skew:.1f}°)"))

    all_detections: List[Tuple[int, int, int, int]] = []

    for work_img, pre_M, label in candidates:
        work_shape = work_img.shape[:2]
        if debug and label != "original":
            print(f"\n[deskew] Trying candidate: {label}")

        for angle in base_angles:
            if debug:
                print(f"\n--- Trying rotation: {angle}° ---")

            # Rotate candidate image by current angle
            rotated, M_rot = rotate_image(work_img, angle)

            # Enhance for OCR
            enhanced = cv2.convertScaleAbs(rotated, alpha=1.5, beta=10)

            # OCR
            try:
                ocr_results = reader.readtext(enhanced, detail=1, paragraph=False,
                                             min_size=10, text_threshold=0.5)
            except Exception as e:
                if debug:
                    print(f"OCR failed at {angle}°: {e}")
                continue

            # Digit tokens
            digit_tokens = extract_digit_tokens(ocr_results)
            if debug:
                print(f"Found {len(digit_tokens)} digit tokens")
                if digit_tokens:
                    print(f"Sample tokens: {[t['text'] for t in digit_tokens[:10]]}")
            if len(digit_tokens) < 10:
                continue

            # Cluster and extract
            for eps in [120, 100, 80, 60]:
                clusters = cluster_nearby_digits(digit_tokens, eps=eps)
                for cluster in clusters:
                    if len(cluster) < 4:
                        continue

                    cc_number = extract_cc_number_from_cluster(cluster)
                    if debug and cc_number:
                        print(f"  Extracted candidate: {cc_number} (len={len(cc_number)})")

                    if cc_number and len(cc_number) >= 13:
                        is_valid = True
                        if require_luhn:
                            is_valid = luhn_check(cc_number)
                            if debug and not is_valid:
                                print("    ✗ Failed Luhn validation")
                        if not is_valid:
                            continue

                        if debug:
                            validation_status = "(Luhn ✓)" if require_luhn else "(Luhn not required)"
                            print(f"✓ Valid CC found at {angle}° (eps={eps}): {cc_number} {validation_status}")

                        # Compute bounding box in rotated coords
                        xs = [t['bbox'][0] for t in cluster] + [t['bbox'][0] + t['bbox'][2] for t in cluster]
                        ys = [t['bbox'][1] for t in cluster] + [t['bbox'][1] + t['bbox'][3] for t in cluster]
                        x_min, y_min = int(min(xs)), int(min(ys))
                        x_max, y_max = int(max(xs)), int(max(ys))
                        w, h = x_max - x_min, y_max - y_min

                        # Pad box slightly
                        padding = 10
                        x_min = max(0, x_min - padding)
                        y_min = max(0, y_min - padding)
                        w += 2 * padding
                        h += 2 * padding

                        # Map back to work image coords
                        box_work = inverse_transform_box((x_min, y_min, w, h), M_rot, work_shape)
                        # Then map to original image coords if we had a pre-rotation
                        if pre_M is not None:
                            original_box = inverse_transform_box(box_work, pre_M, original_shape)
                        else:
                            original_box = box_work

                        all_detections.append(original_box)
                        break  # proceed to next cluster/angle

    if debug:
        print(f"\nTotal detections before deduplication: {len(all_detections)}")

    # Deduplicate
    if len(all_detections) > 1:
        all_detections = remove_overlapping_boxes(all_detections, iou_threshold=0.3)

    return all_detections


def remove_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """Remove highly overlapping boxes, keeping the larger one."""
    if len(boxes) <= 1:
        return boxes
    
    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter_area
        
        return inter_area / union if union > 0 else 0
    
    keep = []
    used = set()
    
    # Sort by area (largest first)
    sorted_boxes = sorted(enumerate(boxes), key=lambda x: x[1][2] * x[1][3], reverse=True)
    
    for idx, box in sorted_boxes:
        if idx in used:
            continue
        keep.append(box)
        # Mark overlapping boxes as used
        for idx2, box2 in sorted_boxes:
            if idx2 != idx and compute_iou(box, box2) > iou_threshold:
                used.add(idx2)
    
    return keep
