"""
Detection Overlays Module
Handles all detection algorithms and overlay drawing for silhouettes, bright spots, dark spots, and moving objects
"""
import cv2
import numpy as np
import random
from utils import hsv2bgr

# Detection names
hand_names = ["Gesture", "Wave", "Grip", "Touch", "Pulse"]
bright_names = ["Flash", "Glow", "Spark", "Beam", "Blaze", "Pixel", "Node", "Point", "Dot", "Ray"]
moving_names = ["Shift", "Drift", "Pulse", "Flow", "Surge"]
silhouette_names = ["Ghost", "Shadow", "Echo", "Outline", "Trace"]

class DetectionConfig:
    def __init__(self):
        self.top_bright_spots = 10  # default to 10
        self.top_dark_spots = 10
        self.top_moving_spots = 5
        self.min_silhouette_area = 500
        self.min_dark_area = 5
        self.min_bright_area = 5
        self.min_moving_area = 5

def detect_silhouettes(frame, bg_subtractor, silhouette_threshold, config, hue_shift):
    """Detect and draw silhouettes using background subtraction"""
    fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    _, fg_mask = cv2.threshold(fg_mask, silhouette_threshold, 255, cv2.THRESH_BINARY)
    contours_silhouette, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    overlay = np.zeros_like(frame, dtype=np.uint8)

    for idx, contour in enumerate(contours_silhouette):
        area = cv2.contourArea(contour)
        if area > config.min_silhouette_area:
            color = hsv2bgr((hue_shift + idx * 10) % 180)
            cv2.drawContours(overlay, [contour], -1, color, 1)
            label = random.choice(silhouette_names)
            percent = random.randint(0, 100)
            obj_id = random.randint(1000, 9999)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(overlay, f"Silhouette: {label} [{obj_id}] {{{percent}%}}",
                       (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)

            # 1/20 chance to invert this silhouette area
            should_invert = random.randint(1, 20) == 1
            detections.append({
                'type': 'silhouette',
                'bbox': (x, y, w, h),
                'should_invert': should_invert
            })

    return overlay, detections

def detect_bright_spots(frame, config, hue_shift):
    """Detect and draw bright spots"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_brightness = []
    for cnt in contours:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        brightness = cv2.mean(gray, mask=mask)[0]
        contour_brightness.append((brightness, cnt))

    contour_brightness.sort(reverse=True, key=lambda x: x[0])
    top_bright_contours = [cnt for _, cnt in contour_brightness[:config.top_bright_spots]]

    overlay = np.zeros_like(frame, dtype=np.uint8)
    detections = []
    bright_spot_centers = []

    for idx, cnt in enumerate(top_bright_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        box_w, box_h = max(w // 2, config.min_bright_area), max(h // 2, config.min_bright_area)
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2
        color = (255, 255, 255)  # WHITE for bright spot boxes
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        label = random.choice(bright_names)
        percent = random.randint(0, 100)
        obj_id = random.randint(1000, 9999)
        text = f"Bright: {label} [{obj_id}] {{{percent}%}}"
        cv2.putText(overlay, text, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)
        bright_spot_centers.append((cx, cy))

        # 1/20 chance to invert this box
        should_invert = random.randint(1, 20) == 1
        detections.append({
            'type': 'bright',
            'bbox': (x1, y1, x2-x1, y2-y1),
            'center': (cx, cy),
            'should_invert': should_invert
        })

    return overlay, detections, bright_spot_centers

def detect_dark_spots(frame, config, hue_shift):
    """Detect and draw dark spots"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh_dark = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours_dark, _ = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_darkness = []
    for cnt in contours_dark:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        darkness = cv2.mean(gray, mask=mask)[0]
        contour_darkness.append((darkness, cnt))

    contour_darkness.sort(key=lambda x: x[0])
    top_dark_contours = [cnt for _, cnt in contour_darkness[:config.top_dark_spots]]

    overlay = np.zeros_like(frame, dtype=np.uint8)
    detections = []

    for idx, cnt in enumerate(top_dark_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        box_w, box_h = max(w // 2, config.min_dark_area), max(h // 2, config.min_dark_area)
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2
        color = hsv2bgr((hue_shift + idx * 10 + 90) % 180)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        label = "Void"
        percent = random.randint(0, 100)
        obj_id = random.randint(1000, 9999)
        text = f"Dark: {label} [{obj_id}] {{{percent}%}}"
        cv2.putText(overlay, text, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)

        # 1/20 chance to invert this box
        should_invert = random.randint(1, 20) == 1
        detections.append({
            'type': 'dark',
            'bbox': (x1, y1, x2-x1, y2-y1),
            'center': (cx, cy),
            'should_invert': should_invert
        })

    return overlay, detections

def detect_moving_spots(frame, prev_gray, config, hue_shift):
    """Detect and draw moving spots"""
    if prev_gray is None:
        return np.zeros_like(frame, dtype=np.uint8), []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    flat_diff = diff.flatten()

    overlay = np.zeros_like(frame, dtype=np.uint8)
    detections = []

    if len(flat_diff) > config.top_moving_spots:
        top_moving_idx = np.argpartition(flat_diff, -config.top_moving_spots)[-config.top_moving_spots:]
        coords_move = [np.unravel_index(idx, diff.shape) for idx in top_moving_idx]

        for idx, (y, x) in enumerate(coords_move):
            cx, cy = x, y
            box_w, box_h = config.min_moving_area, config.min_moving_area
            x1, y1 = cx - box_w // 2, cy - box_h // 2
            x2, y2 = cx + box_w // 2, cy + box_h // 2
            color = hsv2bgr((hue_shift + idx * 10 + 120) % 180)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            label = random.choice(moving_names)
            percent = random.randint(0, 100)
            obj_id = random.randint(1000, 9999)
            text = f"Move: {label} [{obj_id}] {{{percent}%}}"
            cv2.putText(overlay, text, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)

            # 1/20 chance to invert this box
            should_invert = random.randint(1, 20) == 1
            detections.append({
                'type': 'moving',
                'bbox': (x1, y1, x2-x1, y2-y1),
                'center': (cx, cy),
                'should_invert': should_invert
            })

    return overlay, detections

def draw_connecting_lines(overlay, bright_spot_centers, all_centers):
    """Draw lines from bright spots to nearest boxes"""
    if len(all_centers) > 2 and len(bright_spot_centers) > 0:
        for b_c in bright_spot_centers:
            # Calculate distances to all other boxes
            distances = []
            for center in all_centers:
                if center != b_c:  # Don't include the bright spot itself
                    dx = center[0] - b_c[0]
                    dy = center[1] - b_c[1]
                    distance = (dx * dx + dy * dy) ** 0.5
                    distances.append((distance, center))

            # Sort by distance and get the 2 nearest
            distances.sort(key=lambda x: x[0])
            nearest_boxes = distances[:min(2, len(distances))]

            # Draw lines to the 2 nearest boxes
            for distance, target_center in nearest_boxes:
                color = (255, 255, 255)  # WHITE for lines
                cv2.line(overlay, b_c, target_center, color, 1, lineType=cv2.LINE_AA)

            # 1/5 chance to draw a third line to a random box
            if random.randint(1, 5) == 1 and len(distances) > 2:
                used_centers = {center for _, center in nearest_boxes}
                remaining_centers = [center for _, center in distances if center not in used_centers]

                if remaining_centers:
                    random_target_center = random.choice(remaining_centers)
                    color = (255, 255, 255)  # WHITE for lines
                    cv2.line(overlay, b_c, random_target_center, color, 1, lineType=cv2.LINE_AA)

def apply_inversion_effects(frame, detections, scale_x, scale_y):
    """Apply inversion effects to detected areas"""
    for detection in detections:
        if detection.get('should_invert', False):
            if detection['type'] == 'silhouette':
                x, y, w, h = detection['bbox']
            else:
                x, y, w, h = detection['bbox']

            # Scale coordinates to full resolution
            x_full = int(x * scale_x)
            y_full = int(y * scale_y)
            w_full = int(w * scale_x)
            h_full = int(h * scale_y)

            # Ensure coordinates are within frame bounds
            x_full = max(0, min(x_full, frame.shape[1] - 1))
            y_full = max(0, min(y_full, frame.shape[0] - 1))
            w_full = min(w_full, frame.shape[1] - x_full)
            h_full = min(h_full, frame.shape[0] - y_full)

            # Invert the region
            if w_full > 0 and h_full > 0:
                frame[y_full:y_full+h_full, x_full:x_full+w_full] = cv2.bitwise_not(
                    frame[y_full:y_full+h_full, x_full:x_full+w_full])

    return frame
