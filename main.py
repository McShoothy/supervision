import cv2
import numpy as np
import random
import time
import json
import os
import multiprocessing
import platform

# Control state file for communication between windows
STATE_FILE = "/tmp/vj_state.json" if platform.system() != "Windows" else "vj_state.json"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
prev_gray = None

hand_names = ["Gesture", "Wave", "Grip", "Touch", "Pulse"]
bright_names = ["Flash", "Glow", "Spark", "Beam", "Blaze", "Pixel", "Node", "Point", "Dot", "Ray"]
moving_names = ["Shift", "Drift", "Pulse", "Flow", "Surge"]

# Reduce history for faster background subtraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
prev_silhouette_area = 0

frame_count = 0

# --- Dark mode colors ---
DARK_BG = (18, 18, 18)
DARK_TEXT = (200, 200, 200)
DARK_ACCENT = (80, 80, 255)

# Track inverted boxes: list of dicts with keys: coords, until, typ
inverted_boxes = []

class DetectionConfig:
    def __init__(self):
        self.top_bright_spots = 10  # default to 10
        self.top_dark_spots = 10
        self.top_moving_spots = 5
        self.min_silhouette_area = 500
        self.min_dark_area = 5
        self.min_bright_area = 5
        self.min_moving_area = 5

config = DetectionConfig()

class Keybinds:
    def __init__(self):
        self.quit = 27  # ESC
        self.toggle_bw = ord('b')
        self.toggle_fullscreen = ord('f')
        self.toggle_presentation = ord('p')
        self.increase_bright = 83  # Right arrow key
        self.decrease_bright = 81  # Left arrow key


keybinds = Keybinds()

def hsv2bgr(h, s=255, v=255):
    """Convert hue (0-179) to BGR color tuple."""
    color = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0,0])

hue_shift = 0  # For color cycling

# Global variables for fullscreen state
is_fullscreen = False
fullscreen_monitor = 0  # Default to primary monitor
cv2_window_name = "VJ Video Stream"

def load_state():
    """Load control state from file"""
    default_state = {
        'invert': False,
        'monochrome': False,
        'mono_hue': 90,
        'show_boxes': True,
        'presentation_mode': False,
        'fullscreen': False,
        'fullscreen_monitor': 0,
        'show_silhouettes': True,
        'show_bright': True,
        'show_dark': True,
        'show_moving': True,
        'max_bright': 10,      # default to 10
        'max_dark': 10,
        'max_moving': 5,
        'start_video': False,
        'camera_brightness': 50,
        'camera_contrast': 50,
        'camera_hue': 50,
        'filter_cvr': False,
        'filter_static': False,
        'filter_grain': False,
        'filter_vlines': False,
        'shift_x': 0,
        'shift_y': 0,
        'static_intensity': 50,
        'grain_intensity': 50,
        'vlines_intensity': 50,
        'running': True,
        'silhouette_threshold': 50,  # add default
    }
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                return {**default_state, **state}
    except:
        pass
    return default_state

def save_detection_counts(counts):
    """Save detection counts to state file for control window"""
    try:
        state = load_state()
        state['detection_counts'] = counts
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except:
        pass

def get_monitor_info():
    """Get monitor information cross-platform"""
    if platform.system() == "Windows":
        try:
            # Try to get Windows monitor info using ctypes
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            monitors = []

            def enum_display_monitors_proc(hdc, lprcClip, lpfnEnum, dwData):
                monitors.append((
                    lprcClip.contents.left,
                    lprcClip.contents.top,
                    lprcClip.contents.right - lprcClip.contents.left,
                    lprcClip.contents.bottom - lprcClip.contents.top
                ))
                return True

            # Define the callback function type
            MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool,
                                               wintypes.HDC,
                                               ctypes.POINTER(wintypes.RECT),
                                               wintypes.LPARAM)

            # Get all monitors
            try:
                user32.EnumDisplayMonitors(None, None, MonitorEnumProc(enum_display_monitors_proc), 0)
            except:
                # Fallback: get primary monitor size
                width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
                height = user32.GetSystemMetrics(1)  # SM_CYSCREEN
                monitors = [(0, 0, width, height)]

            if not monitors:
                monitors = [(0, 0, 1920, 1080)]  # Ultimate fallback

            return monitors

        except Exception as e:
            print(f"Windows monitor detection failed: {e}")
            return [(0, 0, 1920, 1080)]  # Fallback
    else:
        # Linux/Unix - use xrandr
        try:
            import subprocess
            result = subprocess.run(['xrandr', '--current'], capture_output=True, text=True)
            if result.returncode != 0:
                return [(0, 0, 1920, 1080)]  # Fallback

            monitors = []
            lines = result.stdout.split('\n')

            for line in lines:
                if ' connected' in line and '+' in line:
                    # Parse line like: "HDMI-1 connected 1920x1080+1920+0"
                    parts = line.split()
                    for part in parts:
                        if '+' in part and 'x' in part:
                            # Extract geometry: widthxheight+x_offset+y_offset
                            try:
                                geom = part.split('+')
                                width_height = geom[0].split('x')
                                width = int(width_height[0])
                                height = int(width_height[1])
                                x_offset = int(geom[1]) if len(geom) > 1 else 0
                                y_offset = int(geom[2]) if len(geom) > 2 else 0
                                monitors.append((x_offset, y_offset, width, height))
                                break
                            except (ValueError, IndexError):
                                continue

            if not monitors:
                monitors = [(0, 0, 1920, 1080)]  # Fallback

            # Sort by x_offset to ensure primary monitor is first
            monitors.sort(key=lambda m: m[0])
            return monitors
        except:
            return [(0, 0, 1920, 1080)]  # Fallback

def create_fullscreen_window(monitor_index=0):
    """Create a true fullscreen OpenCV window on specified monitor"""
    global is_fullscreen

    monitors = get_monitor_info()

    # Ensure monitor_index is valid
    if monitor_index >= len(monitors):
        monitor_index = 0

    x_offset, y_offset, width, height = monitors[monitor_index]

    # Create or recreate window
    try:
        cv2.destroyWindow(cv2_window_name)
    except:
        pass

    cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)

    # Position window on the correct monitor BEFORE setting fullscreen
    cv2.moveWindow(cv2_window_name, x_offset + 100, y_offset + 100)
    cv2.resizeWindow(cv2_window_name, width, height)

    # Small delay to ensure window is positioned
    import time
    time.sleep(0.1)

    # Set to fullscreen
    cv2.setWindowProperty(cv2_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    is_fullscreen = True
    return width, height

def create_windowed_window():
    """Create a normal windowed OpenCV window"""
    global is_fullscreen

    # Safely destroy existing window
    try:
        cv2.destroyWindow(cv2_window_name)
    except:
        pass

    cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cv2_window_name, 1280, 720)

    # Position on primary monitor
    cv2.moveWindow(cv2_window_name, 100, 100)

    is_fullscreen = False

def safe_window_transition(target_fullscreen, target_monitor):
    """Safely transition between window modes without breaking"""
    global is_fullscreen

    if target_fullscreen == is_fullscreen:
        return  # No change needed

    try:
        if target_fullscreen:
            create_fullscreen_window(target_monitor)
        else:
            create_windowed_window()
    except Exception as e:
        print(f"Window transition failed: {e}")
        # Emergency fallback - create basic windowed mode
        try:
            cv2.destroyWindow(cv2_window_name)
        except:
            pass
        cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cv2_window_name, 1280, 720)
        is_fullscreen = False

def apply_camera_adjustments(frame, brightness, contrast, hue_shift):
    """Apply brightness, contrast, and hue adjustments to the camera frame"""
    # Convert brightness and contrast from 0-100 range to usable values
    # Brightness: 0-100 -> -50 to +50
    brightness_value = int((brightness - 50) * 1.0)

    # Contrast: 0-100 -> 0.5 to 2.0
    contrast_value = (contrast / 50.0)

    # Apply brightness and contrast
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast_value, beta=brightness_value)

    # Apply hue shift if not at neutral (50)
    if hue_shift != 50:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        # Hue shift: 0-100 -> -90 to +90 degrees
        hue_offset = int((hue_shift - 50) * 1.8)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + hue_offset) % 180
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return adjusted

def apply_crt_filter(frame):
    # Simulate CRT/VHS effect: add scanlines and slight color shift
    out = frame.copy()
    for y in range(0, out.shape[0], 2):
        out[y:y+1, :, :] = (out[y:y+1, :, :] * 0.7).astype(np.uint8)
    # Slight color shift
    out[..., 1] = np.clip(out[..., 1] * 0.95, 0, 255).astype(np.uint8)
    return out

def apply_static_filter(frame):
    # Add random static noise
    noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
    out = cv2.add(frame, noise)
    return out

def apply_grain_filter(frame):
    # Film grain effect removed
    return frame

def apply_shift_filter(frame, shift_x, shift_y):
    # Roll the image horizontally and vertically
    return np.roll(np.roll(frame, shift_y, axis=0), shift_x, axis=1)

# Add these globals to accumulate roll offsets
roll_offset_x = 0
roll_offset_y = 0

def update_video_texture():
    global prev_gray, frame_count
    ret, frame = cap.read()
    if not ret:
        cv2.destroyAllWindows()
        return

    # Load current state from control window
    vj_state = load_state()

    # Check if control window requested exit
    if not vj_state.get('running', True):
        cv2.destroyAllWindows()
        return

    # Handle fullscreen toggle
    current_fullscreen_state = vj_state.get('fullscreen', False)
    current_monitor = vj_state.get('fullscreen_monitor', 0)

    if current_fullscreen_state != is_fullscreen:
        cv2.destroyWindow(cv2_window_name)
        if current_fullscreen_state:
            create_fullscreen_window(current_monitor)
        else:
            create_windowed_window()

    now = time.time()
    global hue_shift
    hue_shift = (hue_shift + 2) % 180

    # Use loaded state for controls
    invert = vj_state['invert']
    mono = vj_state['monochrome']
    mono_hue = vj_state['mono_hue']
    show_boxes = vj_state['show_boxes']
    presentation_mode = vj_state['presentation_mode']

    # Individual box controls
    show_silhouettes = vj_state.get('show_silhouettes', True)
    show_bright = vj_state.get('show_bright', True)
    show_dark = vj_state.get('show_dark', True)
    show_moving = vj_state.get('show_moving', True)
    silhouette_threshold = vj_state.get('silhouette_threshold', 50)

    # Max box counts from state
    config.top_bright_spots = vj_state.get('max_bright', config.top_bright_spots)
    config.top_dark_spots = vj_state.get('max_dark', config.top_dark_spots)
    config.top_moving_spots = vj_state.get('max_moving', config.top_moving_spots)

    # Apply monochrome if enabled
    if mono:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = mono_hue
        hsv[..., 1] = 255
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Apply invert if enabled
    if invert:
        frame = cv2.bitwise_not(frame)

    full_res_frame = cv2.resize(frame, (1280, 720))

    # --- Dark mode: darken the video feed ---
    dark_overlay = np.full_like(full_res_frame, DARK_BG, dtype=np.uint8)
    alpha = 0.25 if not presentation_mode else 0.5
    frame_dark = cv2.addWeighted(full_res_frame, 1 - alpha, dark_overlay, alpha, 0)

    # Downscale for detection
    proc_frame = cv2.resize(full_res_frame, (640, 360))
    gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)

    # Only update detection and overlays every 3 frames for performance
    if frame_count % 2 == 0:
        overlay_small = np.zeros_like(proc_frame, dtype=np.uint8)
        box_centers = []
        detection_counts = {'humans': 0, 'bright': 0, 'dark': 0, 'move': 0, 'faces': 0}

        if show_boxes:
            # Initialize bright_spot_centers to avoid reference errors
            bright_spot_centers = []

            # Silhouette detection
            if show_silhouettes:
                fg_mask = bg_subtractor.apply(proc_frame)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                # --- use silhouette_threshold from state ---
                _, fg_mask = cv2.threshold(fg_mask, silhouette_threshold, 255, cv2.THRESH_BINARY)
                contours_silhouette, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                silhouette_names = ["Ghost", "Shadow", "Echo", "Outline", "Trace"]
                for idx, contour in enumerate(contours_silhouette):
                    area = cv2.contourArea(contour)
                    if area > config.min_silhouette_area:
                        color = hsv2bgr((hue_shift + idx * 10) % 180)
                        cv2.drawContours(overlay_small, [contour], -1, color, 1)  # 1px thick
                        label = random.choice(silhouette_names)
                        percent = random.randint(0, 100)
                        obj_id = random.randint(1000, 9999)
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.putText(overlay_small, f"Silhouette: {label} [{obj_id}] {{{percent}%}}", (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)

                        # 1/20 chance to invert this silhouette area
                        if random.randint(1, 20) == 1:
                            # Scale coordinates to full resolution
                            scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
                            scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
                            x_full = int(x * scale_x)
                            y_full = int(y * scale_y)
                            w_full = int(w * scale_x)
                            h_full = int(h * scale_y)

                            # Ensure coordinates are within frame bounds
                            x_full = max(0, min(x_full, full_res_frame.shape[1] - w_full))
                            y_full = max(0, min(y_full, full_res_frame.shape[0] - h_full))
                            w_full = min(w_full, full_res_frame.shape[1] - x_full)
                            h_full = min(h_full, full_res_frame.shape[0] - y_full)

                            # Invert the region inside the silhouette bounding box
                            if w_full > 0 and h_full > 0:
                                frame_dark[y_full:y_full+h_full, x_full:x_full+w_full] = cv2.bitwise_not(frame_dark[y_full:y_full+h_full, x_full:x_full+w_full])

            # Bright spots (configurable amount)
            if show_bright:
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
                detection_counts['bright'] = len(top_bright_contours)

                bright_spot_centers = []
                for idx, cnt in enumerate(top_bright_contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w // 2, y + h // 2
                    box_w, box_h = max(w // 2, config.min_bright_area), max(h // 2, config.min_bright_area)
                    x1, y1 = cx - box_w // 2, cy - box_h // 2
                    x2, y2 = cx + box_w // 2, cy + box_h // 2
                    color = (255, 255, 255)  # WHITE for bright spot boxes
                    cv2.rectangle(overlay_small, (x1, y1), (x2, y2), color, 1)  # 1px thick
                    label = random.choice(bright_names)
                    percent = random.randint(0, 100)
                    obj_id = random.randint(1000, 9999)
                    text = f"Bright: {label} [{obj_id}] {{{percent}%}}"
                    cv2.putText(overlay_small, text, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)
                    box_centers.append(((cx, cy), (x1, y1, x2, y2), 'bright'))
                    bright_spot_centers.append((cx, cy))

                    # 1/20 chance to invert this box
                    if random.randint(1, 20) == 1:
                        # Scale coordinates to full resolution
                        scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
                        scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
                        x1_full = int(x1 * scale_x)
                        y1_full = int(y1 * scale_y)
                        x2_full = int(x2 * scale_x)
                        y2_full = int(y2 * scale_y)

                        # Ensure coordinates are within frame bounds
                        x1_full = max(0, min(x1_full, full_res_frame.shape[1] - 1))
                        y1_full = max(0, min(y1_full, full_res_frame.shape[0] - 1))
                        x2_full = max(0, min(x2_full, full_res_frame.shape[1] - 1))
                        y2_full = max(0, min(y2_full, full_res_frame.shape[0] - 1))

                        # Invert the region inside the box on the final frame
                        if x2_full > x1_full and y2_full > y1_full:
                            frame_dark[y1_full:y2_full, x1_full:x2_full] = cv2.bitwise_not(frame_dark[y1_full:y2_full, x1_full:x2_full])

            else:
                detection_counts['bright'] = 0

            # Darkest spots (configurable amount)
            if show_dark:
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
                detection_counts['dark'] = len(top_dark_contours)

                for idx, cnt in enumerate(top_dark_contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w // 2, y + h // 2
                    box_w, box_h = max(w // 2, config.min_dark_area), max(h // 2, config.min_dark_area)
                    x1, y1 = cx - box_w // 2, cy - box_h // 2
                    x2, y2 = cx + box_w // 2, cy + box_h // 2
                    color = hsv2bgr((hue_shift + idx * 10 + 90) % 180)
                    cv2.rectangle(overlay_small, (x1, y1), (x2, y2), color, 1)  # 1px thick
                    label = "Void"
                    percent = random.randint(0, 100)
                    obj_id = random.randint(1000, 9999)
                    text = f"Dark: {label} [{obj_id}] {{{percent}%}}"
                    cv2.putText(overlay_small, text, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)
                    box_centers.append(((cx, cy), (x1, y1, x2, y2), 'dark'))

                    # 1/20 chance to invert this box
                    if random.randint(1, 20) == 1:
                        # Scale coordinates to full resolution
                        scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
                        scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
                        x1_full = int(x1 * scale_x)
                        y1_full = int(y1 * scale_y)
                        x2_full = int(x2 * scale_x)
                        y2_full = int(y2 * scale_y)

                        # Ensure coordinates are within frame bounds
                        x1_full = max(0, min(x1_full, full_res_frame.shape[1] - 1))
                        y1_full = max(0, min(y1_full, full_res_frame.shape[0] - 1))
                        x2_full = max(0, min(x2_full, full_res_frame.shape[1] - 1))
                        y2_full = max(0, min(y2_full, full_res_frame.shape[0] - 1))

                        # Invert the region inside the box on the final frame
                        if x2_full > x1_full and y2_full > y1_full:
                            frame_dark[y1_full:y2_full, x1_full:x2_full] = cv2.bitwise_not(frame_dark[y1_full:y2_full, x1_full:x2_full])

            else:
                detection_counts['dark'] = 0

            # Moving spots (configurable amount)
            if show_moving and prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                flat_diff = diff.flatten()
                if len(flat_diff) > config.top_moving_spots:
                    top_moving_idx = np.argpartition(flat_diff, -config.top_moving_spots)[-config.top_moving_spots:]
                    coords_move = [np.unravel_index(idx, diff.shape) for idx in top_moving_idx]
                    detection_counts['move'] = len(coords_move)
                    for idx, (y, x) in enumerate(coords_move):
                        cx, cy = x, y
                        box_w, box_h = config.min_moving_area, config.min_moving_area
                        x1, y1 = cx - box_w // 2, cy - box_h // 2
                        x2, y2 = cx + box_w // 2, cy + box_h // 2
                        color = hsv2bgr((hue_shift + idx * 10 + 120) % 180)
                        cv2.rectangle(overlay_small, (x1, y1), (x2, y2), color, 1)  # 1px thick
                        label = random.choice(moving_names)
                        percent = random.randint(0, 100)
                        obj_id = random.randint(1000, 9999)
                        text = f"Move: {label} [{obj_id}] {{{percent}%}}"
                        cv2.putText(overlay_small, text, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)
                        box_centers.append(((cx, cy), (x1, y1, x2, y2), 'move'))

                        # 1/20 chance to invert this box
                        if random.randint(1, 20) == 1:
                            # Scale coordinates to full resolution
                            scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
                            scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
                            x1_full = int(x1 * scale_x)
                            y1_full = int(y1 * scale_y)
                            x2_full = int(x2 * scale_x)
                            y2_full = int(y2 * scale_y)

                            # Ensure coordinates are within frame bounds
                            x1_full = max(0, min(x1_full, full_res_frame.shape[1] - 1))
                            y1_full = max(0, min(y1_full, full_res_frame.shape[0] - 1))
                            x2_full = max(0, min(x2_full, full_res_frame.shape[1] - 1))
                            y2_full = max(0, min(y2_full, full_res_frame.shape[0] - 1))

                            # Invert the region inside the box on the final frame
                            if x2_full > x1_full and y2_full > y1_full:
                                frame_dark[y1_full:y2_full, x1_full:x2_full] = cv2.bitwise_not(frame_dark[y1_full:y2_full, x1_full:x2_full])

            else:
                detection_counts['move'] = 0

            # Draw lines from each bright spot to the 2 nearest boxes
            if len(box_centers) > 2 and 'bright_spot_centers' in locals():
                for idx, b_c in enumerate(bright_spot_centers):
                    # Calculate distances to all other boxes
                    distances = []
                    for i, (center, _, _) in enumerate(box_centers):
                        if center != b_c:  # Don't include the bright spot itself
                            dx = center[0] - b_c[0]
                            dy = center[1] - b_c[1]
                            distance = (dx * dx + dy * dy) ** 0.5  # Euclidean distance
                            distances.append((distance, i, center))

                    # Sort by distance and get the 2 nearest
                    distances.sort(key=lambda x: x[0])
                    nearest_boxes = distances[:min(2, len(distances))]

                    # Draw lines to the 2 nearest boxes
                    for distance, box_idx, target_center in nearest_boxes:
                        color = (255, 255, 255)  # WHITE for lines
                        # Draw as thin as possible (1px), anti-aliased line
                        cv2.line(overlay_small, b_c, target_center, color, 1, lineType=cv2.LINE_AA)

                    # 1/5 chance to draw a third line to a random box (not considering distance)
                    if random.randint(1, 5) == 1 and len(distances) > 2:
                        # Get boxes that weren't already connected (exclude the 2 nearest)
                        used_indices = {box_idx for _, box_idx, _ in nearest_boxes}
                        remaining_boxes = [item for item in distances if item[1] not in used_indices]

                        if remaining_boxes:
                            # Pick a random box from the remaining ones
                            random_box = random.choice(remaining_boxes)
                            _, _, random_target_center = random_box
                            color = (255, 255, 255)  # WHITE for lines
                            cv2.line(overlay_small, b_c, random_target_center, color, 1, lineType=cv2.LINE_AA)

        # Apply random box inversions (1/10 chance per box)
        if box_centers and random.randint(1, 10) == 1:
            # Select a random box to invert
            selected_box = random.choice(box_centers)
            _, (x1, y1, x2, y2), _ = selected_box

            # Scale coordinates to full resolution
            scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
            scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
            x1_full = int(x1 * scale_x)
            y1_full = int(y1 * scale_y)
            x2_full = int(x2 * scale_x)
            y2_full = int(y2 * scale_y)

            # Ensure coordinates are within frame bounds
            x1_full = max(0, min(x1_full, full_res_frame.shape[1] - 1))
            y1_full = max(0, min(y1_full, full_res_frame.shape[0] - 1))
            x2_full = max(0, min(x2_full, full_res_frame.shape[1] - 1))
            y2_full = max(0, min(y2_full, full_res_frame.shape[0] - 1))

            # Invert the region inside the box on the final frame
            if x2_full > x1_full and y2_full > y1_full:
                frame_dark[y1_full:y2_full, x1_full:x2_full] = cv2.bitwise_not(frame_dark[y1_full:y2_full, x1_full:x2_full])

        # Save detection counts for control window
        save_detection_counts(detection_counts)

        # Store for reuse in skipped frames
        main_video_loop.last_overlay = overlay_small
    else:
        # Reuse previous overlay
        overlay_small = getattr(main_video_loop, 'last_overlay', np.zeros_like(proc_frame, dtype=np.uint8))

    # Scale overlay to full resolution and apply to frame
    overlay_full = cv2.resize(overlay_small, (1280, 720), interpolation=cv2.INTER_LINEAR)
    final_frame = cv2.addWeighted(frame_dark, 1.0, overlay_full, 1.0, 0)

    # Apply filters based on vj_state
    if vj_state.get('filter_cvr', False):
        final_frame = apply_crt_filter(final_frame)
    if vj_state.get('filter_static', False):
        final_frame = apply_static_filter(final_frame)
    if vj_state.get('filter_grain', False):
        final_frame = apply_grain_filter(final_frame)

    # --- Rolling logic ---
    global roll_offset_x, roll_offset_y
    speed_x = vj_state.get('shift_x', 0)
    speed_y = vj_state.get('shift_y', 0)
    roll_offset_x = (roll_offset_x + speed_x) % final_frame.shape[1]
    roll_offset_y = (roll_offset_y + speed_y) % final_frame.shape[0]
    if speed_x != 0 or speed_y != 0:
        final_frame = apply_shift_filter(final_frame, roll_offset_x, roll_offset_y)

    # --- Tiling logic ---
    allowed_tiles = [1, 4, 9, 16, 25, 36]
    tile_count = vj_state.get('tile_count', 1)
    # Snap to nearest allowed tile count
    tile_count = min(allowed_tiles, key=lambda x: abs(x - tile_count))
    grid_size = int(np.sqrt(tile_count))
    h, w = final_frame.shape[:2]
    tile_h, tile_w = h // grid_size, w // grid_size
    tile_img = cv2.resize(final_frame, (tile_w, tile_h))
    tiled_frame = np.zeros_like(final_frame)
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx >= tile_count:
                break
            y1, y2 = i * tile_h, (i + 1) * tile_h
            x1, x2 = j * tile_w, (j + 1) * tile_w
            tiled_frame[y1:y2, x1:x2] = tile_img
    final_frame = tiled_frame

    # --- FPS calculation ---
    if not hasattr(main_video_loop, 'last_fps_time'):
        main_video_loop.last_fps_time = time.time()
        main_video_loop.frame_counter = 0
    main_video_loop.frame_counter += 1
    now_fps = time.time()
    if now_fps - main_video_loop.last_fps_time >= 1.0:
        fps = main_video_loop.frame_counter / (now_fps - main_video_loop.last_fps_time)
        # Save FPS to state file for control window
        vj_state['video_fps'] = round(fps, 1)
        with open(STATE_FILE, 'w') as f:
            json.dump(vj_state, f)
        main_video_loop.last_fps_time = now_fps
        main_video_loop.frame_counter = 0

    # --- Letterbox for fullscreen: fill bars with black ---
    if is_fullscreen:
        # Get current monitor size
        monitors = get_monitor_info()
        mon_w, mon_h = 1920, 1080
        if vj_state.get('fullscreen_monitor', 0) < len(monitors):
            _, _, mon_w, mon_h = monitors[vj_state.get('fullscreen_monitor', 0)]
        # Letterbox the frame to fit monitor, black bars
        final_frame = letterbox_image(final_frame, (mon_h, mon_w), fill_color=(0, 0, 0))

    # Display the frame - with error handling
    try:
        cv2.imshow(cv2_window_name, final_frame)
    except cv2.error:
        create_windowed_window()
        cv2.imshow(cv2_window_name, final_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        cv2.destroyAllWindows()
        return
    elif key == ord('f'):  # F key to toggle fullscreen
        current_state = load_state()
        current_state['fullscreen'] = not current_state['fullscreen']
        with open(STATE_FILE, 'w') as f:
            json.dump(current_state, f)

    prev_gray = gray.copy()
    frame_count += 1

def get_overlay_video_from_state(vj_state):
    """Return overlay video path if enabled, else None."""
    if vj_state.get('overlay_enabled', False):
        overlay_file = vj_state.get('overlay_video', None)
        if overlay_file:
            overlay_path = os.path.join("./videos", overlay_file)
            if os.path.exists(overlay_path):
                return overlay_path
    return None

def get_video_source_from_state(vj_state):
    """Return video source (Camera or Video File) from state."""
    return vj_state.get('video_source', 'Camera')

def letterbox_image(image, target_size, fill_color=(0, 0, 0)):
    """Resize image to fit target_size with aspect ratio, pad with fill_color."""
    ih, iw = image.shape[:2]
    th, tw = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((th, tw, 3), fill_color, dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    new_image[top:top+nh, left:left+nw] = image_resized
    return new_image

def init_camera():
    """Initialize camera with cross-platform compatibility"""
    global cap
    try:
        # Try DirectShow backend on Windows for better compatibility
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Test if camera is working
        ret, test_frame = cap.read()
        if not ret:
            print("Warning: Camera initialization failed, trying fallback...")
            cap.release()
            cap = cv2.VideoCapture(0)  # Fallback without specific backend
            ret, test_frame = cap.read()
            if not ret:
                print("Error: No camera found or camera access denied")
                return False

        print(f"Camera initialized successfully on {platform.system()}")
        return True
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return False

# Initialize camera at startup
init_camera()

def main_video_loop():
    global prev_gray, frame_count, is_fullscreen

    print("Waiting for video stream to start...")
    print("Check 'Start Video Stream' in the control panel to begin.")

    window_created = False
    overlay_cap = None
    overlay_active = False
    overlay_path_last = None  # Track last overlay path

    video_file_cap = None
    video_file_path_last = None

    while True:
        vj_state = load_state()

        if not vj_state.get('running', True):
            break

        if not vj_state.get('start_video', False):
            time.sleep(0.1)
            continue

        # --- Video File Progress Tracking ---
        video_source = vj_state.get('video_source', 'Camera')
        video_file = vj_state.get('video_file', None)
        video_progress = 0.0
        video_time_str = "0:00 / 0:00"

        if video_source == "Video File" and video_file:
            video_path = os.path.join("./videos", video_file)
            if os.path.exists(video_path):
                if video_file_cap is None or video_file_path_last != video_path:
                    if video_file_cap is not None:
                        video_file_cap.release()
                    video_file_cap = cv2.VideoCapture(video_path)
                    video_file_path_last = video_path
                cap_to_use = video_file_cap
            else:
                cap_to_use = cap
        else:
            cap_to_use = cap

        # Overlay video logic
        overlay_path = get_overlay_video_from_state(vj_state)
        if overlay_path:
            if overlay_cap is None or not overlay_active or overlay_path != overlay_path_last:
                if overlay_cap is not None:
                    overlay_cap.release()
                overlay_cap = cv2.VideoCapture(overlay_path)
                overlay_active = True
                overlay_path_last = overlay_path
            ret, frame = overlay_cap.read()
            if not ret:
                overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = overlay_cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
        else:
            overlay_active = False
            overlay_path_last = None
            if overlay_cap is not None:
                overlay_cap.release()
                overlay_cap = None
            ret, frame = cap_to_use.read()
            # --- Loop video file if at end ---
            if not ret and cap_to_use is video_file_cap:
                video_file_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = video_file_cap.read()
            if not ret:
                print("Failed to read from camera or video file")
                time.sleep(0.1)
                continue

        # --- Video progress bar logic ---
        if cap_to_use is video_file_cap and video_file_cap is not None:
            try:
                total_frames = video_file_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = video_file_cap.get(cv2.CAP_PROP_FPS)
                current_frame = video_file_cap.get(cv2.CAP_PROP_POS_FRAMES)
                if total_frames > 0 and fps > 0:
                    video_progress = min(current_frame / total_frames, 1.0)
                    total_sec = int(total_frames / fps)
                    cur_sec = int(current_frame / fps)
                    def sec2str(s):
                        return f"{s//60}:{s%60:02d}"
                    video_time_str = f"{sec2str(cur_sec)} / {sec2str(total_sec)}"
            except:
                pass
        else:
            video_progress = 0.0
            video_time_str = "0:00 / 0:00"

        # Save progress info to state file for control window
        vj_state['video_progress'] = video_progress
        vj_state['video_time_str'] = video_time_str
        with open(STATE_FILE, 'w') as f:
            json.dump(vj_state, f)

        # Handle fullscreen toggle - only when state actually changes
        current_fullscreen_state = vj_state.get('fullscreen', False)
        current_monitor = vj_state.get('fullscreen_monitor', 0)

        # Track previous states to detect changes
        if not hasattr(main_video_loop, 'prev_fullscreen_state'):
            main_video_loop.prev_fullscreen_state = False
            main_video_loop.prev_monitor = 0

        # Only change window mode if state actually changed
        if (current_fullscreen_state != main_video_loop.prev_fullscreen_state or
            (current_fullscreen_state and current_monitor != main_video_loop.prev_monitor)):

            safe_window_transition(current_fullscreen_state, current_monitor)
            main_video_loop.prev_fullscreen_state = current_fullscreen_state
            main_video_loop.prev_monitor = current_monitor

        now = time.time()
        global hue_shift
        hue_shift = (hue_shift + 2) % 180

        # Use loaded state for controls
        invert = vj_state['invert']
        mono = vj_state['monochrome']
        mono_hue = vj_state['mono_hue']
        show_boxes = vj_state['show_boxes']
        presentation_mode = vj_state['presentation_mode']

        # Individual box controls
        show_silhouettes = vj_state.get('show_silhouettes', True)
        show_bright = vj_state.get('show_bright', True)
        show_dark = vj_state.get('show_dark', True)
        show_moving = vj_state.get('show_moving', True)
        silhouette_threshold = vj_state.get('silhouette_threshold', 50)

        # Max box counts from state
        config.top_bright_spots = vj_state.get('max_bright', config.top_bright_spots)
        config.top_dark_spots = vj_state.get('max_dark', config.top_dark_spots)
        config.top_moving_spots = vj_state.get('max_moving', config.top_moving_spots)

        # Apply monochrome if enabled
        if mono:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = mono_hue
            hsv[..., 1] = 255
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # Apply invert if enabled
        if invert:
            frame = cv2.bitwise_not(frame)

        full_res_frame = cv2.resize(frame, (1280, 720))

        # --- Dark mode: darken the video feed ---
        dark_overlay = np.full_like(full_res_frame, DARK_BG, dtype=np.uint8)
        alpha = 0.25 if not presentation_mode else 0.5
        frame_dark = cv2.addWeighted(full_res_frame, 1 - alpha, dark_overlay, alpha, 0)

        # Downscale for detection
        proc_frame = cv2.resize(full_res_frame, (640, 360))
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)

        # Only update detection and overlays every 3 frames for performance
        if frame_count % 1 == 0:
            overlay_small = np.zeros_like(proc_frame, dtype=np.uint8)
            box_centers = []
            detection_counts = {'humans': 0, 'bright': 0, 'dark': 0, 'move': 0, 'faces': 0}

            if show_boxes:
                # Initialize bright_spot_centers to avoid reference errors
                bright_spot_centers = []

                # Silhouette detection
                if show_silhouettes:
                    fg_mask = bg_subtractor.apply(proc_frame)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                    # --- use silhouette_threshold from state ---
                    _, fg_mask = cv2.threshold(fg_mask, silhouette_threshold, 255, cv2.THRESH_BINARY)
                    contours_silhouette, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    silhouette_names = ["Ghost", "Shadow", "Echo", "Outline", "Trace"]
                    for idx, contour in enumerate(contours_silhouette):
                        area = cv2.contourArea(contour)
                        if area > config.min_silhouette_area:
                            color = hsv2bgr((hue_shift + idx * 10) % 180)
                            cv2.drawContours(overlay_small, [contour], -1, color, 1)  # 1px thick
                            label = random.choice(silhouette_names)
                            percent = random.randint(0, 100)
                            obj_id = random.randint(1000, 9999)
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.putText(overlay_small, f"Silhouette: {label} [{obj_id}] {{{percent}%}}", (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)

                            # 1/20 chance to invert this silhouette area
                            if random.randint(1, 20) == 1:
                                # Scale coordinates to full resolution
                                scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
                                scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
                                x_full = int(x * scale_x)
                                y_full = int(y * scale_y)
                                w_full = int(w * scale_x)
                                h_full = int(h * scale_y)

                                # Ensure coordinates are within frame bounds
                                x_full = max(0, min(x_full, full_res_frame.shape[1] - w_full))
                                y_full = max(0, min(y_full, full_res_frame.shape[0] - h_full))
                                w_full = min(w_full, full_res_frame.shape[1] - x_full)
                                h_full = min(h_full, full_res_frame.shape[0] - y_full)

                                # Invert the region inside the silhouette bounding box
                                if w_full > 0 and h_full > 0:
                                    frame_dark[y_full:y_full+h_full, x_full:x_full+w_full] = cv2.bitwise_not(frame_dark[y_full:y_full+h_full, x_full:x_full+w_full])

                # Bright spots (configurable amount)
                if show_bright:
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
                    detection_counts['bright'] = len(top_bright_contours)

                    bright_spot_centers = []
                    for idx, cnt in enumerate(top_bright_contours):
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx, cy = x + w // 2, y + h // 2
                        box_w, box_h = max(w // 2, config.min_bright_area), max(h // 2, config.min_bright_area)
                        x1, y1 = cx - box_w // 2, cy - box_h // 2
                        x2, y2 = cx + box_w // 2, cy + box_h // 2
                        color = (255, 255, 255)  # WHITE for bright spot boxes
                        cv2.rectangle(overlay_small, (x1, y1), (x2, y2), color, 1)  # 1px thick
                        label = random.choice(bright_names)
                        percent = random.randint(0, 100)
                        obj_id = random.randint(1000, 9999)
                        text = f"Bright: {label} [{obj_id}] {{{percent}%}}"
                        cv2.putText(overlay_small, text, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)
                        box_centers.append(((cx, cy), (x1, y1, x2, y2), 'bright'))
                        bright_spot_centers.append((cx, cy))

                        # 1/20 chance to invert this box
                        if random.randint(1, 20) == 1:
                            # Scale coordinates to full resolution
                            scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
                            scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
                            x1_full = int(x1 * scale_x)
                            y1_full = int(y1 * scale_y)
                            x2_full = int(x2 * scale_x)
                            y2_full = int(y2 * scale_y)

                            # Ensure coordinates are within frame bounds
                            x1_full = max(0, min(x1_full, full_res_frame.shape[1] - 1))
                            y1_full = max(0, min(y1_full, full_res_frame.shape[0] - 1))
                            x2_full = max(0, min(x2_full, full_res_frame.shape[1] - 1))
                            y2_full = max(0, min(y2_full, full_res_frame.shape[0] - 1))

                            # Invert the region inside the box on the final frame
                            if x2_full > x1_full and y2_full > y1_full:
                                frame_dark[y1_full:y2_full, x1_full:x2_full] = cv2.bitwise_not(frame_dark[y1_full:y2_full, x1_full:x2_full])

                else:
                    detection_counts['bright'] = 0

                # Darkest spots (configurable amount)
                if show_dark:
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
                    detection_counts['dark'] = len(top_dark_contours)

                    for idx, cnt in enumerate(top_dark_contours):
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx, cy = x + w // 2, y + h // 2
                        box_w, box_h = max(w // 2, config.min_dark_area), max(h // 2, config.min_dark_area)
                        x1, y1 = cx - box_w // 2, cy - box_h // 2
                        x2, y2 = cx + box_w // 2, cy + box_h // 2
                        color = hsv2bgr((hue_shift + idx * 10 + 90) % 180)
                        cv2.rectangle(overlay_small, (x1, y1), (x2, y2), color, 1)  # 1px thick
                        label = "Void"
                        percent = random.randint(0, 100)
                        obj_id = random.randint(1000, 9999)
                        text = f"Dark: {label} [{obj_id}] {{{percent}%}}"
                        cv2.putText(overlay_small, text, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)
                        box_centers.append(((cx, cy), (x1, y1, x2, y2), 'dark'))

                        # 1/20 chance to invert this box
                        if random.randint(1, 20) == 1:
                            # Scale coordinates to full resolution
                            scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
                            scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
                            x1_full = int(x1 * scale_x)
                            y1_full = int(y1 * scale_y)
                            x2_full = int(x2 * scale_x)
                            y2_full = int(y2 * scale_y)

                            # Ensure coordinates are within frame bounds
                            x1_full = max(0, min(x1_full, full_res_frame.shape[1] - 1))
                            y1_full = max(0, min(y1_full, full_res_frame.shape[0] - 1))
                            x2_full = max(0, min(x2_full, full_res_frame.shape[1] - 1))
                            y2_full = max(0, min(y2_full, full_res_frame.shape[0] - 1))

                            # Invert the region inside the box on the final frame
                            if x2_full > x1_full and y2_full > y1_full:
                                frame_dark[y1_full:y2_full, x1_full:x2_full] = cv2.bitwise_not(frame_dark[y1_full:y2_full, x1_full:x2_full])

                else:
                    detection_counts['dark'] = 0

                # Moving spots (configurable amount)
                if show_moving and prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    flat_diff = diff.flatten()
                    if len(flat_diff) > config.top_moving_spots:
                        top_moving_idx = np.argpartition(flat_diff, -config.top_moving_spots)[-config.top_moving_spots:]
                        coords_move = [np.unravel_index(idx, diff.shape) for idx in top_moving_idx]
                        detection_counts['move'] = len(coords_move)
                        for idx, (y, x) in enumerate(coords_move):
                            cx, cy = x, y
                            box_w, box_h = config.min_moving_area, config.min_moving_area
                            x1, y1 = cx - box_w // 2, cy - box_h // 2
                            x2, y2 = cx + box_w // 2, cy + box_h // 2
                            color = hsv2bgr((hue_shift + idx * 10 + 120) % 180)
                            cv2.rectangle(overlay_small, (x1, y1), (x2, y2), color, 1)  # 1px thick
                            label = random.choice(moving_names)
                            percent = random.randint(0, 100)
                            obj_id = random.randint(1000, 9999)
                            text = f"Move: {label} [{obj_id}] {{{percent}%}}"
                            cv2.putText(overlay_small, text, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.3, color, 1)
                            box_centers.append(((cx, cy), (x1, y1, x2, y2), 'move'))

                            # 1/20 chance to invert this box
                            if random.randint(1, 20) == 1:
                                # Scale coordinates to full resolution
                                scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
                                scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
                                x1_full = int(x1 * scale_x)
                                y1_full = int(y1 * scale_y)
                                x2_full = int(x2 * scale_x)
                                y2_full = int(y2 * scale_y)

                                # Ensure coordinates are within frame bounds
                                x1_full = max(0, min(x1_full, full_res_frame.shape[1] - 1))
                                y1_full = max(0, min(y1_full, full_res_frame.shape[0] - 1))
                                x2_full = max(0, min(x2_full, full_res_frame.shape[1] - 1))
                                y2_full = max(0, min(y2_full, full_res_frame.shape[0] - 1))

                                # Invert the region inside the box on the final frame
                                if x2_full > x1_full and y2_full > y1_full:
                                    frame_dark[y1_full:y2_full, x1_full:x2_full] = cv2.bitwise_not(frame_dark[y1_full:y2_full, x1_full:x2_full])

                else:
                    detection_counts['move'] = 0

                # Draw lines from each bright spot to the 2 nearest boxes
                if len(box_centers) > 2 and 'bright_spot_centers' in locals():
                    for idx, b_c in enumerate(bright_spot_centers):
                        # Calculate distances to all other boxes
                        distances = []
                        for i, (center, _, _) in enumerate(box_centers):
                            if center != b_c:  # Don't include the bright spot itself
                                dx = center[0] - b_c[0]
                                dy = center[1] - b_c[1]
                                distance = (dx * dx + dy * dy) ** 0.5  # Euclidean distance
                                distances.append((distance, i, center))

                        # Sort by distance and get the 2 nearest
                        distances.sort(key=lambda x: x[0])
                        nearest_boxes = distances[:min(2, len(distances))]

                        # Draw lines to the 2 nearest boxes
                        for distance, box_idx, target_center in nearest_boxes:
                            color = (255, 255, 255)  # WHITE for lines
                            # Draw as thin as possible (1px), anti-aliased line
                            cv2.line(overlay_small, b_c, target_center, color, 1, lineType=cv2.LINE_AA)

                        # 1/5 chance to draw a third line to a random box (not considering distance)
                        if random.randint(1, 5) == 1 and len(distances) > 2:
                            # Get boxes that weren't already connected (exclude the 2 nearest)
                            used_indices = {box_idx for _, box_idx, _ in nearest_boxes}
                            remaining_boxes = [item for item in distances if item[1] not in used_indices]

                            if remaining_boxes:
                                # Pick a random box from the remaining ones
                                random_box = random.choice(remaining_boxes)
                                _, _, random_target_center = random_box
                                color = (255, 255, 255)  # WHITE for lines
                                cv2.line(overlay_small, b_c, random_target_center, color, 1, lineType=cv2.LINE_AA)

            # Save detection counts for control window
            save_detection_counts(detection_counts)

            # Store for reuse in skipped frames
            main_video_loop.last_overlay = overlay_small
        else:
            # Reuse previous overlay
            overlay_small = getattr(main_video_loop, 'last_overlay', np.zeros_like(proc_frame, dtype=np.uint8))

        # Scale overlay to full resolution and apply to frame
        overlay_full = cv2.resize(overlay_small, (1280, 720), interpolation=cv2.INTER_LINEAR)
        final_frame = cv2.addWeighted(frame_dark, 1.0, overlay_full, 1.0, 0)

        # Apply filters based on vj_state
        if vj_state.get('filter_cvr', False):
            final_frame = apply_crt_filter(final_frame)
        if vj_state.get('filter_static', False):
            final_frame = apply_static_filter(final_frame)
        if vj_state.get('filter_grain', False):
            final_frame = apply_grain_filter(final_frame)

        # --- Rolling logic ---
        global roll_offset_x, roll_offset_y
        speed_x = vj_state.get('shift_x', 0)
        speed_y = vj_state.get('shift_y', 0)
        roll_offset_x = (roll_offset_x + speed_x) % final_frame.shape[1]
        roll_offset_y = (roll_offset_y + speed_y) % final_frame.shape[0]
        if speed_x != 0 or speed_y != 0:
            final_frame = apply_shift_filter(final_frame, roll_offset_x, roll_offset_y)

        # --- Tiling logic ---
        allowed_tiles = [1, 4, 9, 16, 25, 36]
        tile_count = vj_state.get('tile_count', 1)
        # Snap to nearest allowed tile count
        tile_count = min(allowed_tiles, key=lambda x: abs(x - tile_count))
        grid_size = int(np.sqrt(tile_count))
        h, w = final_frame.shape[:2]
        tile_h, tile_w = h // grid_size, w // grid_size
        tile_img = cv2.resize(final_frame, (tile_w, tile_h))
        tiled_frame = np.zeros_like(final_frame)
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx >= tile_count:
                    break
                y1, y2 = i * tile_h, (i + 1) * tile_h
                x1, x2 = j * tile_w, (j + 1) * tile_w
                tiled_frame[y1:y2, x1:x2] = tile_img
        final_frame = tiled_frame

        # --- FPS calculation ---
        if not hasattr(main_video_loop, 'last_fps_time'):
            main_video_loop.last_fps_time = time.time()
            main_video_loop.frame_counter = 0
        main_video_loop.frame_counter += 1
        now_fps = time.time()
        if now_fps - main_video_loop.last_fps_time >= 1.0:
            fps = main_video_loop.frame_counter / (now_fps - main_video_loop.last_fps_time)
            # Save FPS to state file for control window
            vj_state['video_fps'] = round(fps, 1)
            with open(STATE_FILE, 'w') as f:
                json.dump(vj_state, f)
            main_video_loop.last_fps_time = now_fps
            main_video_loop.frame_counter = 0

        # --- Letterbox for fullscreen: fill bars with black ---
        if is_fullscreen:
            # Get current monitor size
            monitors = get_monitor_info()
            mon_w, mon_h = 1920, 1080
            if vj_state.get('fullscreen_monitor', 0) < len(monitors):
                _, _, mon_w, mon_h = monitors[vj_state.get('fullscreen_monitor', 0)]
            # Letterbox the frame to fit monitor, black bars
            final_frame = letterbox_image(final_frame, (mon_h, mon_w), fill_color=(0, 0, 0))

        # Display the frame - with error handling
        try:
            cv2.imshow(cv2_window_name, final_frame)
        except cv2.error:
            create_windowed_window()
            cv2.imshow(cv2_window_name, final_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('f'):  # F key to toggle fullscreen
            current_state = load_state()
            current_state['fullscreen'] = not current_state['fullscreen']
            with open(STATE_FILE, 'w') as f:
                json.dump(current_state, f)

        prev_gray = gray.copy()
        frame_count += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def run_main_video_loop():
    main_video_loop()

if __name__ == "__main__":
    # Use multiprocessing to run the video loop in a separate process
    video_proc = multiprocessing.Process(target=run_main_video_loop)
    video_proc.start()
    video_proc.join()
