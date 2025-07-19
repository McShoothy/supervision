"""
Utilities Module
Shared utility functions and state management
"""
import cv2
import numpy as np
import json
import os
import platform

# Control state file for communication between windows
STATE_FILE = "/tmp/vj_state.json" if platform.system() != "Windows" else "vj_state.json"

def hsv2bgr(h, s=255, v=255):
    """Convert hue (0-179) to BGR color tuple."""
    color = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0,0])

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
        'tile_count': 1,
        'overlay_video': "",
        'overlay_enabled': False,
        'flash_color_mode': 'white',
        'trigger_flash': False,
    }
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                return {**default_state, **state}
    except:
        pass
    return default_state

def save_state(state):
    """Save state to file"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except:
        pass

def save_detection_counts(counts):
    """Save detection counts to state file for control window"""
    try:
        state = load_state()
        state['detection_counts'] = counts
        save_state(state)
    except:
        pass

# --- Configuration class ---
class DetectionConfig:
    def __init__(self):
        self.top_bright_spots = 10  # default to 10
        self.top_dark_spots = 10
        self.top_moving_spots = 5
        self.min_silhouette_area = 500
        self.min_dark_area = 5
        self.min_bright_area = 5
        self.min_moving_area = 5

# --- Keybinds class ---
class Keybinds:
    def __init__(self):
        self.quit = 27  # ESC
        self.toggle_bw = ord('b')
        self.toggle_fullscreen = ord('f')
        self.toggle_presentation = ord('p')
        self.increase_bright = 83  # Right arrow key
        self.decrease_bright = 81  # Left arrow key
