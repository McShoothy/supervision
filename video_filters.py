"""
Video Filters Module
Handles video filter effects like CRT, static, grain, and other visual filters
"""
import cv2
import numpy as np

def apply_crt_filter(frame):
    """Simulate CRT/VHS effect: add scanlines and slight color shift"""
    out = frame.copy()
    for y in range(0, out.shape[0], 2):
        out[y:y+1, :, :] = (out[y:y+1, :, :] * 0.7).astype(np.uint8)
    # Slight color shift
    out[..., 1] = np.clip(out[..., 1] * 0.95, 0, 255).astype(np.uint8)
    return out

def apply_static_filter(frame):
    """Add random static noise"""
    noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
    out = cv2.add(frame, noise)
    return out

def apply_grain_filter(frame):
    """Film grain effect (currently a passthrough)"""
    return frame

def apply_vlines_filter(frame):
    """Vertical lines effect (placeholder for future implementation)"""
    return frame

def apply_dark_mode_overlay(frame, alpha=0.25, dark_bg=(18, 18, 18)):
    """Apply dark mode overlay to darken the video feed"""
    dark_overlay = np.full_like(frame, dark_bg, dtype=np.uint8)
    return cv2.addWeighted(frame, 1 - alpha, dark_overlay, alpha, 0)
