"""
Camera Controls Module
Handles camera initialization, settings, and cross-platform compatibility
"""
import cv2
import platform
import os

def init_camera():
    """Initialize camera with cross-platform compatibility"""
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
        return cap
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return None

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

def get_video_source_from_state(vj_state):
    """Return video source (Camera or Video File) from state."""
    return vj_state.get('video_source', 'Camera')

def get_overlay_video_from_state(vj_state):
    """Return overlay video path if enabled, else None."""
    if vj_state.get('overlay_enabled', False):
        overlay_file = vj_state.get('overlay_video', None)
        if overlay_file:
            overlay_path = os.path.join("./videos", overlay_file)
            if os.path.exists(overlay_path):
                return overlay_path
    return None
