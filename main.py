import cv2
import numpy as np
import time
import os
import multiprocessing
import platform

# Import helper modules with error handling
print("Loading helper modules...")

# Default fallback functions in case imports fail
def fallback_function(frame, *args, **kwargs):
    """Fallback function that just returns the original frame"""
    return frame

def fallback_window_function(*args, **kwargs):
    """Fallback function for window operations"""
    pass

def fallback_detect_function(*args, **kwargs):
    """Fallback function for detection operations"""
    return np.zeros_like(args[0], dtype=np.uint8) if args else None, []

def fallback_detect_bright_function(*args, **kwargs):
    """Fallback function for bright detection operations"""
    return np.zeros_like(args[0], dtype=np.uint8) if args else None, [], []

# Try importing utils first (most critical)
try:
    from utils import load_state, save_state, save_detection_counts, DetectionConfig, Keybinds, STATE_FILE
    print("✓ Successfully imported utils module")
except ImportError as e:
    print(f"✗ Failed to import utils module: {e}")
    print("  Using fallback state management...")
    # Fallback implementations
    def load_state():
        return {'invert': False, 'monochrome': False, 'mono_hue': 90, 'show_boxes': True,
                'presentation_mode': False, 'fullscreen': False, 'fullscreen_monitor': 0,
                'show_silhouettes': True, 'show_bright': True, 'show_dark': True,
                'show_moving': True, 'max_bright': 10, 'max_dark': 10, 'max_moving': 5,
                'start_video': False, 'running': True, 'silhouette_threshold': 50,
                'tile_count': 1, 'flash_color_mode': 'white', 'trigger_flash': False}

    def save_state(state):
        pass

    def save_detection_counts(counts):
        pass

    class DetectionConfig:
        def __init__(self):
            self.top_bright_spots = 10
            self.top_dark_spots = 10
            self.top_moving_spots = 5
            self.min_silhouette_area = 500
            self.min_dark_area = 5
            self.min_bright_area = 5
            self.min_moving_area = 5

    class Keybinds:
        def __init__(self):
            self.quit = 27

    STATE_FILE = "/tmp/vj_state.json" if platform.system() != "Windows" else "vj_state.json"

# Try importing camera controls
try:
    from camera_controls import init_camera, apply_camera_adjustments, get_video_source_from_state, get_overlay_video_from_state
    print("✓ Successfully imported camera_controls module")
except ImportError as e:
    print(f"✗ Failed to import camera_controls module: {e}")
    print("  Using fallback camera functions...")

    def init_camera():
        try:
            return cv2.VideoCapture(0)
        except:
            return None

    apply_camera_adjustments = fallback_function

    def get_video_source_from_state(vj_state):
        return 'Camera'

    def get_overlay_video_from_state(vj_state):
        return None

# Try importing main effects
try:
    from main_effects import apply_shift_filter, letterbox_image, apply_monochrome_effect, apply_invert_effect, apply_tiling_effect, apply_flash_effect
    print("✓ Successfully imported main_effects module")
except ImportError as e:
    print(f"✗ Failed to import main_effects module: {e}")
    print("  Using fallback effect functions...")

    apply_shift_filter = fallback_function
    apply_monochrome_effect = fallback_function
    apply_invert_effect = fallback_function
    apply_tiling_effect = fallback_function
    apply_flash_effect = fallback_function

    def letterbox_image(image, target_size, fill_color=(0, 0, 0)):
        return cv2.resize(image, (target_size[1], target_size[0]))

# Try importing video filters
try:
    from video_filters import apply_crt_filter, apply_static_filter, apply_grain_filter, apply_dark_mode_overlay
    print("✓ Successfully imported video_filters module")
except ImportError as e:
    print(f"✗ Failed to import video_filters module: {e}")
    print("  Using fallback filter functions...")

    apply_crt_filter = fallback_function
    apply_static_filter = fallback_function
    apply_grain_filter = fallback_function

    def apply_dark_mode_overlay(frame, alpha=0.25, dark_bg=(18, 18, 18)):
        dark_overlay = np.full_like(frame, dark_bg, dtype=np.uint8)
        return cv2.addWeighted(frame, 1 - alpha, dark_overlay, alpha, 0)

# Try importing detection overlays
try:
    from detection_overlays import detect_silhouettes, detect_bright_spots, detect_dark_spots, detect_moving_spots, draw_connecting_lines, apply_inversion_effects
    print("✓ Successfully imported detection_overlays module")
except ImportError as e:
    print(f"✗ Failed to import detection_overlays module: {e}")
    print("  Using fallback detection functions...")

    detect_silhouettes = fallback_detect_function
    detect_dark_spots = fallback_detect_function
    detect_moving_spots = fallback_detect_function
    detect_bright_spots = fallback_detect_bright_function

    def draw_connecting_lines(overlay, bright_spot_centers, all_centers):
        pass

    def apply_inversion_effects(frame, detections, scale_x, scale_y):
        return frame

# Try importing overlay/window management
try:
    from overlay import get_monitor_info, create_fullscreen_window, create_windowed_window, safe_window_transition
    print("✓ Successfully imported overlay module")
except ImportError as e:
    print(f"✗ Failed to import overlay module: {e}")
    print("  Using fallback window functions...")

    def get_monitor_info():
        return [(0, 0, 1920, 1080)]

    def create_fullscreen_window(window_name, monitor_index=0):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return 1920, 1080

    def create_windowed_window(window_name):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    def safe_window_transition(window_name, target_fullscreen, target_monitor, current_fullscreen_state):
        if target_fullscreen:
            create_fullscreen_window(window_name, target_monitor)
            return True
        else:
            create_windowed_window(window_name)
            return False

print("All modules loaded (with fallbacks where necessary)")
print("-" * 50)

# Initialize camera variable but don't create capture object yet
cap = None
prev_gray = None

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

# Flash effect variables
flash_active = False
flash_frames_remaining = 0
FLASH_DURATION = 2  # Number of frames to show flash effect
flash_color_hue = 0  # For rotating color mode

config = DetectionConfig()
keybinds = Keybinds()

hue_shift = 0  # For color cycling

# Global variables for fullscreen state
is_fullscreen = False
fullscreen_monitor = 0  # Default to primary monitor
cv2_window_name = "VJ Video Stream"

# Add these globals to accumulate roll offsets
roll_offset_x = 0
roll_offset_y = 0

# Initialize camera at startup
cap = init_camera()

def main_video_loop():
    global prev_gray, frame_count, is_fullscreen, cap, flash_active, flash_frames_remaining, flash_color_hue
    global hue_shift, roll_offset_x, roll_offset_y

    # Ensure camera is initialized
    if cap is None:
        print("Camera not initialized, attempting to initialize...")
        cap = init_camera()
        if cap is None:
            print("Failed to initialize camera. Exiting.")
            return

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

        # Create window if not created yet
        if not window_created:
            create_windowed_window(cv2_window_name)
            window_created = True

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
        save_state(vj_state)

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

            is_fullscreen = safe_window_transition(cv2_window_name, current_fullscreen_state, current_monitor, is_fullscreen)
            main_video_loop.prev_fullscreen_state = current_fullscreen_state
            main_video_loop.prev_monitor = current_monitor

        now = time.time()
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
            frame = apply_monochrome_effect(frame, mono_hue)
        # Apply invert if enabled
        if invert:
            frame = apply_invert_effect(frame)

        full_res_frame = cv2.resize(frame, (1280, 720))

        # --- Dark mode: darken the video feed ---
        alpha = 0.25 if not presentation_mode else 0.5
        frame_dark = apply_dark_mode_overlay(full_res_frame, alpha, DARK_BG)

        # Downscale for detection
        proc_frame = cv2.resize(full_res_frame, (640, 360))

        # Only update detection and overlays every frame for performance
        if frame_count % 1 == 0:
            overlay_small = np.zeros_like(proc_frame, dtype=np.uint8)
            detection_counts = {'humans': 0, 'bright': 0, 'dark': 0, 'move': 0, 'faces': 0}
            all_detections = []
            all_centers = []

            if show_boxes:
                # Silhouette detection
                if show_silhouettes:
                    silhouette_overlay, silhouette_detections = detect_silhouettes(
                        proc_frame, bg_subtractor, silhouette_threshold, config, hue_shift)
                    overlay_small = cv2.addWeighted(overlay_small, 1.0, silhouette_overlay, 1.0, 0)
                    all_detections.extend(silhouette_detections)
                    detection_counts['humans'] = len(silhouette_detections)

                # Bright spots
                if show_bright:
                    bright_overlay, bright_detections, bright_spot_centers = detect_bright_spots(
                        proc_frame, config, hue_shift)
                    overlay_small = cv2.addWeighted(overlay_small, 1.0, bright_overlay, 1.0, 0)
                    all_detections.extend(bright_detections)
                    all_centers.extend([d['center'] for d in bright_detections])
                    detection_counts['bright'] = len(bright_detections)
                else:
                    bright_spot_centers = []
                    detection_counts['bright'] = 0

                # Dark spots
                if show_dark:
                    dark_overlay, dark_detections = detect_dark_spots(proc_frame, config, hue_shift)
                    overlay_small = cv2.addWeighted(overlay_small, 1.0, dark_overlay, 1.0, 0)
                    all_detections.extend(dark_detections)
                    all_centers.extend([d['center'] for d in dark_detections])
                    detection_counts['dark'] = len(dark_detections)
                else:
                    detection_counts['dark'] = 0

                # Moving spots
                if show_moving:
                    moving_overlay, moving_detections = detect_moving_spots(
                        proc_frame, prev_gray, config, hue_shift)
                    overlay_small = cv2.addWeighted(overlay_small, 1.0, moving_overlay, 1.0, 0)
                    all_detections.extend(moving_detections)
                    all_centers.extend([d['center'] for d in moving_detections])
                    detection_counts['move'] = len(moving_detections)
                else:
                    detection_counts['move'] = 0

                # Draw connecting lines
                draw_connecting_lines(overlay_small, bright_spot_centers, all_centers)

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

            # Apply inversion effects to detected areas
            scale_x = full_res_frame.shape[1] / proc_frame.shape[1]
            scale_y = full_res_frame.shape[0] / proc_frame.shape[0]
            final_frame = apply_inversion_effects(final_frame, all_detections, scale_x, scale_y)

            # Apply filters based on vj_state
            if vj_state.get('filter_cvr', False):
                final_frame = apply_crt_filter(final_frame)
            if vj_state.get('filter_static', False):
                final_frame = apply_static_filter(final_frame)
            if vj_state.get('filter_grain', False):
                final_frame = apply_grain_filter(final_frame)

            # --- Rolling logic ---
            speed_x = vj_state.get('shift_x', 0)
            speed_y = vj_state.get('shift_y', 0)
            roll_offset_x = (roll_offset_x + speed_x) % final_frame.shape[1]
            roll_offset_y = (roll_offset_y + speed_y) % final_frame.shape[0]
            if speed_x != 0 or speed_y != 0:
                final_frame = apply_shift_filter(final_frame, roll_offset_x, roll_offset_y)

            # --- Tiling logic ---
            tile_count = vj_state.get('tile_count', 1)
            final_frame = apply_tiling_effect(final_frame, tile_count)

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
                save_state(vj_state)
                main_video_loop.last_fps_time = now_fps
                main_video_loop.frame_counter = 0

            # Check for flash trigger from control window
            if vj_state.get('trigger_flash', False):
                flash_active = True
                flash_frames_remaining = FLASH_DURATION
                flash_color_hue = (flash_color_hue + 15) % 180
                # Reset the trigger in state file
                vj_state['trigger_flash'] = False
                save_state(vj_state)

            # Apply flash effect if active
            if flash_active and flash_frames_remaining > 0:
                flash_color_mode = vj_state.get('flash_color_mode', 'white')
                flash_intensity = flash_frames_remaining / FLASH_DURATION
                final_frame = apply_flash_effect(final_frame, flash_color_mode, flash_intensity, flash_color_hue)
                flash_frames_remaining -= 1
                if flash_frames_remaining <= 0:
                    flash_active = False

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
                create_windowed_window(cv2_window_name)
                cv2.imshow(cv2_window_name, final_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('f'):  # F key to toggle fullscreen
                current_state = load_state()
                current_state['fullscreen'] = not current_state['fullscreen']
                save_state(current_state)
            elif key == ord(' '):  # Space key to trigger flash
                flash_active = True
                flash_frames_remaining = FLASH_DURATION

        if prev_gray is None:
            prev_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY).copy()
        frame_count += 1

    # Cleanup
    if cap:
        cap.release()
    cv2.destroyAllWindows()

def run_main_video_loop():
    main_video_loop()

if __name__ == "__main__":
    # Use multiprocessing to run the video loop in a separate process
    video_proc = multiprocessing.Process(target=run_main_video_loop)
    video_proc.start()
    video_proc.join()
