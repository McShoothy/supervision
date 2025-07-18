import dearpygui.dearpygui as dpg
import json
import os
import time
import subprocess
import glob

# Control state file for communication between windows
STATE_FILE = "/tmp/vj_state.json"

# State variables for controls
vj_state = {
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
    'max_bright': 20,
    'max_dark': 10,
    'max_moving': 5,
    'start_video': False,
    'camera_brightness': 50,
    'camera_contrast': 50,
    'camera_hue': 50,
    'camera_index': 0,  # Default to camera 0
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
    'tile_count': 1,
    'overlay_video': "",
    'overlay_enabled': False,
    'silhouette_threshold': 50,
    'flash_color_mode': 'white',  # Default flash color mode
}

# FPS tracking
fps_counter = 0
fps_display = 0
fps_last_time = time.time()

def save_state():
    """Save current state to file for video window to read"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(vj_state, f)
    except:
        pass

def on_invert(sender, app_data):
    vj_state['invert'] = app_data
    save_state()

def on_mono(sender, app_data):
    vj_state['monochrome'] = app_data
    save_state()

def on_hue(sender, app_data):
    vj_state['mono_hue'] = app_data
    save_state()

def on_show_boxes(sender, app_data):
    vj_state['show_boxes'] = app_data
    save_state()

def on_presentation(sender, app_data):
    vj_state['presentation_mode'] = app_data
    save_state()

def on_fullscreen(sender, app_data):
    vj_state['fullscreen'] = app_data
    save_state()

def on_fullscreen_monitor(sender, app_data):
    vj_state['fullscreen_monitor'] = app_data
    save_state()

def on_show_silhouettes(sender, app_data):
    vj_state['show_silhouettes'] = app_data
    save_state()

def on_show_bright(sender, app_data):
    vj_state['show_bright'] = app_data
    save_state()

def on_show_dark(sender, app_data):
    vj_state['show_dark'] = app_data
    save_state()

def on_show_moving(sender, app_data):
    vj_state['show_moving'] = app_data
    save_state()

def on_max_bright(sender, app_data):
    vj_state['max_bright'] = app_data
    save_state()

def on_max_dark(sender, app_data):
    vj_state['max_dark'] = app_data
    save_state()

def on_max_moving(sender, app_data):
    vj_state['max_moving'] = app_data
    save_state()

def on_start_video(sender, app_data):
    vj_state['start_video'] = app_data
    save_state()

def on_camera_brightness(sender, app_data):
    vj_state['camera_brightness'] = app_data
    save_state()

def on_camera_contrast(sender, app_data):
    vj_state['camera_contrast'] = app_data
    save_state()

def on_camera_hue(sender, app_data):
    vj_state['camera_hue'] = app_data
    save_state()

def on_reset_camera():
    """Reset camera controls to default values"""
    vj_state['camera_brightness'] = 50
    vj_state['camera_contrast'] = 50
    vj_state['camera_hue'] = 50
    save_state()
    # Update the GUI sliders
    dpg.set_value("brightness_slider", 50)
    dpg.set_value("contrast_slider", 50)
    dpg.set_value("hue_slider", 50)

def on_filter_cvr(sender, app_data):
    vj_state['filter_cvr'] = app_data
    save_state()

def on_filter_static(sender, app_data):
    vj_state['filter_static'] = app_data
    save_state()

def on_filter_grain(sender, app_data):
    vj_state['filter_grain'] = app_data
    save_state()

def on_filter_vlines(sender, app_data):
    vj_state['filter_vlines'] = app_data
    save_state()

def on_shift_x(sender, app_data):
    vj_state['shift_x'] = app_data
    save_state()

def on_shift_y(sender, app_data):
    vj_state['shift_y'] = app_data
    save_state()

def on_static_intensity(sender, app_data):
    vj_state['static_intensity'] = app_data
    save_state()

def on_grain_intensity(sender, app_data):
    vj_state['grain_intensity'] = app_data
    save_state()

def on_vlines_intensity(sender, app_data):
    vj_state['vlines_intensity'] = app_data
    save_state()

def on_tile_count(sender, app_data):
    vj_state['tile_count'] = app_data
    save_state()

def on_exit():
    vj_state['running'] = False
    save_state()
    dpg.stop_dearpygui()

def get_available_monitors():
    """Get list of available monitors for display in control panel"""
    try:
        result = subprocess.run(['xrandr', '--current'], capture_output=True, text=True)
        if result.returncode != 0:
            return ["Monitor 0 (Primary)"]

        monitors = []
        lines = result.stdout.split('\n')
        monitor_index = 0

        for line in lines:
            if ' connected' in line:
                # Extract monitor name and resolution
                parts = line.split()
                monitor_name = parts[0]
                resolution = "Unknown"

                for part in parts:
                    if 'x' in part and '+' in part:
                        resolution = part.split('+')[0]
                        break

                if monitor_index == 0:
                    monitors.append(f"Monitor {monitor_index}: {monitor_name} ({resolution}) - Primary")
                else:
                    monitors.append(f"Monitor {monitor_index}: {monitor_name} ({resolution})")
                monitor_index += 1

        if not monitors:
            monitors = ["Monitor 0 (Primary)"]

        return monitors
    except:
        return ["Monitor 0 (Primary)"]

def on_overlay_video(sender, app_data):
    vj_state['overlay_video'] = app_data
    save_state()

def on_overlay_enable(sender, app_data):
    vj_state['overlay_enabled'] = app_data
    save_state()

def on_silhouette_threshold(sender, app_data):
    vj_state['silhouette_threshold'] = app_data
    save_state()

def on_trigger_flash():
    """Trigger flash effect in video window"""
    vj_state['trigger_flash'] = True
    save_state()

def on_flash_color_mode(sender, app_data):
    """Callback for flash color mode selection"""
    vj_state['flash_color_mode'] = app_data.lower()
    save_state()

def on_video_source(sender, app_data):
    """Callback for switching between camera and video file as video source."""
    vj_state['video_source'] = app_data
    save_state()

def on_video_file(sender, app_data):
    """Callback for selecting a video file as the source."""
    vj_state['selected_video_file'] = app_data
    save_state()

def on_camera_selection(sender, app_data):
    """Callback for selecting which camera to use."""
    # Extract camera index from the selected string (e.g., "Camera 0" -> 0)
    camera_index = int(app_data.split()[-1])
    vj_state['camera_index'] = camera_index
    save_state()

def get_video_duration(filepath):
    """Get video duration in seconds using OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return frames / fps
        return 0
    except:
        return 0

def get_video_files_with_durations():
    video_dir = "./videos"
    files = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        files.extend(glob.glob(os.path.join(video_dir, ext)))
    file_list = [os.path.basename(f) for f in files]
    durations = {}
    for f in files:
        durations[os.path.basename(f)] = get_video_duration(f)
    return file_list, durations

def get_video_files():
    video_dir = "./videos"
    files = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        files.extend(glob.glob(os.path.join(video_dir, ext)))
    return [os.path.basename(f) for f in files]

# Get available monitors for display
available_monitors = get_available_monitors()
max_monitor_index = len(available_monitors) - 1

# Get video files for video file combo
video_files = get_video_files()
if not video_files:
    video_files = ["No videos found"]

# Initialize DearPyGui
dpg.create_context()

# Create the controls window
with dpg.window(label="VJ Controls", width=450, height=1000, tag="control_window"):
    dpg.add_text("VJ Control Panel", color=(255, 255, 0))
    dpg.add_separator()

    # Video Stream Controls
    dpg.add_text("Video Stream:", color=(255, 150, 50))
    dpg.add_checkbox(label="Start Video Stream", default_value=False, callback=on_start_video)
    dpg.add_text("Check this to start the video window", color=(150, 150, 150))

    dpg.add_separator()

    # Camera Controls
    dpg.add_text("Camera Controls:", color=(150, 255, 150))
    dpg.add_slider_int(label="Brightness", default_value=50, min_value=0, max_value=100, callback=on_camera_brightness, tag="brightness_slider")
    dpg.add_slider_int(label="Contrast", default_value=50, min_value=0, max_value=100, callback=on_camera_contrast, tag="contrast_slider")
    dpg.add_slider_int(label="Hue Shift", default_value=50, min_value=0, max_value=100, callback=on_camera_hue, tag="hue_slider")
    dpg.add_button(label="Reset Camera Controls", callback=lambda: on_reset_camera())

    dpg.add_separator()

    # Video Filters
    dpg.add_text("Video Filters:", color=(255, 100, 255))
    dpg.add_checkbox(label="CRT/VHS Filter", default_value=False, callback=on_filter_cvr)
    dpg.add_checkbox(label="Static Noise", default_value=False, callback=on_filter_static)
    dpg.add_slider_int(label="Static Intensity", default_value=50, min_value=0, max_value=100, callback=on_static_intensity)
    dpg.add_checkbox(label="Film Grain", default_value=False, callback=on_filter_grain)
    dpg.add_slider_int(label="Grain Intensity", default_value=50, min_value=0, max_value=100, callback=on_grain_intensity)
    dpg.add_checkbox(label="Vertical Lines", default_value=False, callback=on_filter_vlines)
    dpg.add_slider_int(label="V-Lines Intensity", default_value=50, min_value=0, max_value=100, callback=on_vlines_intensity)

    dpg.add_text("Image Roll (Speed):", color=(200, 200, 255))
    dpg.add_slider_int(label="Roll X Speed", default_value=0, min_value=-100, max_value=100, callback=on_shift_x)
    dpg.add_slider_int(label="Roll Y Speed", default_value=0, min_value=-100, max_value=100, callback=on_shift_y)
    dpg.add_text("Tiling:", color=(200, 200, 255))
    dpg.add_combo(label="Tiles (NxN)", items=["1", "4", "9", "16", "25", "36"], default_value="1", callback=lambda s, a: on_tile_count(s, int(a)), tag="tile_combo")

    dpg.add_separator()

    # Main Effects
    dpg.add_text("Main Effects:", color=(100, 255, 100))
    dpg.add_checkbox(label="Invert Colors", default_value=False, callback=on_invert)
    dpg.add_checkbox(label="Monochrome Mode", default_value=False, callback=on_mono)
    dpg.add_slider_int(label="Monochrome Hue", default_value=90, min_value=0, max_value=179, callback=on_hue)
    dpg.add_checkbox(label="Presentation Mode", default_value=False, callback=on_presentation)

    dpg.add_separator()
    dpg.add_text("Fullscreen Settings:", color=(255, 100, 100))
    dpg.add_checkbox(label="Fullscreen Video", default_value=False, callback=on_fullscreen)

    # Display available monitors
    dpg.add_text("Available Monitors:", color=(150, 150, 150))
    for i, monitor_info in enumerate(available_monitors):
        dpg.add_text(f"  {monitor_info}", color=(120, 120, 120))

    dpg.add_slider_int(label="Monitor Selection", default_value=0, min_value=0, max_value=max_monitor_index, callback=on_fullscreen_monitor)
    dpg.add_text("Press 'F' in video window to toggle fullscreen", color=(150, 150, 150))
    dpg.add_text("⚠️ Fullscreen will appear on selected monitor", color=(255, 200, 100))

    dpg.add_separator()

    # Box Controls
    dpg.add_text("Detection Overlays:", color=(100, 255, 100))
    dpg.add_checkbox(label="Show All Boxes", default_value=True, callback=on_show_boxes)

    dpg.add_text("Individual Box Types:", color=(200, 200, 200))
    dpg.add_checkbox(label="Show Silhouettes", default_value=True, callback=on_show_silhouettes)
    dpg.add_slider_int(label="Silhouette Threshold", default_value=50, min_value=0, max_value=255, callback=on_silhouette_threshold, tag="silhouette_threshold_slider")
    dpg.add_checkbox(label="Show Bright Spots", default_value=True, callback=on_show_bright)
    dpg.add_slider_int(label="Max Bright Spots", default_value=10, min_value=1, max_value=100, callback=on_max_bright)
    dpg.add_checkbox(label="Show Dark Spots", default_value=True, callback=on_show_dark)
    dpg.add_slider_int(label="Max Dark Spots", default_value=10, min_value=1, max_value=100, callback=on_max_dark)
    dpg.add_checkbox(label="Show Moving Spots", default_value=True, callback=on_show_moving)
    dpg.add_slider_int(label="Max Moving Spots", default_value=5, min_value=1, max_value=100, callback=on_max_moving)

    dpg.add_separator()

    # Flash Control
    dpg.add_text("Flash Effect:", color=(255, 255, 100))
    dpg.add_button(label="⚡ FLASH ⚡", callback=lambda: on_trigger_flash(), width=200, height=40)
    dpg.add_combo(label="Flash Color", items=["White", "Black", "Red", "Color"], default_value="White", callback=on_flash_color_mode, tag="flash_color_combo")
    dpg.add_text("Click to trigger flash effect", color=(150, 150, 150))
    dpg.add_text("Also works with SPACE key in video window", color=(150, 150, 150))
    dpg.add_text("'Color' mode cycles through rainbow colors", color=(150, 150, 150))

    dpg.add_separator()

    # Performance & Stats
    dpg.add_text("Performance:", color=(100, 255, 100))
    dpg.add_text("FPS: 0", tag="fps_display", color=(255, 255, 100))

    dpg.add_separator()
    dpg.add_text("Detection Counts:")
    dpg.add_text("Humans: 0", tag="human_count")
    dpg.add_text("Bright Spots: 0", tag="bright_count")
    dpg.add_text("Dark Spots: 0", tag="dark_count")
    dpg.add_text("Moving Spots: 0", tag="move_count")
    dpg.add_text("Faces: 0", tag="face_count")

    dpg.add_separator()
    dpg.add_button(label="Exit VJ Program", callback=lambda: on_exit())
    dpg.add_text("This window controls the video stream in the other window.")

    dpg.add_separator()
    dpg.add_text("Overlay MP4:", color=(255, 200, 100))
    overlay_files = get_video_files()
    if not overlay_files:
        overlay_files = ["No videos found"]
    dpg.add_combo(label="Select Overlay Video", items=overlay_files, default_value=overlay_files[0], callback=on_overlay_video, tag="overlay_video_combo")
    dpg.add_checkbox(label="Enable Overlay Video", default_value=False, callback=on_overlay_enable, tag="overlay_enable_checkbox")
    dpg.add_text("When enabled, overlays selected video on top of camera feed.", color=(150, 150, 150))

    dpg.add_separator()
    dpg.add_text("Camera Selection:", color=(255, 150, 50))
    dpg.add_combo(label="Select Camera", items=["Camera 0", "Camera 1", "Camera 2", "Camera 3"], default_value="Camera 0", callback=on_camera_selection, tag="camera_selection_combo")
    dpg.add_text("Select which camera device to use for video input", color=(150, 150, 150))

def update_counts():
    """Read detection counts from state file and update FPS"""
    global fps_counter, fps_display, fps_last_time

    # Update FPS counter
    fps_counter += 1
    current_time = time.time()
    if current_time - fps_last_time >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_last_time = current_time
        dpg.set_value("fps_display", f"FPS: {fps_display}")

    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # Use video_fps from state if available
                fps_val = state.get('video_fps', None)
                if fps_val is not None:
                    dpg.set_value("fps_display", f"FPS: {fps_val}")
                if 'detection_counts' in state:
                    counts = state['detection_counts']
                    dpg.set_value("human_count", f"Humans: {counts.get('humans', 0)}")
                    dpg.set_value("bright_count", f"Bright Spots: {counts.get('bright', 0)}")
                    dpg.set_value("dark_count", f"Dark Spots: {counts.get('dark', 0)}")
                    dpg.set_value("move_count", f"Moving Spots: {counts.get('move', 0)}")
                    dpg.set_value("face_count", f"Faces: {counts.get('faces', 0)}")
    except:
        pass

def update_video_progress():
    """Read progress info from state file and update progress bar."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                progress = state.get('video_progress', 0.0)
                time_str = state.get('video_time_str', "0:00 / 0:00")
                dpg.set_value("video_progress_bar", progress)
                dpg.set_value("video_progress_text", time_str)
    except:
        pass

# Setup viewport and show
dpg.create_viewport(title="VJ Controls", width=500, height=900)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("control_window", True)

# Save initial state
save_state()

# Main loop
while dpg.is_dearpygui_running():
    update_counts()
    update_video_progress()
    dpg.render_dearpygui_frame()
    time.sleep(0.001)

dpg.destroy_context()
