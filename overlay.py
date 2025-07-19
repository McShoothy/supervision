"""
Overlay Module
Handles window management, fullscreen transitions, and overlay video functionality
"""
import cv2
import platform
import time
import subprocess

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

def create_fullscreen_window(window_name, monitor_index=0):
    """Create a true fullscreen OpenCV window on specified monitor"""
    monitors = get_monitor_info()

    # Ensure monitor_index is valid
    if monitor_index >= len(monitors):
        monitor_index = 0

    x_offset, y_offset, width, height = monitors[monitor_index]

    # Create or recreate window
    try:
        cv2.destroyWindow(window_name)
    except:
        pass

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Position window on the correct monitor BEFORE setting fullscreen
    cv2.moveWindow(window_name, x_offset + 100, y_offset + 100)
    cv2.resizeWindow(window_name, width, height)

    # Small delay to ensure window is positioned
    time.sleep(0.1)

    # Set to fullscreen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    return width, height

def create_windowed_window(window_name):
    """Create a normal windowed OpenCV window"""
    # Safely destroy existing window
    try:
        cv2.destroyWindow(window_name)
    except:
        pass

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Position on primary monitor
    cv2.moveWindow(window_name, 100, 100)

def safe_window_transition(window_name, target_fullscreen, target_monitor, current_fullscreen_state):
    """Safely transition between window modes without breaking"""
    if target_fullscreen == current_fullscreen_state:
        return current_fullscreen_state  # No change needed

    try:
        if target_fullscreen:
            create_fullscreen_window(window_name, target_monitor)
            return True
        else:
            create_windowed_window(window_name)
            return False
    except Exception as e:
        print(f"Window transition failed: {e}")
        # Emergency fallback - create basic windowed mode
        try:
            cv2.destroyWindow(window_name)
        except:
            pass
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        return False
