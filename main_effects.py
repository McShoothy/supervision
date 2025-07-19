"""
Main Effects Module
Handles core visual effects including image roll, tiling, monochrome, invert, and letterboxing
"""
import cv2
import numpy as np

def apply_shift_filter(frame, shift_x, shift_y):
    """Roll the image horizontally and vertically"""
    return np.roll(np.roll(frame, shift_y, axis=0), shift_x, axis=1)

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

def apply_monochrome_effect(frame, mono_hue):
    """Apply monochrome effect with specified hue"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = mono_hue
    hsv[..., 1] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_invert_effect(frame):
    """Apply color inversion effect"""
    return cv2.bitwise_not(frame)

def apply_tiling_effect(frame, tile_count):
    """Apply tiling effect with specified number of tiles"""
    allowed_tiles = [1, 4, 9, 16, 25, 36]
    # Snap to nearest allowed tile count
    tile_count = min(allowed_tiles, key=lambda x: abs(x - tile_count))
    grid_size = int(np.sqrt(tile_count))
    h, w = frame.shape[:2]
    tile_h, tile_w = h // grid_size, w // grid_size
    tile_img = cv2.resize(frame, (tile_w, tile_h))
    tiled_frame = np.zeros_like(frame)
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx >= tile_count:
                break
            y1, y2 = i * tile_h, (i + 1) * tile_h
            x1, x2 = j * tile_w, (j + 1) * tile_w
            tiled_frame[y1:y2, x1:x2] = tile_img
    return tiled_frame

def apply_flash_effect(frame, flash_color_mode, flash_intensity, flash_color_hue):
    """Apply flash effect with specified color mode and intensity"""
    try:
        from utils import hsv2bgr
    except ImportError:
        # Fallback hsv2bgr function if utils import fails
        def hsv2bgr(h, s=255, v=255):
            color = np.uint8([[[h, s, v]]])
            return tuple(int(c) for c in cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0,0])

    if flash_color_mode == 'white':
        flash_overlay = np.full_like(frame, 255, dtype=np.uint8)
    elif flash_color_mode == 'black':
        flash_overlay = np.full_like(frame, 0, dtype=np.uint8)
    elif flash_color_mode == 'red':
        flash_overlay = np.full_like(frame, [0, 0, 255], dtype=np.uint8)  # BGR format
    elif flash_color_mode == 'color':
        color_bgr = hsv2bgr(flash_color_hue, 255, 255)
        flash_overlay = np.full_like(frame, color_bgr, dtype=np.uint8)
    else:
        # Default to white if unknown mode
        flash_overlay = np.full_like(frame, 255, dtype=np.uint8)

    # Apply flash with specified intensity
    return cv2.addWeighted(frame, 1 - flash_intensity, flash_overlay, flash_intensity, 0)
