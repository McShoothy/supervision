# Supervision

Supervision is a very basic, custom-built VJ software costume made for an event themed around AI vision and cyber warfare.

This project is a real-time video processing and visualization tool written in Python. It captures video from a camera or an MP4 file, detects features like humans, faces, and movement, and highlights bright, dark, and moving spots. Visual effects and basic annotations are overlaid in real-time.

The application provides controls for processing and display options through a separate GUI window (built with DearPyGui), which communicates with the main video window using a shared JSON state file. This architecture allows for responsive, live tweaking of effects and detections during a performance.

### Key Features
- Real-time video capture from camera or file (MP4)
- Detection of people, faces, and moving/bright/dark spots
- Customizable visual overlays and effects
- Separate control window (DearPyGui) for adjusting parameters live
- Fullscreen and tiled display modes
- Multiple visual filters
- Designed for live VJing or interactive art installations
- Main focus is on playful, experimental detection and annotation features

**Note:** The detection features are intentionally basic and experimental, prioritizing visual impact and flexibility for live performance over accuracy.

### TODO: 
1. add config files to the GUI and presets. 
2. optimize for better framerate on edge devices. 
