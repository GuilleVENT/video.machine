# Enduro Bike Video Overlay Tool

This tool overlays various KPIs (Key Performance Indicators) on a video taken during an enduro bike ride. It uses the GPS data embedded in the video to generate visualizations such as speed, elevation, and g-force, and then overlays these visualizations on the original video.

## Features:

1. **Elevation GIF**: Shows the elevation profile of the ride with a moving point indicating the current elevation.
2. **Speed Map GIF**: Displays the route on a map with color-coded speed data. A moving marker indicates the current position.
3. **KPIs**: Displays the current speed and g-force on the video.

## Requirements:

- Python 3.x
- Libraries: `sys`, `os`, `logging`, `imageio`, `matplotlib`, `seaborn`, `warnings`, `numpy`, `pandas`, `gpxpy`, `geopy`, `folium`, `selenium`, `time`, `srtm`.
- Firefox browser (for the `selenium` webdriver).
- FFmpeg (for video processing).
- GoPro video with embedded GPS data.

## Usage:

1. Set the `video` variable to the path of your GoPro video.
2. Run the script: `python script_name.py`
3. The script will generate various GIFs and overlay them on the original video. The final output will be saved as `final_out.mp4`.

## Workflow:

1. **Extract GPS Data**: Extracts the GPS data from the GoPro video.
2. **Parse GPX Data**: Parses the extracted GPS data.
3. **Compute Speed and Acceleration**: Calculates speed and acceleration (g-force) from the GPS data.
4. **Generate Elevation GIF**: Creates a GIF showing the elevation profile of the ride.
5. **Generate Speed Map GIF**: Creates a GIF showing the route on a map with color-coded speed data.
6. **Generate KPIs GIF**: Creates a GIF displaying the current speed and g-force.
7. **Overlay All Elements**: Overlays all the generated GIFs on the original video.

## Notes:

- All temporary files are saved in a `temp` directory. This directory can be deleted after processing to clean up temporary files.
- The Firefox browser runs in the background (headless mode) during processing.
