# Video Conversion Tool

This script (`vidconv.py`) provides a flexible command-line interface for converting video files. It uses `ffmpeg` and `ffprobe` for video processing, and supports various operations including changing resolution, bitrate, codec, cropping, and cutting the videos.

## Features:

1. **Resolution Change**: Convert videos to various resolutions such as 360p, 480p, 720p, 1080p, and 2160p.
2. **Bitrate Adjustment**: Modify the bitrate to 'low', 'mid', 'high', or specify any numeric value.
3. **Codec Conversion**: Convert videos to H.265 (HEVC) or H.264.
4. **Cropping**: Crop videos for portrait mode on phones.
5. **Cutting**: Extract a specific segment of the video.
6. **File Removal**: Option to remove the original video after conversion.
7. **Bulk Conversion**: Convert all videos in a directory and its subdirectories.

## Prerequisites:

1. `ffmpeg` and `ffprobe` need to be installed on your system.
2. Ensure the script (`vidconv.py`) has the necessary permissions to execute.

## Usage:

```
python vidconv.py INPUT_PATH [OPTIONS]
```

### Arguments:

- `INPUT_PATH`: Path to the input video file or directory.
- `--resolution`: Target video resolution. Default is `1080`.
- `--bitrate`: Target bitrate level. Can be 'low', 'mid', 'high', or a specific value like '4000k'. Default is 'unchanged'.
- `--codec`: Video codec to use. Options are 'h265' or 'h264'. Default is 'h265'.
- `--quality`: Quality value for encoding (0-51). Lower is better but will produce bigger files. Default is `13`.
- `--crop`: Crop video for portrait mode on phones.
- `--remove`: Remove the input file after conversion.
- `--cut`: Cut video from specific timestamps. Provide start and optionally end timestamps.
- `--length`: Length of the video in seconds. If the value is decimal, it's interpreted as minutes.

### Examples:

Convert a single video to 720p with high bitrate:

```
python vidconv.py video.mp4 --resolution 720 --bitrate high
```

Convert all videos in a directory to H.265 with default quality:

```
python vidconv.py /path/to/directory --codec h265
```

Crop and convert a video for phone portrait mode:

```
python vidconv.py video.mp4 --crop
```

Cut a video segment from 2 minutes to 5 minutes:

```
python vidconv.py video.mp4 --cut 00:02:00 00:05:00
```

## Note:

- Always backup your videos before using this tool, especially if you choose the `--remove` option.
- The script checks the properties of the video before conversion. If the video already satisfies the given criteria, it won't be converted to save processing time.

-----------------------------------------------------------

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
