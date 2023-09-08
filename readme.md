
    # Video Conversion Tool

    This script provides an easy way to convert video files using `ffmpeg`. You can specify desired resolutions, bitrates, codecs, and even crop the videos for portrait mode on phones.

    ## Requirements:
    - `ffmpeg`
    - `ffprobe`

    ## Usage:

    ```bash
    python script_name.py <input_path> [--resolution RESOLUTION] [--bitrate BITRATE] [--codec CODEC] [--crop]
    ```

    - `<input_path>`: Path to the video file or a directory containing multiple video files.
    - `--resolution`: Target video resolution (Default: 1080p). Choices: ["360", "480", "720", "1080", "2160"].
    - `--bitrate`: Target bitrate level (Default: unchanged). Choices: ["low", "mid", "high", "unchanged"].
    - `--codec`: Video codec to use (Default: h265). Choices: ["h265", "h264"].
    - `--crop`: Crop video for portrait mode on phones.

    Example:

    ```bash
    python script_name.py input_video.mp4 --resolution 720 --bitrate mid --codec h265 --crop
    ```

    Replace `script_name.py` with the name of this script and `input_video.mp4` with your video's name or path.
    
