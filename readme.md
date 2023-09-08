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
- `--remove`: Remove the original video file after conversion.


## Examples:

```bash
python vidconv.py input_video.mp4 
```
This will reduce (if higher) the resolution to 1080 and the codec to h265. 

### 1. Save Space by Lowering Bitrate:
If you're running out of storage or want to save some space, you can compromise the bitrate of the video. Reducing the bitrate will lower the file size, though it may also reduce video quality.

```bash
# Reduce bitrate to 'low' for a 1080p video
python script_name.py input_video.mp4 --resolution 1080 --bitrate low
```

### 2. Convert to Lower Resolution:
Another way to save space is to convert the video to a lower resolution. This can be useful when sharing videos on platforms that do not require high-definition content.
```bash
# Convert video to 480p with a 'mid' bitrate setting
python script_name.py input_video.mp4 --resolution 480 --bitrate mid
```

### 3. Sharing on Social Media:
For platforms like Instagram, videos are often viewed on mobile screens, and it's better to just crop the video for Reels format 

```bash
# Convert video for Instagram with 1080p resolution, unchanged bitrate, and cropped for portrait mode
python script_name.py input_video.mp4 --crop
```

### 4. Using a More Efficient Codec:
The H.265 (or HEVC) codec offers better compression than its predecessor, H.264. This means you can often get similar video quality in a smaller file size with H.265.

```bash
# Convert video to H.265 codec for better compression
python script_name.py input_video.mp4 --codec h265
```
NOTE: This is on by default. More codec (+ output file container) options coming in future updates. 

### 5. Preserving Quality:
If you don't want to compromise on video quality but need to convert it for compatibility reasons, you can keep the bitrate and resolution unchanged.

```bash
# Convert video codec to H.265 without changing bitrate or resolution
python script_name.py input_video.mp4 --codec h265 --bitrate unchanged
```
Remember, video editing is a balance between file size and quality. Depending on the platform and the audience, you might prioritize one over the other. Adjust the parameters as necessary for your specific needs.

### 6. Remove Original File:
If you want to delete the original video file after the conversion process is completed, you can use the --remove option. This can be helpful to save space if you don't need the original file anymore.

```bash
# Convert video to 720p resolution and delete the original file after conversion
python script_name.py input_video.mp4 --resolution 720 --remove
```
**Warning**: Using the --remove option will permanently delete the original video file. Ensure you have a backup if needed.
