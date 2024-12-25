import os
import argparse
import subprocess
import sys
import re
import json
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
#logging.info("To Do: improve logging")
logging.info("Video Converter Initialized.")

# ------------------------------------------------------------
# Global/Module-level Constants & Helper Dictionaries
# ------------------------------------------------------------

## AVAILABLE FILE EXTENSIONS
VALID_VIDEO_EXTENSIONS = ['mp4', 'mkv', 'avi', 'mov']

## BITRATE settings
BITRATES = {
    'low': {
        '360': '700k',
        '480': '1200k',
        '720': '2500k',
        '1080': '5000k',
        '2160': '20000k'
    },
    'mid': {
        '360': '1000k',
        '480': '1500k',
        '720': '4000k',
        '1080': '8000k',
        '2160': '35000k'
    },
    'high': {
        '360': '1500k',
        '480': '1800k',
        '720': '7500k',
        '1080': '16000k',
        '2160': '50000k'
    }
}

# GPX Template
GPX_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="VideoConverter">
  <trk>
    <name>Video GPS Track</name>
    <trkseg>
    </trkseg>
  </trk>
</gpx>
"""

# ----------------------------------------------------------
# Function to create temporary gpx file to fill up
# ----------------------------------------------------------
def write_gpx_template():
    temp_dir = tempfile.gettempdir()
    gpx_template_path = os.path.join(temp_dir, "gpx.fmt")
    with open(gpx_template_path, "w") as f:
        f.write(GPX_TEMPLATE)
    return gpx_template_path

# ------------------------------------------------------------
# VideoConverter Class
# ------------------------------------------------------------
class VideoConverter:
    """
    A class responsible for handling video conversions with ffmpeg and ffprobe.

    Attributes
    ----------
        TODO
    """

    def __init__(
            self,
            resolution: str = "1080",
            bitrate: str = "unchanged",
            codec: str = "h265",
            quality: int = 0,
            crop: bool = False,
            remove: bool = False,
            cut = None,               
            length = None,            
            rm_audio: bool = False,
            keep_gps: bool = False,
            extract_gpx: bool = False
        ):
        """
        Initialize the VideoConverter with an `options` dictionary.

        Parameters
        ----------
            - 'resolution': Target video resolution (e.g., "1080").
            - 'bitrate': Target bitrate level (e.g., "low", "mid", "high", "unchanged").
            - 'codec': Video codec to use (e.g., "h265", "h264").
            - 'quality': Encoding Quality. Lower is better, but will produce bigger files.
            - 'crop': Boolean indicating if video should be cropped for portrait mode on phones.
            - 'remove': Boolean indicating if the input file should be removed after conversion.
            - 'cut': List containing start (and optionally end) timestamps for cutting the video.
            - 'length': Float indicating the length of the video in minutes. Fractions represent additional seconds (e.g., 1.5 = 1 minute 30 seconds).
            - 'rm_audio': Whether to remove audio track of the video.
            - 'keep_gps': Whether to preserve GPS metadata in the output video.
            - 'extract_gpx': Whether to extract GPS data into a separate GPX file.
        """
        
        self.resolution = resolution
        self.bitrate = bitrate
        self.codec = codec
        self.quality = quality
        self.crop = crop 
        self.remove = remove
        self.cut = cut
        self.length = length
        self.rm_audio = rm_audio
        self.keep_gps = keep_gps
        self.extract_gpx = extract_gpx

        # Immediately validate all options.
        self.validate_options()

        for required_dependency in ['ffmpeg', 'ffprobe', 'exiftool']:
            self.check_command_availability(required_dependency)
            ## TO DO check for other availability 

        self.hw_acceleration = self.get_available_hw_acceleration()

    def get_available_hw_acceleration(self): 
        
        cmd = ["ffmpeg", "-hwaccels"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to query ffmpeg -hwaccels: {result.stderr}")

        hwaccels = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and not line.startswith("Hardware acceleration methods:"):
                hwaccels.append(line)


        # Hard-coded priority list
        priority = ["videotoolbox", "cuda", "qsv", "vaapi", "dxva2", "amf"]
    
        for method in priority:
            if method in hwaccels:
                return method
        return None ## Fallback to software

        '''
        to run -> 
        if hw == "videotoolbox":
            cmd.extend(["-c:v", "hevc_videotoolbox"])
        elif hw == "cuda":
            cmd.extend(["-c:v", "hevc_nvenc"])
        elif hw == "qsv":
            cmd.extend(["-c:v", "hevc_qsv"])
        # ...
        else:
            # fallback to software
            cmd.extend(["-c:v", "libx265"])
        '''

    def validate_options(self):
        """
        Validate and normalize each option in self.options.
        Raises ValueError if any option is invalid.
        """

        # 1. Resolution
        valid_resolutions = ['360', '480', '720', '1080', '2160']
        levels = ['low', 'mid', 'high']
        for level in levels:
            for resolution in valid_resolutions: 
                if resolution not in BITRATES.get(level, {}):
                    raise ValueError(f"Missing bitrate entry for level '{level}' and resolution '{resolution}'.")

        # 2. bitrate
        # Allowed: 'unchanged', 'low', 'mid', 'high', or something like '4000k', '4M', etc.
        valid_bitrate_keywords = ["unchanged", "low", "mid", "high"]
        if self.bitrate not in valid_bitrate_keywords:
            # If not a known keyword, ensure it's numeric w/ optional K or M (e.g. 4000k).
            pattern = r"^\d+[kKmM]?$"
            if not re.match(pattern, self.bitrate):
                raise ValueError(
                    f"Invalid bitrate '{self.bitrate}'. "
                    "Must be 'unchanged', 'low', 'mid', 'high', "
                    "or a numeric value ending in 'k'/'K'/'M'/'m' (e.g. '4000k')."
                )

        # 3. codec
        valid_codecs = ["h264", "h265"]
        if self.codec not in valid_codecs:
            raise ValueError(
                f"Invalid codec '{self.codec}'. Must be one of {valid_codecs}."
            )

        # 4. quality
        # For x264/x265, 0..51 is typical CRF range. 0=lossless, 51=worst quality.
        if not isinstance(self.quality, int) or not (0 <= self.quality <= 51):
            raise ValueError(
                f"Invalid quality '{self.quality}'. Must be an integer in [0..51]."
            )
        
        # 5. crop (boolean)
        if not isinstance(self.crop, bool):
            raise ValueError("Option 'crop' must be True or False.")

        # 6. remove original file (boolean)
        if not isinstance(self.remove, bool):
            raise ValueError("Option 'remove' must be True or False.")

        # 7. cut (list or None)
        # If provided, must be a list of 1 or 2 timestamps (strings like "00:01:30"), or None.
        if self.cut is not None:
            if not isinstance(self.cut, list):
                raise ValueError("Option 'cut' must be a list of timestamps or None.")
            if len(self.cut) < 1 or len(self.cut) > 2:
                raise ValueError("Option 'cut' must have 1 or 2 timestamps when provided.")
            # Optionally validate each timestamp (regex for HH:MM:SS or similar).
            timestamp_pattern = r"^\d{2}:\d{2}:\d{2}$"
            for timestamp in self.cut:
                if not re.match(timestamp_pattern, timestamp):
                    raise ValueError(
                        f"Invalid timestamp format '{timestamp}'. "
                        "Must be in HH:MM:SS format (e.g., '00:01:30')."
                    )
        # 8. length (float or None)
        # If it's not None, it should be > 0.
        if self.length is not None:
            if not isinstance(self.length, (int, float)) or self.length <= 0:
                raise ValueError(
                    f"Option 'length' must be a positive numeric value, got '{self.length}'."
                )

        # If user provided two timestamps (start & end) in cut,
        # they must not also provide length.
        if self.cut is not None and len(self.cut) > 1 and self.length is not None:
            raise ValueError(
                "Error: When providing two timestamps in '--cut', "
                "you cannot also provide '--length'. "
                "The '--length' is only for a single start-timestamp scenario."
            )

        # 9. rm_audio (boolean)
        if not isinstance(self.rm_audio, bool):
            raise ValueError("Option 'rm_audio' must be True or False.")

        # 10. keep_gps (boolean)
        if not isinstance(self.keep_gps, bool):
            raise ValueError("Option 'keep_gps' must be True or False.")

        # 11. extract_gpx (boolean)
        if not isinstance(self.extract_gpx, bool):
            raise ValueError("Option 'extract_gpx' must be True or False.")

        # Cross-check: GPS options with cut and length
        if self.extract_gpx and not self.keep_gps:
            # Ensure that extraction does not rely on preserving GPS in video
            pass  # No conflict, can extract and not preserve


        logging.info("Options validated successfully.")



    @staticmethod
    def check_command_availability(command):
        """
        Check if a command (e.g., ffmpeg, ffprobe) is installed on the system.

        Parameters
        ----------
        command : str
            The command to check for.

        Returns
        -------
        None

        Raises
        ------
        SystemExit
            If the command is not found on the system.
        """
        try:
            subprocess.run([command, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            sys.exit(f"Error: {command} is not installed. "
                     f"Please install it using your package manager "
                     f"(e.g., `brew install ffmpeg` or `yay -S perl-image-exiftool`).")

    @staticmethod
    def get_video_info(input_file):
        """
        Get video metadata using ffprobe.

        Parameters
        ----------
        input_file : str
            Path to the video file.

        Returns
        -------
        dict
            Dictionary containing the video metadata.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            input_file
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logging.error(f"Command failed: {cmd}\nError: {result.stderr}")
            return

        try:
            info = json.loads(result.stdout)
            if 'streams' not in info:
                sys.exit(
                    f"Error: 'streams' key not found in ffprobe output for file {input_file}. "
                    "The file might be corrupted or not supported."
                )
            return info
        except json.JSONDecodeError:
            sys.exit("Error: Invalid JSON output from ffprobe. "
                     "Please ensure the input file is a valid video.")


    def should_process_video(self, input_file):
        """
        Determine which parts of the video should be processed based on given criteria.

        Parameters
        ----------
        input_file : str
            Path to the video file.

        Returns
        -------
        dict
            Dictionary containing flags for each conversion option.
        """
        process_flags = {
            'codec': True,
            'resolution': True,
            'bitrate': True,
            'valid_filetype': True,
            'keep_gps': self.keep_gps,
            'extract_gpx': self.extract_gpx
        }

        # Check if the file is one of the valid file extensions (global)
        if input_file.split('.')[-1].lower() not in VALID_VIDEO_EXTENSIONS:
            process_flags['valid_filetype'] = False

        try:
            info = self.get_video_info(input_file)
            video_stream = [stream for stream in info['streams'] if stream['codec_type'] == 'video'][0]

            # Check codec
            if self.codec == "h265" and video_stream['codec_name'].lower() in ['hevc','h265']:
                process_flags['codec'] = False

            # Check resolution
            current_height = int(video_stream['height'])
            target_height = int(self.resolution)
            if current_height == target_height:
                process_flags['resolution'] = False

            # Check bitrate
            if self.bitrate != 'unchanged':
                if self.bitrate in BITRATES:
                    target_bitrate_str = BITRATES[self.bitrate].get(self.resolution, None)
                    if target_bitrate_str:
                        target_bitrate = int(target_bitrate_str.replace('k', '000'))
                    else:
                        logging.warning(f"Undefined bitrate for resolution '{self.resolution}' and bitrate level '{self.bitrate}'.")
                        target_bitrate = 0
                else:
                    # Assume it's a numeric value like '4000k' or '4M'
                    match = re.match(r'^(\d+)([kKmM]?)$', self.bitrate)
                    if match:
                        number = int(match.group(1))
                        unit = match.group(2).lower()
                        if unit == 'k':
                            target_bitrate = number * 1000
                        elif unit == 'm':
                            target_bitrate = number * 1000000
                        else:
                            target_bitrate = number
                    else:
                        logging.warning(f"Invalid bitrate format '{self.bitrate}'. Skipping bitrate adjustment.")
                        process_flags['bitrate'] = False  # unchanged bitrate

                # Get current bitrate
                current_bitrate_str = video_stream.get('bit_rate', None)
                if current_bitrate_str and current_bitrate_str.isdigit():
                    current_bitrate = int(current_bitrate_str)
                else:
                    current_bitrate = 0  # Assume 0 if not available

                if current_bitrate <= target_bitrate:
                    process_flags['bitrate'] = False
            else:
                process_flags['bitrate'] = False  # unchanged bitrate

            # Handle GPS flags
            if self.keep_gps:
                # Metadata handling is done in ffmpeg command
                pass
            else:
                pass  # Metadata stripping is done in ffmpeg command

        except Exception as e:
            logging.info(f"Warning: Could not process file '{input_file}'. Reason: {e}")
            # If an error occurs, skip all processes
            for key in process_flags:
                process_flags[key] = False

        return process_flags


    def prepare_strings_outfiles(self, in_file, flags):
        """
        Prepare input and output filenames for conversion.

        Parameters
        ----------
        in_file : str
            Path to the video file.
        options : dict
            Dictionary containing conversion criteria.
        flags   : dict
            see flags creator function

        Returns
        -------
        output_file
            path to the output file

        """
        codec_str = f"-{self.codec}" if self.codec and flags['codec'] else ""
        quality_str = f"-q{self.quality}" if flags['codec'] else ""
        resolution_str = f"-{self.resolution}p" if flags['resolution'] else ""
        rm_audio_str = "-noaudio" if self.rm_audio else ""
        bitrate_str = ""
        if flags['bitrate']:
            if self.bitrate in BITRATES:
                bitrate_in_mbps = int(BITRATES[self.bitrate][self.resolution].replace('k', '')) / 1000
                bitrate_str = f"-{bitrate_in_mbps}Mbps"
            else:
                bitrate_str = f"-{self.bitrate}"  # Use the user-provided numeric value
        crop_str = "-cropped" if self.crop else ""
        # GPS flags
        gps_str = "-keepGPS" if self.keep_gps else "-noGPS"
        extract_gpx_str = "-extractGPX" if self.extract_gpx else ""

        components = [codec_str, quality_str, resolution_str, bitrate_str, rm_audio_str, crop_str]
        # Filter out empty strings and join with underscores for clarity
        components = [comp for comp in components if comp]
        suffix = "_".join(components) if components else "converted"
        base_name = os.path.splitext(in_file)[0]
        output_file = f"{base_name}_{suffix}.mp4"

        logging.info(f"FILENAMES: \n Input: {in_file} \n Output: {output_file}")    

        return output_file


    def run_ffmpeg_conversion(self, input_file, output_file, flags):
        """
        Convert the video using ffmpeg based on the provided options and flags.

        Parameters
        ----------
        input_file : str
            Full path to the input video file.
        output_file : str
            Full path to the desired output file.
        flags : dict
            Dictionary containing flags for each conversion option. A flag is True if the corresponding
            option should be processed, and False otherwise.

        Returns
        -------
        None
        """

        temp_cut_file = None


        # Handle the cut operation
        if self.cut:
            # We'll create a temp file for the cut portion
            base_name = os.path.splitext(input_file)[0]
            temp_cut_file = f"{base_name}_temp_cut.mp4"
            cmd_cut = ["ffmpeg", "-i", input_file]


            # If the user provided exactly one timestamp
            # (e.g., --cut 00:01:30), optionally combine it with --length
            if len(self.cut) == 1:
                start_time = self.cut[0]

                if self.length:
                    # Interpret self.length as fractional minutes; e.g. 1.5 = 1 minute + 30 seconds
                    minutes, fraction = divmod(self.length, 1)
                    duration_seconds = int(minutes * 60 + fraction * 60)
                    cmd_cut.extend(["-ss", start_time, "-t", str(duration_seconds)])
                else:
                    cmd_cut.extend(["-ss", start_time])

            # If the user provided two timestamps
            # (e.g., --cut 00:01:30 00:02:00), we do -ss start -to end
            else:
                start_time, end_time = self.cut
                cmd_cut.extend(["-ss", start_time, "-to", end_time])

            # copy codec 
            cmd_cut.extend(["-c", "copy", temp_cut_file])

            logging.info(f"Cutting video {input_file} with command: {' '.join(cmd_cut)}")
            try:
                subprocess.run(cmd_cut, check=True)
                input_file = temp_cut_file
            except subprocess.CalledProcessError as e:
                logging.error(f"Cutting failed: {e}")
                return

        # If no --cut but a --length was provided, cut from the beginning
        elif self.length:
            base_name = os.path.splitext(input_file)[0]
            temp_cut_file = f"{base_name}_temp_cut.mp4"
            cmd_cut = ["ffmpeg", "-i", input_file]

            # Default start at 0 if user didn’t specify a start timestamp
            start_time = "00:00:00"  
            minutes, fraction = divmod(self.length, 1)
            duration_seconds = int(minutes * 60 + fraction * 60)

            cmd_cut.extend(["-ss", start_time, "-t", str(duration_seconds)])
            cmd_cut.append(temp_cut_file)

            # copy codec
            cmd_cut.extend(["-c", "copy", temp_cut_file])

            logging.info(f"Cutting video {input_file} with command: {' '.join(cmd_cut)}")
            try:
                subprocess.run(cmd_cut, check=True)
                input_file = temp_cut_file
            except subprocess.CalledProcessError as e:
                logging.error(f"Cutting failed: {e}")
                return


        # Main conversion
        cmd = ["ffmpeg", "-i", input_file]

        # Metadata handling based on GPS flags
        if self.keep_gps:
            cmd.extend(["-map_metadata", "0"])  # Preserve all metadata
        else:
            cmd.extend(["-map_metadata", "-1"])  # Strip all metadata

        # Only add the codec option if it's being changed
        # always convert h264 videos to HEVC/h265
        if flags['codec']:

            hw = self.hw_acceleration

            ## macOS hardware acceleration
            if hw == "videotoolbox":
                if self.codec == "h265":
                    cmd.extend([
                        '-c:v', 'hevc_videotoolbox',
                        '-global_quality', str(self.quality)
                    ])
                else:
                    cmd.extend([
                        '-c:v', 'h264_videotoolbox',
                        '-global_quality', str(self.quality)
                    ])

            ## NVIDIA NVENC
            elif hw == "cuda":
            
                nvenc_codec = "hevc_nvenc" if self.codec == "h265" else "h264_nvenc"
                cmd.extend([
                    '-c:v', nvenc_codec,
                    '-rc', 'vbr',
                    '-cq', str(self.quality)
                ])


            ## Intel QuickSync
            elif hw == "qsv":
                qsv_codec = "hevc_qsv" if self.codec == "h265" else "h264_qsv"
                cmd.extend([
                    '-c:v', qsv_codec,
                    '-global_quality', str(self.quality)
                ])


            ## VAAPI on Linux
            elif hw == "vaapi":
                vaapi_codec = "hevc_vaapi" if self.codec == "h265" else "h264_vaapi"
                cmd.extend([
                    '-hwaccel', 'vaapi',
                    '-hwaccel_output_format', 'vaapi',
                    '-c:v', vaapi_codec,
                    '-crf', str(self.quality)
                ])


            else:
                # Fallback to software if none of the above match
                cmd.extend([
                    '-c:v', 'libx265' if self.codec == 'h265' else 'libx264',
                    '-crf', str(self.quality)
                ])

        # removing audio if requested
        if self.rm_audio: 
            ## removes the audio track entirely 
            cmd.append('-an')

        # Only add the bitrate option if it's not 'unchanged' and it's being changed
        if self.bitrate != 'unchanged' and flags['bitrate']:
            bitrate = BITRATES[self.bitrate][self.resolution]
            cmd.extend(["-b:v", bitrate])

        # Crop and/or scale
        vf_options = []
        if self.crop:
            vf_options.append("crop=ih*9/16:ih")
        if flags['resolution']:
            vf_options.append(f"scale=-1:{self.resolution}")
        if vf_options:
            cmd.extend(["-vf", ",".join(vf_options)])


        # Handle GPS extraction before running ffmpeg
        if self.extract_gpx:
            # Extract GPS data from the input file (post-cut if any)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            gpx_file = os.path.join(os.path.dirname(output_file), f"{base_name}.gpx")
            
            # Write the GPX template to a temporary file
            gpx_template_path = write_gpx_template()
            
            extract_gps_cmd = [
                "exiftool",
                "-ee",
                "-p", gpx_template_path,
                input_file
            ]
            logging.info(f"Extracting GPS data to '{gpx_file}'...")            
            try:
                with open(gpx_file, "w") as gpx_out:
                    subprocess.run(extract_gps_cmd, stdout=gpx_out, stderr=subprocess.PIPE, check=True, text=True)
                if os.path.getsize(gpx_file) == 0:
                    logging.warning("No GPS data found; removing empty GPX file.")
                    os.remove(gpx_file)
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to extract GPS data: {e.stderr}")
                gpx_file = None


        ## overwriting (possibly input arg option)
        cmd.extend(["-y", output_file]) ## overwrite if it exists here

        logging.info(f"Running ffmpeg on {input_file} with:")
        logging.info(f" >> {' '.join(cmd)}")
        
        # Execution with error handling
        try:
            subprocess.run(cmd,  check=True, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"ffmpeg failed with error: {e.stderr}")
            return        

        # Clean up temporary cut file
        if temp_cut_file and os.path.isfile(temp_cut_file):
            logging.info('Removing cut temp-file...')
            os.remove(temp_cut_file)


    def run_file_conversion(self, in_file):
        """
        Process a single video file based on the provided options. It checks whether
        these conversions actually have to be made to the file or the file actually has
        those properties already, accelerating large processes.

        Single-entry method to:
          1) Determine if the file is valid/needs processing
          2) Prepare strings for output
          3) Run the actual ffmpeg conversion
          4) (Optionally) remove the original if needed

        Parameters
        ----------
        in_file : str
            Path to the video file to be processed.

        Returns
        -------
        None
        """
        logging.info(f'- Input file: {in_file}')
        #logging.info(f'{os.path.basename(in_file)}')

        if in_file.split('.')[-1].lower() in VALID_VIDEO_EXTENSIONS:
            flags = self.should_process_video(in_file)
            if flags['valid_filetype'] and any([
                    flags['codec'], flags['resolution'], flags['bitrate'], flags['keep_gps'], flags['extract_gpx']]):                
                output_file = self.prepare_strings_outfiles(in_file, flags)
                if not os.path.isfile(output_file):
                    self.run_ffmpeg_conversion(in_file, output_file, flags)
                if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
                    if self.remove:
                        logging.info(f"Removing original file: {in_file}")
                        os.remove(in_file)
                    else:
                        logging.info(f"Retaining original file: {in_file}")
            else:
                logging.info(f"Skipping {in_file} (no processing needed).")
        else:
            logging.info(f"Skipping {in_file} (unsupported file extension).")

def main():
    """
    Command-line interface for the video conversion tool.

    This function parses command-line arguments and processes video files or directories
    based on the provided options. It supports various operations like changing resolution,
    bitrate, codec, cropping, and cutting.

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(description="Video Conversion Tool")
    parser.add_argument("input_path", help="Input file or directory path.")
    parser.add_argument("--resolution", choices=["360", "480", "720", "1080", "2160"], default="1080", help="Target video resolution.")
    parser.add_argument("--bitrate", default="unchanged", help="Target bitrate level. Can be 'low', 'mid', 'high', or any numeric value (e.g., '4000k').")
    parser.add_argument("--codec", choices=["h265", "h264"], default="h265", help="Video codec to use, default is HEVC ;) .")
    parser.add_argument("--quality",type=int, default=0, choices=range(0, 51), help="Quality (integer-) value for encoding. Lower is better, but will produce bigger files. Range: [0-51]. Default lossless = 0.")
    parser.add_argument("--crop", action="store_true", help="Crop video for portrait mode on phones.")
    parser.add_argument("--remove", action="store_true", help="Remove the input file after conversion.")
    parser.add_argument("--cut", nargs='+', help="Cut video from specific timestamps.")
    parser.add_argument("--length", type=float, help="Length of the video in minutes. Fractions represent additional seconds (e.g., 1.5 = 1 minute 30 seconds).")
    parser.add_argument("--rm_audio", action='store_true', help="Whether to remove the audio track of the video file.")
    parser.add_argument("--keep_gps", action='store_true', help="Preserve GPS metadata in the output video.")
    parser.add_argument("--extract_gpx", action='store_true', help="Extract GPS metadata into a GPX file.")
    parser.add_argument("--recursive", action="store_true", help="Process files in subdirectories.")

    args = parser.parse_args()
    
    converter = VideoConverter(
        resolution = args.resolution,
        bitrate = args.bitrate,
        codec=args.codec,
        quality=args.quality,
        crop=args.crop,
        remove=args.remove,
        cut=args.cut,
        length=args.length,
        rm_audio=args.rm_audio,
        keep_gps=args.keep_gps,
        extract_gpx=args.extract_gpx
        )

    ## if INPUT FILE:
    if os.path.isfile(args.input_path):
        converter.run_file_conversion(args.input_path)
       
    ## if INPUT FOLDER:
    elif os.path.isdir(args.input_path):
        # If the remove option is set, ask for confirmation
        if args.remove:
            confirm = input(f"After conversion, you will be removing all files in the directory '{args.input_path}' and its subdirectories. Are you sure? (then type yes or y): ")
            if confirm.lower() not in ['yes','y']:
                sys.exit("Operation aborted.")

        if args.recursive:
            # Process files in all subdirectories
            for dirpath, dirnames, filenames in os.walk(args.input_path):
                for file in filenames:
                    full_path = os.path.join(dirpath, file)
                    if os.path.isfile(full_path):  # Skip directories
                        converter.run_file_conversion(full_path)
        else:
            # Process only files in the current directory
            for file in os.listdir(args.input_path):
                full_path = os.path.join(args.input_path, file)
                if os.path.isfile(full_path):  # Skip directories
                    converter.run_file_conversion(full_path)

    else:
        logging.info("Invalid path. Please provide a valid video file or directory.")

if __name__ == "__main__":
    main()
