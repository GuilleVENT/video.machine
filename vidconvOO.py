import os
import argparse
import subprocess
import sys
import re
import json
import logging
import platform
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logging.info("To Do: improve logging")

# ------------------------------------------------------------
# Global/Module-level Constants & Helper Dictionaries
# ------------------------------------------------------------

VALID_VIDEO_EXTENSIONS = ['mp4', 'mkv', 'avi', 'mov']

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

def extract_gps_data(input_video_path, output_filename):
    """
    Extract GPS data from a video file using ExifTool and save it as a .gpx file
    with the same base name as the provided output file.

    Parameters
    ----------
    input_video_path : str
        Full path to the input video file (original video file).
    output_filename : str
        Name of the output file (no directory), used as the base for the .gpx file.

    Returns
    -------
    str or None
        Full path to the .gpx file if created successfully, else None.
    """

    # Create the directory of the input video
    output_dir = os.path.dirname(input_video_path)

    # Use the provided output_filename (change extension to .gpx)
    gpx_out = os.path.join(output_dir, os.path.splitext(output_filename)[0] + ".gpx")

    # Construct the exiftool command
    cmd = [
        "exiftool",
        "-ee",  # Extract embedded data
        "-p", "gpx.fmt",  # Format as GPX
        input_video_path
    ]

    logging.info(f"Extracting GPS data from '{input_video_path}' to '{gpx_out}'...")

    # Run exiftool and write to .gpx file
    with open(gpx_out, "wb") as outfile:
        try:
            subprocess.run(cmd, stdout=outfile, stderr=subprocess.PIPE, check=True)
        except FileNotFoundError:
            logging.error("Exiftool not found or not installed.")
            return None
        except subprocess.CalledProcessError as e:
            logging.error(f"Exiftool extraction failed: {e}")
            return None

    # Check if the .gpx file was created and has content
    if os.path.isfile(gpx_out) and os.path.getsize(gpx_out) > 0:
        logging.info(f"GPS data successfully saved to: {gpx_out}")
        return gpx_out
    else:
        logging.info("No GPS data found, or exiftool could not extract any location info.")
        if os.path.exists(gpx_out):
            os.remove(gpx_out)  # Clean up empty file
        return None




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
            gps: bool = False
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
            - 'length': Float indicating the length of the video in seconds or minutes.
            - 'rm_audio': Whether to remove audio track of the video.
            - 'gps': Whether to extract the gps data into a separate gpx-file.
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
        self.gps = gps

        # Immediately validate all options.
        self.validate_options()

        for requiered_dependency in ['ffmpeg', 'ffprobe']:
            self.check_command_availability(requiered_dependency)
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
        if self.resolution not in valid_resolutions:
            raise ValueError(
                f"Invalid resolution '{self.resolution}'. "
                f"Must be one of {valid_resolutions}."
            )

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
            # Optionally validate each timestamp (regex for HH:MM:SS or something similar).

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

        # 10. gps (boolean)
        if not isinstance(self.gps, bool):
            raise ValueError("Option 'gps' must be True or False.")

        # Cross-check: if user gave two timestamps in cut, length must not also be used.
        if self.cut is not None and len(self.cut) > 1 and self.length is not None:
            raise ValueError(
                "Cannot supply two timestamps in 'cut' and also supply 'length'. "
                "The 'length' is only for a single start-timestamp scenario."
            )

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
                     f"(e.g., `brew install ffmpeg`).")

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
            'valid_filetype': True
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
                current_bitrate = int(video_stream.get('bit_rate', '0'))
                target_bitrate = int(BITRATES[self.bitrate][self.resolution].replace('k', '000'))
                if current_bitrate <= target_bitrate:
                    process_flags['bitrate'] = False
            else:
                process_flags['bitrate'] = False

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
        tuple
            Tuple containing paths to the input and output files.

        """

        codec_str = "-h265" if self.codec == "h265" and flags['codec'] else ""
       
        quality_str = f"-encode-quality-{self.quality}" if flags['codec'] else ""    

        resolution_str = f"-{self.resolution}" if flags['resolution'] else ""

        rm_audio_str = f"-no_audio" if self.rm_audio else ""
       
        # Only add the bitrate to the filename if it's being changed
        if flags['bitrate']:
            bitrate_in_mbps = int(BITRATES[self.bitrate][self.resolution].replace('k', '')) / 1000
            bitrate_str = f"-{bitrate_in_mbps}Mbps"
        else:
            bitrate_str = ""

        crop_str = "-vertical" if self.crop else ""

        base_name = os.path.splitext(in_file)[0]
        output_file = f"{base_name}{codec_str}{quality_str}{resolution_str}{bitrate_str}{rm_audio_str}{crop_str}.mp4"

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
            temp_cut_file = input_file.split('.')[0] + "_temp_cut.mp4"
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

            cmd_cut.append(temp_cut_file)
            logging.info(f"Cutting video {input_file} with command: {' '.join(cmd_cut)}")
            subprocess.run(cmd_cut)
            input_file = temp_cut_file

        # If no --cut but a --length was provided, cut from the beginning
        elif self.length:
            temp_cut_file = input_file.split('.')[0] + "_temp_cut.mp4"
            cmd_cut = ["ffmpeg", "-i", input_file]

            # Default start at 0 if user didn’t specify a start timestamp
            start_time = "00:00:00"  
            minutes, fraction = divmod(self.length, 1)
            duration_seconds = int(minutes * 60 + fraction * 60)

            cmd_cut.extend(["-ss", start_time, "-t", str(duration_seconds)])
            cmd_cut.append(temp_cut_file)

            logging.info(f"Cutting video {input_file} with command: {' '.join(cmd_cut)}")
            subprocess.run(cmd_cut)
            input_file = temp_cut_file


        # Main conversion
        cmd = ["ffmpeg", "-i", input_file]

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

        # telling ffmpeg to copy all metadata (gps and audio if deleted above)
        if self.gps and not self.cut and not self.rm_audio:
            cmd.extend(["-map_metadata", "0"])

        '''
        DEVELOPING:
        # extract gps data if requested
        if self.gps: 
            output_filename = output_file.split('/')[-1]
            gpx_file = extract_gps_data(input_file, output_filename )
        '''


        # Only add the bitrate option if it's not 'unchanged' and it's being changed
        if self.bitrate != 'unchanged' and flags['bitrate']:
            bitrate = BITRATES[self.bitrate][self.resolution]
            cmd.extend(["-b:v", bitrate])

        # Crop and/or scale
        vf_options = []
        if flags['resolution']:
            vf_options.append(f"scale=-1:{self.resolution.split('p')[0]}")
        if self.crop:
            vf_options.append("crop=ih*9/16:ih")
        if vf_options:
            cmd.extend(["-vf", ",".join(vf_options)])

        logging.info(f"Running these changes on {input_file} \n {' '.join(cmd)}")
        cmd.extend(["-y", output_file]) ## overwrite if it exists here
        
        logging.info(' '.join(cmd))
        ## RUN! 
        subprocess.run(cmd)
        

        # Clean up
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
            if flags['valid_filetype'] and any(flags.values()):
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
    parser.add_argument("--length", type=float, help="Length of the video in seconds. If the value is decimal, it's interpreted as minutes.")
    parser.add_argument("--rm_audio", action='store_true', help="Whether to remove the audio track of the video file.")
    parser.add_argument("--gps", action='store_true', help="Whether to maintain GPS-metadata in the video. Currently not supported with --cut or --lenght")
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
        gps=args.gps)

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

