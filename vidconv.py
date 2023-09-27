import os
import argparse
import subprocess
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logging.info("To Do: logging")

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
        '720': '6000k',
        '1080': '12000k',
        '2160': '50000k'
    }
}

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
    
    try:
        info = json.loads(result.stdout)
        if 'streams' not in info:
            sys.exit(f"Error: 'streams' key not found in ffprobe output for file {input_file}. The file might be corrupted or not supported.")
        return info
    except json.JSONDecodeError:
        sys.exit("Error: Invalid JSON output from ffprobe. Please ensure the input file is a valid video.")

def should_process_video(input_file, options):
    """
    Determine which parts of the video should be processed based on given criteria.

    Parameters
    ----------
    input_file : str
        Path to the video file.
    options : dict
        Dictionary containing conversion criteria.

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

    # Check if the file is an mp4 or not
    if input_file.split('.')[-1].lower() not in VALID_VIDEO_EXTENSIONS:# or input_file.startswith('.'):
        process_flags['valid_filetype'] = False

    info = get_video_info(input_file)
    video_stream = [stream for stream in info['streams'] if stream['codec_type'] == 'video'][0]

    # Check codec
    if options['codec'] == "h265" and video_stream['codec_name'] == 'hevc':
        process_flags['codec'] = False

    # Check resolution
    current_height = int(video_stream['height'])
    target_height = int(options['resolution'].split('p')[0])
    if current_height == target_height:
        process_flags['resolution'] = False

    # Check bitrate
    if options['bitrate'] != 'unchanged':
        current_bitrate = int(video_stream.get('bit_rate', '0'))
        target_bitrate = int(BITRATES[options['bitrate']][options['resolution']].replace('k', '000'))
        if current_bitrate <= target_bitrate:
            process_flags['bitrate'] = False
    else:
        process_flags['bitrate'] = False

    return process_flags

def prepare_strings_outfiles(in_file, options, flags):
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

    codec_str = "-h265" if options['codec'] == "h265" and flags['codec'] else ""
    
    # Only add the resolution to the filename if it's being changed
    resolution_str = f"-{options['resolution']}" if flags['resolution'] else ""
    
    # Only add the bitrate to the filename if it's being changed
    if flags['bitrate']:
        bitrate_in_mbps = int(BITRATES[options['bitrate']][options['resolution']].replace('k', '')) / 1000
        bitrate_str = f"-{bitrate_in_mbps}Mbps" 
    else:
        bitrate_str = ""

    crop_str = "-phone" if options['crop'] else ""

    base_name = os.path.splitext(in_file)[0]
    output_file = f"{base_name}{codec_str}{resolution_str}{bitrate_str}{crop_str}.mp4"
    input_file = in_file

    print('FILENAMES','\n',input_file,'\n', output_file)
    
    return input_file, output_file

def run_ffmpeg_conversion(input_file, output_file, options, flags):
    """
    Convert the video using ffmpeg based on the provided options and flags.

    Parameters
    ----------
    input_file : str
        Full path to the input video file.
    output_file : str
        Full path to the desired output file.
    options : dict
        Dictionary containing the desired codec, resolution, bitrate, crop setting, cut, and length.
    flags : dict
        Dictionary containing flags for each conversion option. A flag is True if the corresponding
        option should be processed, and False otherwise.

    Returns
    -------
    None
    """
    temp_cut_file = None

    # Handle the cut operation
    if options.get('cut'):
        temp_cut_file = input_file + ".temp_cut.mp4"
        cmd_cut = ["ffmpeg", "-i", input_file]

        if len(options['cut']) == 1:
            start_time = options['cut'][0]

            if options.get('length'):
                minutes, seconds = divmod(options['length'], 1)  # Split whole number from decimal
                duration = minutes * 60 + seconds * 60  # Convert minutes to seconds and add additional seconds
                cmd_cut.extend(["-ss", start_time, "-t", str(duration)])
            else:
                cmd_cut.extend(["-ss", start_time])
        else:
            start_time, end_time = options['cut']
            cmd_cut.extend(["-ss", start_time, "-to", end_time])

        cmd_cut.append(temp_cut_file)
        logging.info(f"Cutting video {input_file}...")
        subprocess.run(cmd_cut)
        input_file = temp_cut_file

    elif options.get('lengths'):
        temp_cut_file = input_file + ".temp_cut.mp4"
        minutes, seconds = divmod(options['length'], 1)  # Split whole number from decimal
        duration = minutes * 60 + seconds * 60  # Convert minutes to seconds and add additional seconds
        cmd_cut.extend(["-ss", start_time, "-t", str(duration)])

        cmd_cut.append(temp_cut_file)
        logging.info(f"Cutting video {input_file}...")
        subprocess.run(cmd_cut)
        input_file = temp_cut_file

    # Main conversion
    cmd = ["ffmpeg", "-i", input_file]

    # Only add the codec option if it's being changed
    if flags['codec']:
        codec_options = "libx265" if options['codec'] == "h265" else "libx264"
        cmd.extend(["-c:v", codec_options])

    # Only add the bitrate option if it's not 'unchanged' and it's being changed
    if options['bitrate'] != 'unchanged' and flags['bitrate']:
        bitrate = BITRATES[options['bitrate']][options['resolution']]
        cmd.extend(["-b:v", bitrate])

    # Only add the crop and scale options if they're being changed
    vf_options = []
    if flags['resolution']:
        vf_options.append(f"scale=-1:{options['resolution'].split('p')[0]}")
    if options['crop']:
        vf_options.append("crop=ih*9/16:ih")
    if vf_options:
        cmd.extend(["-vf", ",".join(vf_options)])

    logging.info(f"Running these changes on {input_file} \n {' '.join(cmd)}")
    cmd.extend(["-y", output_file])
    subprocess.run(cmd)
    logging.info(' '.join(cmd))


    # Clean up
    if temp_cut_file:
        logging.info('Removing cut temp-file...')
        os.remove(temp_cut_file)


def run_file(in_file, options):
    """
    Process a single video file based on the provided options. It checks whether 
    these conversions actually have to be made to the file or the file actually has 
    those properties already, accelerating large processes. 

    Parameters
    ----------
    in_file : str
        Path to the video file to be processed.
    options : dict
        Dictionary containing conversion criteria. Supported keys are:
        - 'resolution': Target video resolution (e.g., "1080").
        - 'bitrate': Target bitrate level (e.g., "low", "mid", "high", "unchanged").
        - 'codec': Video codec to use (e.g., "h265", "h264").
        - 'crop': Boolean indicating if video should be cropped for portrait mode on phones.
        - 'remove': Boolean indicating if the input file should be removed after conversion.
        - 'cut': List containing start (and optionally end) timestamps for cutting the video.
        - 'length': Float indicating the length of the video in seconds or minutes.

    Returns
    -------
    None
    """
    logging.info(f'- Input file: {in_file}')
    logging.info(f'{os.path.basename(in_file)}')

    if in_file.split('.')[-1].lower() in VALID_VIDEO_EXTENSIONS : ## not .DS_Store
        flags = should_process_video(in_file, options)
        if flags['valid_filetype']:
            if any(flags.values()):  # Check if any of the flags are set to True
                in_file, output_file = prepare_strings_outfiles(in_file, options, flags)
            
                # Modify run_ffmpeg_conversion to take into account the flags
                run_ffmpeg_conversion(in_file, output_file, options, flags)

                if os.path.isfile(output_file) and options.remove:  # Check if output file exists before removing input
                    os.remove(in_file)
        else:
            print(f"Skipping {in_file} as it looks like criteria is already be satisfied.")
    else: return


def check_command_availability(command):
    try:
        subprocess.run([command, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        sys.exit(f"Error: {command} is not installed. Please install it to proceed.")



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
    parser.add_argument("--crop", action="store_true", help="Crop video for portrait mode on phones.")
    parser.add_argument("--remove", action="store_true", help="Remove the input file after conversion.")
    parser.add_argument("--cut", nargs='+', choices=range(1, 3), help="Cut video from specific timestamps. Either provide start and end timestamps or just the start timestamp.")
    parser.add_argument("--length", type=float, help="Length of the video in seconds. If the value is decimal, it's interpreted as minutes.")


    args = parser.parse_args()
    options = {
        'resolution': args.resolution,
        'bitrate': args.bitrate,
        'codec': args.codec,
        'crop': args.crop,
        'remove': args.remove,
        'cut': args.cut,
        'length': args.length
    }
    
    if args.cut and len(args.cut) > 1 and args.length:
        raise ValueError("Error: When providing a list for --cut, you cannot also provide --length. Length is to be used with a start-timestamp in --cut without end-stimestamp. ")
    
    check_command_availability("ffmpeg")
    check_command_availability("ffprobe")

    if os.path.isfile(args.input_path):
        run_file(args.input_path, options)
        

    elif os.path.isdir(args.input_path):
        # If the remove option is set, ask for confirmation
        if args.remove:
            confirm = input(f"You are about to remove all files in the directory '{args.input_path}' and its subdirectories. Are you sure? (then type yes or y): ")
            if confirm.lower() not in ['yes','y']:
                sys.exit("Operation aborted.")
                
        for dirpath, dirnames, filenames in os.walk(args.input_path):
            for file in filenames:
                full_path = os.path.join(dirpath, file)
                run_file(full_path, options)
    else:
        print("Invalid path. Please provide a valid video file or directory.")

if __name__ == "__main__":
    main()
