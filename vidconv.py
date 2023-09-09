import os
import argparse
import subprocess
import json

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
    return json.loads(result.stdout)

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
    if input_file.split('.')[-1].lower() != 'mp4':
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
    current_bitrate = int(video_stream.get('bit_rate', '0'))
    target_bitrate = int(BITRATES[options['bitrate']][options['resolution']].replace('k', '000'))
    if current_bitrate <= target_bitrate:
        process_flags['bitrate'] = False

    return process_flags

def prepare_strings_outfiles(in_file, options):
    """
    Prepare input and output filenames for conversion.

    Parameters
    ----------
    in_file : str
        Path to the video file.
    options : dict
        Dictionary containing conversion criteria.

    Returns
    -------
    tuple
        Tuple containing paths to the input and output files.
    """
    codec_str = "-h265" if options['codec'] == "h265" else ""
    resolution_str = f"-{options['resolution']}"
    bitrate_str = f"-{options['bitrate']}"
    crop_str = "-phone" if options['crop'] else ""

    base_name = os.path.splitext(in_file)[0]
    output_file = os.path.join(os.getcwd(), f"{base_name}{codec_str}{resolution_str}{bitrate_str}{crop_str}.mp4")
    input_file = os.path.join(os.getcwd(), in_file)
    
    print('FILENAMES')
    print(input_file, '\n', output_file)
    
    return input_file, output_file

def run_ffmpeg_conversion(input_file, output_file, options):
    """
    Convert the video using ffmpeg based on the provided options.

    Parameters
    ----------
    input_file : str
        Full path to the input video file.
    output_file : str
        Full path to the desired output file.
    options : dict
        Dictionary containing the desired codec, resolution, bitrate, and crop setting.
    """
    codec_options = "-c:v libx265" if options['codec'] == "h265" else ""
    cmd = ["ffmpeg", "-i", input_file, codec_options]

    # Only add the bitrate option if it's not 'unchanged'
    bitrate = BITRATES[options['bitrate']][options['resolution']]
    if options['bitrate'] != 'unchanged':
        cmd.extend(["-b:v", bitrate])

    crop_option = "crop=ih*9/16:ih" if options['crop'] else ""
    cmd.extend([
        "-vf", f"{crop_option},scale=-1:{options['resolution'].split('p')[0]}",
        "-y",
        output_file
    ])

    subprocess.run(cmd)

def run_file(in_file, options):
    if os.path.isfile(in_file):
        flags = should_process_video(in_file, options)
        if any(flags.values()):  # Check if any of the flags are set to True
            in_file, output_file = prepare_strings_outfiles(in_file, options)
        
            # Modify run_ffmpeg_conversion to take into account the flags
            run_ffmpeg_conversion(in_file, output_file, options, flags)

            if options.remove:
                os.remove(in_file)
        else:
            print(f"Skipping {in_file} as it looks like criteria might already be satisfied.")



def main():
    parser = argparse.ArgumentParser(description="Video Conversion Tool")
    parser.add_argument("input_path", help="Input file or directory path.")
    parser.add_argument("--resolution", choices=["360", "480", "720", "1080", "2160"], default="1080", help="Target video resolution.")
    parser.add_argument("--bitrate", choices=["low", "mid", "high", "unchanged"], default="unchanged", help="Target bitrate level.")
    parser.add_argument("--codec", choices=["h265", "h264"], default="h265", help="Video codec to use.")
    parser.add_argument("--crop", action="store_true", help="Crop video for portrait mode on phones.")
    parser.add_argument("--remove", action="store_true", help="Remove the input file after conversion.")

    args = parser.parse_args()
    options = {
        'resolution': args.resolution,
        'bitrate': args.bitrate,
        'codec': args.codec,
        'crop': args.crop,
        'remove': args.remove
    }

    if os.path.isfile(args.input_path):
        run_file(args.input_path, options)
        

    elif os.path.isdir(args.input_path):
        files = [f for f in os.listdir(args.input_path) if os.path.isfile(os.path.join(args.input_path, f))]
        
        for file in files:
            full_path = os.path.join(args.input_path, file)
            run_file(full_path, options)  
    else:
        print("Invalid path. Please provide a valid video file or directory.")