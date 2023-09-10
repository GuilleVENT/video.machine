import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# File paths
video = '/path/to/your/video.mp4'
output_video = '/path/to/your/output_video.mp4'

# Parameters for cutting and manipulating the video
start_time = '00:01:00'  # Start time in HH:MM:SS format
end_time = '00:02:00'   # End time in HH:MM:SS format
bitrate = '500k'        # Desired bitrate
resolution = '640x480'  # Desired resolution

def cut_and_manipulate_video(video_file, output_file, start, end, br, res):
    logging.info('Cutting and manipulating video...')
    
    # FFmpeg command to cut the video, reduce bitrate, and lower resolution
    cmd = (f'ffmpeg -i {video_file} '
           f'-ss {start} -to {end} '  # Cutting the video
           f'-b:v {br} '              # Setting the bitrate
           f'-s {res} '               # Setting the resolution
           f'-c:a copy '              # Copying audio without re-encoding
           f'-map 0 '                 # Mapping all streams (to keep GPX data)
           f'{output_file}')
    
    os.system(cmd)
    logging.info('Video manipulation complete.')

if __name__ == '__main__':
    cut_and_manipulate_video(video, output_video, start_time, end_time, bitrate, resolution)
    logging.info('Done!')
