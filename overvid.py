import sys
import os
import contextily as ctx
from datashader.utils import lnglat_to_meters
import logging
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.animation import PillowWriter
import seaborn as sns
import subprocess
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import gpxpy
from geopy import distance
import folium
from selenium import webdriver
import time
import srtm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot styles
plt.rcParams['axes.xmargin'] = 0.1
plt.rcParams['axes.ymargin'] = 0.1
sns.set_style('whitegrid')
sns.set_context('talk')

# File paths
video = '/Volumes/Fotos/GoPro/Hero8-first/manana/GX010003'
video_file = video + '.MP4'
gpx_file = 'temp/'+video.split('/')[-1].split('.')[0] #+'.gpx'

# Create temp directory if it doesn't exist
if not os.path.exists('temp'):
    os.makedirs('temp')

def extract_gps_data(video_file, gpx_file):
    os.system(f'gopro2gpx -s -vvv {video_file} {gpx_file}')

def parse_gpx_data(gpx_file):
    with open(gpx_file+".gpx") as fh:
        gpx_data = gpxpy.parse(fh)
    segment = gpx_data.tracks[0].segments[0]
    coords = pd.DataFrame([
        {'lat': p.latitude, 
         'lon': p.longitude, 
         'ele': p.elevation,
         'time': p.time} for p in segment.points])
    coords.set_index('time', drop=True, inplace=True)
    return coords

def compute_speed_and_acceleration(df):
    df['speed'] = df['dist'].diff().fillna(0)
    df['speed_kmh'] = df['speed'] * 3.6
    df['acceleration'] = df['speed'].diff().fillna(0)
    df['acceleration_g'] = df['acceleration'] / 9.81
    return df

def compute_distances(df):
    df['lat_shift'] = df['lat'].shift()
    df['lon_shift'] = df['lon'].shift()
    df['lat_shift'].iloc[0] = df['lat'].iloc[0]
    df['lon_shift'].iloc[0] = df['lon'].iloc[0]
    
    def calc_distance(row):
        return distance.distance((row['lat_shift'], row['lon_shift']), (row['lat'], row['lon'])).meters
    
    df['dist'] = df.apply(calc_distance, axis=1)
    df.drop(columns=['lat_shift', 'lon_shift'], inplace=True)
    return df
'''
def generate_elevation_GIF(coords, video_fps):
    logging.info('Generating elevation GIF...')
    coords_plot = coords.reset_index(drop=True)
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    fig.patch.set_facecolor('grey')
    fig.patch.set_alpha(0.5)
    ax.set_xlabel('t')
    ax.set_ylabel('Elevation [m]')
    ax.set_xlim([-1, len(coords_plot)])
    ax.set_ylim([min(coords_plot['ele']), max(coords_plot['ele'])])
    ax.get_xaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.plot(coords_plot['ele'])
    point, = ax.plot(0, coords_plot['ele'].iloc[0], marker="o", markersize=5, color='red', ls="")
    interval = 1000 / video_fps
    ani = animation.FuncAnimation(fig, lambda i: point.set_data(i, coords_plot['ele'].iloc[i]), frames=len(coords_plot), interval=interval)
    writerGIF = animation.PillowWriter(fps=1)#video_fps)
    ani.save('temp/elevation.GIF', writer=writerGIF)
'''
def generate_elevation_GIF(coords, video_fps):
    logging.info('Generating elevation GIF...')
    
    # Resetting the index for the coordinates
    coords_plot = coords.reset_index(drop=True)
    
    # Creating the figure and axis objects
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    
    # Making the exterior of the plot transparent
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Setting labels and limits for the axes
    ax.set_xlabel('t')
    ax.set_ylabel('Elevation (m)', labelpad=20)  # Adjusted label padding
    ax.yaxis.set_label_position("right")  # Moving the y-label inside the plot
    ax.set_xlim([-1, len(coords_plot)])
    ax.set_ylim([min(coords_plot['ele']), max(coords_plot['ele'])])
    
    # Hiding the x-axis and adjusting the spines
     # Hiding the x-axis and adjusting the spines
    ax.get_xaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)  # Hide the left spine
    ax.xaxis.set_ticks_position('none')
    
    # Adjust the position of the y-tick labels to replace the y-axis line
    ax.tick_params(axis='y', pad=5)  # Adjust the padding to bring y-tick labels closer to the plot
    
    # Setting the y-axis label properties
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')  # Making the font bold
        label.set_fontsize(10)  # Adjusting font size
        label.set_backgroundcolor((1, 1, 1, 0.5))  # Setting a semi-transparent white background
        label.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.6))  # Adding a border around the label

    # Plotting the elevation data
    ax.plot(coords_plot['ele'], color='green')  # Green color for the elevation line
    ax.fill_between(coords_plot.index, coords_plot['ele'], color='#B69972', alpha=0.3)  # Updated sand-like color for the fill
    
    # Plotting the moving point on the elevation line
    point, = ax.plot(0, coords_plot['ele'].iloc[0], marker="o", markersize=5, color='red', ls="")
    
    # Creating the animation
    interval = 1000 / video_fps
    ani = animation.FuncAnimation(fig, lambda i: point.set_data(i, coords_plot['ele'].iloc[i]), frames=len(coords_plot), interval=interval)
    
    # Saving the animation as a GIF
    ani.save('temp/elevation.GIF', writer='pillow', savefig_kwargs={'transparent': True})

'''
def generate_speedmap_image(coords):
    m = folium.Map(location=[coords['lat'].mean(), coords['lon'].mean()], zoom_start=15)
    
    # Normalize the speed for color mapping
    max_speed = coords['speed_kmh'].max()
    min_speed = coords['speed_kmh'].min()
    normed_speed = (coords['speed_kmh'] - min_speed) / (max_speed - min_speed)
    
    # Create a color gradient based on speed
    colors = plt.cm.RdYlBu(normed_speed)
    colors = [matplotlib.colors.rgb2hex(c) for c in colors]
    
    # Add segments to the map with appropriate colors
    for i in range(1, len(coords)):
        folium.PolyLine(
            locations=[coords[['lat', 'lon']].iloc[i-1].values, coords[['lat', 'lon']].iloc[i].values],
            color=colors[i],
            weight=5
        ).add_to(m)
    
    m.save(os.path.abspath(os.path.join('temp', 'speed_map.html')))
    
    # Convert HTML to PNG using selenium
    options = webdriver.FirefoxOptions()
    options.headless = True
    driver = webdriver.Firefox(options=options)
    driver.get('file://' + os.path.abspath(os.path.join('temp', 'speed_map.html')))
    time.sleep(5)
    driver.save_screenshot('temp/speed_map.png')
    driver.quit()

def generate_speedmap_GIF(coords):
    images = []
    
    m = folium.Map(location=[coords['lat'].mean(), coords['lon'].mean()], zoom_start=15)
    
    # Normalize the speed for color mapping
    max_speed = coords['speed_kmh'].max()
    min_speed = coords['speed_kmh'].min()
    normed_speed = (coords['speed_kmh'] - min_speed) / (max_speed - min_speed)
    
    # Create a color gradient based on speed
    colors = plt.cm.RdYlBu(normed_speed)
    colors = [matplotlib.colors.rgb2hex(c) for c in colors]
    
    # Add segments to the map with appropriate colors
    for i in range(1, len(coords)):
        folium.PolyLine(
            locations=[coords[['lat', 'lon']].iloc[i-1].values, coords[['lat', 'lon']].iloc[i].values],
            color=colors[i],
            weight=5
        ).add_to(m)
        
        # Add the current position marker
        folium.Marker(
            location=tuple(coords[['lat', 'lon']].iloc[i].values),
            icon=folium.Icon(color='red')
        ).add_to(m)

        ## TO DO: Change marker icon: 
        # https://github.com/python-visualization/folium/blob/461479e7c6657053e5c8285876bb9fecb0a4267b/tests/test_folium.py#L298

        
        m.save(os.path.join('temp', f'speed_map_frame_{i}.html'))
        
        # Convert HTML to PNG using selenium
        options = webdriver.FirefoxOptions()
        options.headless = True
        driver = webdriver.Firefox(options=options)
        driver.get('file://' + os.path.abspath(os.path.join('temp', f'speed_map_frame_{i}.html')))
        time.sleep(2)  # Give it a few seconds to render
        driver.save_screenshot(os.path.join('temp', f'speed_map_frame_{i}.png'))
        driver.quit()
        
        images.append(imageio.imread(os.path.join('temp', f'speed_map_frame_{i}.png')))
    
    # Convert images to GIF
    imageio.mimsave('temp/speed_map.GIF', images, fps=1)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def capture_map_images(coords_chunk, colors):
    images = []
    
    m = folium.Map(location=[coords_chunk['lat'].mean(), coords_chunk['lon'].mean()], zoom_start=15)
    
    for i in range(1, len(coords_chunk)):
        folium.CircleMarker(
            location=tuple(coords_chunk[['lat', 'lon']].iloc[i].values),
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)

        folium.PolyLine(
            locations=[coords_chunk[['lat', 'lon']].iloc[i-1].values, coords_chunk[['lat', 'lon']].iloc[i].values],
            color=colors[i],
            weight=5
        ).add_to(m)

        html_string = m.get_root().render()
        options = webdriver.FirefoxOptions()
        options.headless = True
        driver = webdriver.Firefox(options=options)
        driver.get("data:text/html;charset=utf-8," + html_string)
        time.sleep(2)  # Give it a few seconds to render
        png = driver.get_screenshot_as_png()
        driver.quit()

        images.append(imageio.imread(png))
    return images


def generate_speedmap_GIF_optimized_parallel(coords, video_fps):
    images = []
    
    # Normalize the speed for color mapping
    max_speed = coords['speed_kmh'].max()
    min_speed = coords['speed_kmh'].min()
    normed_speed = (coords['speed_kmh'] - min_speed) / (max_speed - min_speed)
    
    # Create a color gradient based on speed
    colors = plt.cm.RdYlBu(normed_speed)
    colors = [matplotlib.colors.rgb2hex(c) for c in colors]
    
    # Limit the number of processes to the number of available CPU cores
    num_processes = os.cpu_count() - 3
    
    # Split the coordinates into chunks
    coords_chunks = list(chunks(coords, len(coords) // num_processes))
    
    results = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {}
        for idx, chunk in enumerate(coords_chunks):
            # Check if the chunk's result already exists
            temp_filename = f'temp/chunk_{idx}.png'
            if os.path.exists(temp_filename):
                # Load the image directly if it exists
                images.append(imageio.imread(temp_filename))
            else:
                # Otherwise, process the chunk
                futures[executor.submit(capture_map_images, chunk, colors)] = idx
        
        for future in tqdm(futures, total=len(futures), desc="Processing chunks"):
            idx = futures[future]
            chunk_images = future.result()
            results[idx] = chunk_images
            
            # Save the chunk's result as a temporary image file
            for img_idx, img in enumerate(chunk_images):
                temp_filename = f'temp/chunk_{idx}_{img_idx}.png'
                imageio.imsave(temp_filename, img)
    
    # Sort the results by the chunk index and extract the images in order
    for idx in sorted(results.keys()):
        images.extend(results[idx])
    
    # Convert images to GIF
    imageio.mimsave('temp/speed_map_optimized_parallel.GIF', images, fps=1)


def generate_speedmap_GIF_simplified(coords, video_fps):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define the bounding box based on the coordinates
    llcrnrlat, llcrnrlon = coords['lat'].min() - 0.01, coords['lon'].min() - 0.01
    urcrnrlat, urcrnrlon = coords['lat'].max() + 0.01, coords['lon'].max() + 0.01
    
    # Convert lat/lon to web mercator
    coords['x'], coords['y'] = ctx.lnglat_to_meters(coords['lon'], coords['lat'])
    
    # Set axis limits
    ax.set_xlim([coords['x'].min() - 1000, coords['x'].max() + 1000])
    ax.set_ylim([coords['y'].min() - 1000, coords['y'].max() + 1000])
    
    # Normalize the speed for color mapping
    max_speed = coords['speed_kmh'].max()
    min_speed = coords['speed_kmh'].min()
    normed_speed = (coords['speed_kmh'] - min_speed) / (max_speed - min_speed)
    
    # Create a color gradient based on speed
    colors = plt.cm.RdYlBu(normed_speed)
    
    images = []
    for i in range(1, len(coords)):
        ax.plot(coords['x'].iloc[i], coords['y'].iloc[i], 'o', markersize=5, color=colors[i])
        
        # Add the satellite background
        ctx.add_basemap(ax, source=ctx.providers.GoogleMaps.Satellite, zoom=15)
        
        # Capture the current state of the plot as an image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        ax.clear()  # Clear the axis for the next frame
    
    # Convert images to GIF
    imageio.mimsave('temp/speed_map_simplified.GIF', images, fps=1)
'''    

def generate_frame(i, coords, colors):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(coords['x'].iloc[i], coords['y'].iloc[i], 'o', markersize=5, color=colors[i])
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=15)
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
    plt.close(fig)
    return i, image

def generate_speedmap_GIF_with_zoom(coords):
    # Create a new figure for zoom-in frames
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Convert lat/lon to web mercator
    coords['x'], coords['y'] = lnglat_to_meters(coords['lon'], coords['lat'])
    
    # Normalize the speed for color mapping
    max_speed = coords['speed_kmh'].max()
    min_speed = coords['speed_kmh'].min()
    normed_speed = (coords['speed_kmh'] - min_speed) / (max_speed - min_speed)
    
    # Create a color gradient based on speed
    colors = plt.cm.RdYlBu(normed_speed)
    
    images = []
    
    # Generate zoom-in frames
    zoom_levels = np.round(np.linspace(4, 15, 8)).astype(int)
    for zoom in tqdm(zoom_levels, desc="Generating zoom-in frames"):
        ax.set_xlim([coords['x'].min() - 1000 * (16 - zoom), coords['x'].max() + 1000 * (16 - zoom)])
        ax.set_ylim([coords['y'].min() - 1000 * (16 - zoom), coords['y'].max() + 1000 * (16 - zoom)])
        
        # Add the satellite background
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=zoom)
        
        # Capture the current state of the plot as an image
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
        images.append(image)
        ax.clear()

    plt.close(fig)  # Close the figure used for zoom-in frames

    # Parallelize the main video frame generation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(generate_frame, range(1, len(coords)), [coords]*len(coords), [colors]*len(coords)), total=len(coords)-1, desc="Generating main video frames"))

    # Sort results by the frame order and append to images
    for i, image in sorted(results):
        images.append(image)

    # Convert images to GIF
    imageio.mimsave('temp/speed_map_with_zoom.GIF', images, fps=4)


def generate_kpi_gif(coords):
    coords_plot = coords.reset_index(drop=True)
    
    # Create a blank figure and axis
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    canvas = FigureCanvas(fig)
    
    # Use a different font (if available)
    font_properties = {'family': 'monospace', 'weight': 'bold', 'size': 14}
    
    images = []
    
    for i in range(len(coords_plot)):
        ax.clear()
        ax.axis('off')
        
        # Display speed in green
        speed_text = f"Speed: {coords_plot['speed_kmh'].iloc[i]:.2f} km/h"
        ax.text(0.05, 0.05, speed_text, transform=ax.transAxes, fontsize=14, color='green', fontweight='bold', **font_properties)
        
        # Display acceleration in terms of g-force in red
        g_force = coords_plot['acceleration_g'].iloc[i]
        g_force_text = f"G-Force: {g_force:.2f} g"
        ax.text(0.05, 0.1, g_force_text, transform=ax.transAxes, fontsize=14, color='red', fontweight='bold', **font_properties)
        
        # Convert the figure to an image
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
    
    # Convert images to GIF
    imageio.mimsave('temp/kpi.gif', images, fps=1)


def get_video_fps(video_file):
    cmd = f'ffmpeg -i {video_file} 2>&1 | sed -n "s/.*, \\(.*\\) fp.*/\\1/p"'
    fps = float(os.popen(cmd).read())
    return fps

'''
def overlay_elevation_GIF_on_video(video_file):
    os.system(f'ffmpeg -y -i {video_file} -i elevation.GIF -filter_complex "[1:v]scale=300:-1[temp],[0:v][temp] overlay=10:10" elev_video.mp4')

def overlay_speedmap_GIF_on_video(video_file):
    os.system(f'ffmpeg -y -i elev_video.mp4 -i speed_map.GIF -filter_complex "[1:v]scale=300:-1[temp],[0:v][temp] overlay=main_w-overlay_w-10:main_h-overlay_h-10" speedm_video.mp4')

def overlay_kpis_GIF_on_video(video_file):
    os.system(f'ffmpeg -i speedm_video.mp4 -i kpi.gif -filter_complex "[1:v]scale=-1:480[temp],[0:v][temp] overlay=10:main_h-overlay_h-10" final_out.mp4')
'''

def overlay_all_elements_on_video(video_file):
    """
    Overlay the generated GIFs on the original video.
    
    Args:
    - video_file (str): Path to the original video file.
    """

    """
    # speedmap GIF version: 
    os.system(f'''
        ffmpeg -y -i {video_file} \
               -i elevation.GIF \
               -i speed_map.GIF \
               -i kpi.gif \
        -filter_complex "
            [1:v]scale=300:-1[elev_scaled];
            [2:v]scale=300:-1[speedmap_scaled];
            [3:v]scale=-1:480[kpi_scaled];
            [0:v][elev_scaled] overlay=10:10[temp1];
            [temp1][speedmap_scaled] overlay=main_w-overlay_w-10:main_h-overlay_h-10[temp2];
            [temp2][kpi_scaled] overlay=10:main_h-overlay_h-10
        " final_out.mp4
        ''')
    
    # speedmap image version: 
    os.system(f'''
        ffmpeg -y -i {video_file} \
               -i elevation.GIF \
               -i temp/speed_map.png \
               -i kpi.gif \
        -filter_complex "
            [1:v]scale=300:-1[elev_scaled];
            [2:v]scale=300:-1[speedmap_scaled];
            [3:v]scale=-1:480[kpi_scaled];
            [0:v][elev_scaled] overlay=10:10[temp1];
            [temp1][speedmap_scaled] overlay=main_w-overlay_w-10:main_h-overlay_h-10[temp2];
            [temp2][kpi_scaled] overlay=10:main_h-overlay_h-10
        " final_out.mp4
        ''')
    os.system(f'''
    ffmpeg -y -i {video_file} \
           -i temp/elevation.GIF \
           -i temp/speed_map_optimized_parallel.GIF \
           -i temp/kpi.gif \
    -filter_complex "
        [1:v]scale=300:-1[elev_scaled];
        [2:v]scale=300:-1[speedmap_scaled];
        [3:v]scale=-1:480[kpi_scaled];
        [0:v][elev_scaled] overlay=10:10[temp1];
        [temp1][speedmap_scaled] overlay=main_w-overlay_w-10:main_h-overlay_h-10[temp2];
        [temp2][kpi_scaled] overlay=10:main_h-overlay_h-10
    " final_out.mp4
    ''')
    """
    # Construct the output filename
    output_filename = os.path.join(os.path.dirname(video_file), "overlay_" + os.path.basename(video_file))
    
    try:
        # Construct the ffmpeg command
        cmd = [
            'ffmpeg', '-y', 
            '-i', video_file,
            '-i', 'temp/elevation.GIF',
            '-i', 'temp/speed_map_optimized_parallel.GIF',
            '-i', 'temp/kpi.gif',
            '-filter_complex',
            """
            [1:v]scale=300:-1[elev_scaled];
            [2:v]scale=300:-1[speedmap_scaled];
            [3:v]scale=-1:480[kpi_scaled];
            [0:v][elev_scaled] overlay=0:0[temp1];  # Adjusted elevation GIF to top-left
            [temp1][speedmap_scaled] overlay=main_w-overlay_w-10:main_h-overlay_h-10[temp2];
            [temp2][kpi_scaled] overlay=(main_w*0.05):main_h-overlay_h-10  # Adjusted KPIs position
            """,
            output_filename  # Use the constructed output filename
        ]
        
        # Run the command and check for errors
        result = subprocess.run(cmd, capture_output=True, text=True)
        result.check_returncode()
        
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg command failed with error: {e.stderr}")



if __name__ == '__main__':


    if not os.path.isfile(os.path.abspath(os.path.join('temp',f'{gpx_file}.gpx'))):
        logging.info('Extracting GPS data...')
        extract_gps_data(video_file, gpx_file)
    else: 
        logging.info('GPS data was already extracted, moving on... ')

    logging.info('Parsing GPX data...')
    coords = parse_gpx_data(gpx_file)

    logging.info('Computing distance...')
    coords = compute_distances(coords) ## add distances
    
    logging.info('Computing speed and acceleration...')
    coords = compute_speed_and_acceleration(coords)
    logging.info(coords.columns.values)
    logging.info(coords.head())
    
    video_fps = get_video_fps(video_file)
    logging.info(f'Video FPS: {video_fps}')

    if not os.path.isfile(os.path.abspath(os.path.join('temp','elevation.GIF'))):
        logging.info('Computing Elevation...')
        generate_elevation_GIF(coords, video_fps)
    else: 
        logging.info('Elevation GIF is already computed, moving on...')
        
    #logging.info('Overlaying GIF on video...')
    #overlay_elevation_GIF_on_video(video_file)

    if not os.path.isfile(os.path.abspath(os.path.join('temp','speed_map_simplified.GIF'))):
        logging.info('Generating speedmap...')
        #generate_speedmap_image(coords)
        #generate_speedmap_GIF(coords)
        #generate_speedmap_GIF_optimized_parallel(coords, video_fps)
        #generate_speedmap_GIF_simplified(coords, video_fps)
        generate_speedmap_GIF_with_zoom(coords)
    else: 
        logging.info('Speedmap is already generated for this file, moving on...')

    #logging.info('Overlaying speedmap on video...')
    #overlay_speedmap_GIF_on_video(video_file)

    logging.info('Generating Speed and Acc. GIFs...')
    generate_kpi_gif(coords)

    #logging.info('Overlaying KPIs v/a GIF on video...')
    #overlay_kpis_GIF_on_video(video) 

    logging.info('Overlaying all GIFs at once...')
    overlay_all_elements_on_video(video_file)
    

    logging.info('Done!')
