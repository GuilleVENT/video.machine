import sys
import os
import logging
import imageio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import warnings
import numpy as np
import pandas as pd
import gpxpy
from geopy import distance
import folium
from selenium import webdriver
import time
import srtm

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
gpx_file = video

def extract_gps_data(video_file, gpx_file):
    os.system(f'gopro2gpx -s -vvv {video_file} {gpx_file}')

def parse_gpx_data(gpx_file):
    with open(gpx_file + '.gpx') as fh:
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
    return df[['speed_kmh', 'acceleration_g']]

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
    ani.save('elevation.GIF', writer=writerGIF)

def generate_speed_map_image(coords):
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
    
    m.save('speed_map.html')
    
    # Convert HTML to PNG using selenium
    options = webdriver.FirefoxOptions()
    options.headless = True
    driver = webdriver.Firefox(options=options)
    driver.get('file://' + os.path.abspath('speed_map.html'))
    time.sleep(5)
    driver.save_screenshot('speed_map.png')
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
    imageio.mimsave('speed_map.GIF', images, fps=1)

def generate_kpi_gif(coords):
    coords_plot = coords.reset_index(drop=True)
    
    # Create a blank figure and axis
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    canvas = FigureCanvas(fig)
    
    images = []
    
    for i in range(len(coords_plot)):
        ax.clear()
        ax.axis('off')
        
        # Display speed
        speed_text = f"Speed: {coords_plot['speed_kmh'].iloc[i]:.2f} km/h"
        ax.text(0.05, 0.1, speed_text, transform=ax.transAxes, fontsize=12, color='white', fontweight='bold')
        
        # Display acceleration in terms of g-force
        g_force = coords_plot['acceleration_g'].iloc[i]
        g_force_text = f"G-Force: {g_force:.2f} g"
        ax.text(0.9, 0.9, g_force_text, transform=ax.transAxes, fontsize=12, color='white', fontweight='bold', ha='right')
        
        # Convert the figure to an image
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
    
    # Convert images to GIF
    imageio.mimsave('kpi.gif', images, fps=1)

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


if __name__ == '__main__':
    logging.info('Extracting GPS data...')
    extract_gps_data(video_file, gpx_file)
    
    logging.info('Parsing GPX data...')
    coords = parse_gpx_data(gpx_file)

    coords = compute_distances(coords)
    
    logging.info('Computing speed and acceleration...')
    compute_speed_and_acceleration(coords)
    
    video_fps = get_video_fps(video_file)

    logging.info('Computing Elevation...')
    generate_elevation_GIF(coords, video_fps)
    
    #logging.info('Overlaying GIF on video...')
    #overlay_elevation_GIF_on_video(video_file)

    logging.info('Generating speedmap...')
    generate_speedmap_GIF(coords)

    #logging.info('Overlaying speedmap on video...')
    #overlay_speedmap_GIF_on_video(video_file)

    logging.info('Generating KPIs v/a...')
    generate_kpi_gif(coords)

    #logging.info('Overlaying KPIs v/a GIF on video...')
    #overlay_kpis_GIF_on_video(video) 

    logging.info('Overlaying all GIFs at once...')
    overlay_all_elements_on_video(video_file)
    

    logging.info('Done!')
