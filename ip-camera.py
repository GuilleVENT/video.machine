import cv2
import numpy as np
import imutils
import logging
from datetime import datetime
import urllib.request
import os
import threading 
import time 
import urllib.request

# Setup logging
logging.basicConfig(filename='ip-camera.log', filemode='w', 
                    format='%(name)s - %(levelname)s - %(message)s')


flashlight_on = False

ANDROID_IP_PORT = "****"
PI_CAM_IP_PORT  = "****"

IP_PORT  =  ANDROID_IP_PORT#

if IP_PORT == ANDROID_IP_PORT:
    url_suffix = '/video'
else:
    url_suffix = ''

URL_VID_STREAM = f'{IP_PORT}{url_suffix}'

VIDEO_PATH_PC = "\\\\guillermonas.local\\Fotos\\IPcam\\events"
VIDEO_PATH_MAC= "/Volumes/Fotos/IPcam/events"
VIDEO_PATH_PI = "/mnt/nas/Fotos/IPcam/events"
## To Do - detect OS 
## + change path 
VIDEO_PATH    = VIDEO_PATH_PI

def load_labels(url):
    with urllib.request.urlopen(url) as response:
        labels = response.read().decode('utf-8').strip().split('\n')
    return labels

def load_yolo_model():
    try:
        yolo_network = cv2.dnn.readNet(
                              os.path.join(os.getcwd(), "yolov3-tiny.weights")#"yolov3-tiny.weights")
                            , os.path.join(os.getcwd(), "yolov3-tiny.cfg")#"yolov3-tiny.cfg")
                            )
        
        layer_names = yolo_network.getLayerNames()

        output_layers = [layer_names[i - 1] for i in yolo_network.getUnconnectedOutLayers()]

        #output_layers = [layer_names[i[0] - 1] for i in yolo_network.getUnconnectedOutLayers()]
        '''output_layers = []
        unconnected_layers = yolo_network.getUnconnectedOutLayers()
        for i in unconnected_layers:
            print(i)  # See what this prints
            output_layers.append(layer_names[i[0] - 1])
        print('success')
        '''
        return yolo_network, output_layers
    except Exception as e:
        logging.error(f" {str(os.sys.exc_info()[-1].tb_lineno)} - Error loading YOLO model: {e}")
        exit()

def toggle_flashlight(interval=0.8):
    """
    Toggles the flashlight on and off at the specified interval.
    """
    global flashlight_on
    try:
        while flashlight_on:
            urllib.request.urlopen(f'{IP_PORT}/enabletorch')
            time.sleep(interval)
            urllib.request.urlopen(f'{IP_PORT}/disabletorch')
            time.sleep(interval)

        # Ensure the flashlight is turned off when exiting the loop
        urllib.request.urlopen(f'{IP_PORT}/disabletorch')

    except:
        logging.error(f" {str(os.sys.exc_info()[-1].tb_lineno)} - Error toggling flashlight.")


def process_video_frame(frame):
    try:
        resized_frame = imutils.resize(frame, width=500)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        return blurred_frame
    except Exception as e:
        logging.error(f" {str(os.sys.exc_info()[-1].tb_lineno)} - Error processing frame: {e}")


def detect_frame_motion(first_frame, current_frame):
    try:
        frame_delta = cv2.absdiff(first_frame, current_frame)
        threshold_frame = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
        dilated_frame = cv2.dilate(threshold_frame, None, iterations=2)
        frame_contours = cv2.findContours(dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_contours = imutils.grab_contours(frame_contours)
        return len(frame_contours) > 0
    except Exception as e:
        logging.error(f" {str(os.sys.exc_info()[-1].tb_lineno)} - Error detecting motion: {e}")


def detect_objects_in_frame(frame, yolo_network, output_layers, confidence_threshold=0.5):
    try:
        frame_height, frame_width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_network.setInput(blob)
        object_detections_raw = yolo_network.forward(output_layers)
        
        # Filter out detections below the confidence threshold
        object_detections = [detection for detection in object_detections_raw[0] if detection[4] > confidence_threshold]

        return object_detections, frame_width, frame_height
    except Exception as e:
        logging.error(f" {str(os.sys.exc_info()[-1].tb_lineno)} - Error detecting objects: {e}")

def detect_and_annotate_objects(video_path, yolo_network, output_layers, output_path, confidence_threshold=0.5):
    video_capture = cv2.VideoCapture(video_path)
    video_writer = None

    labels_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    labels = load_labels(labels_url)

    while True:
        read_success, frame = video_capture.read()
        if not read_success:
            break

        object_detections_raw, frame_width, frame_height = detect_objects_in_frame(
            frame, yolo_network, output_layers
        )

        # Filter out detections below the confidence threshold
        object_detections = [detection for detection in object_detections_raw if detection[4] > confidence_threshold]

        for obj in object_detections:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and labels[class_id] in ['person', 'cat', 'dog']:
                center_x, center_y, width, height = (
                    obj[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                ).astype('int')
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                label = f"{labels[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

        video_writer.write(frame)

    video_capture.release()
    video_writer.release()



def main():
    try:
        OBJECT_PERSISTENCE = 80  # Number of frames to wait after object is no longer detected (20 seconds for 30fps)

        object_detected = False
        object_counter = 0

        MAX_FRAMES = 600  # This will limit a single video to 30 seconds (assuming 20fps)
        frame_counter = 0

        global flashlight_on
        flashlight_thread = None  # This will hold our flashlight toggling thread

        objects_of_interest = ['person', 
                                'cat',
                                'dog', 
                                'bird', 
                                'teddy bear',
                                'backpack',
                                'handbag',
                                'skateboard',
                                'knife',
                                'surfboard',
                                'bicycle']

        video_capture = cv2.VideoCapture(URL_VID_STREAM)
        if not video_capture.isOpened():
            logging.error("Could not open video stream.")
            exit()

        labels = load_labels('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
        yolo_network, output_layers = load_yolo_model()
        video_writer = None
        first_frame = None

        while True:
            read_success, frame_ = video_capture.read()
            frame = cv2.flip(frame_, -1)

            if not read_success:
                logging.error("Failed to read frame from the camera.")
                break

            # Check for object detection
            object_detections, _, _ = detect_objects_in_frame(frame, yolo_network, output_layers)
            #objects_in_frame = [labels[np.argmax(obj[5:])] for detection in object_detections for obj in detection]
            
            objects_in_frame = list(set([labels[np.argmax(detection[5:])] for detection in object_detections]))

            print(objects_in_frame)
            '''
            if 'person' in objects_in_frame:
                if flashlight_thread is None or not flashlight_thread.is_alive():
                    ## developing: turning off this annoying
                    #flashlight_on = True
                    flashlight_thread = threading.Thread(target=toggle_flashlight, args=(0.2,))  # 0.2 seconds interval for rapid flashing
                    flashlight_thread.start()
            else:
                flashlight_on = False
            '''
            if any(obj in objects_of_interest for obj in objects_in_frame):
                object_detected = True
                object_counter = OBJECT_PERSISTENCE
            elif object_counter > 0:
                object_detected = True
                object_counter -= 1
            else:
                object_detected = False

            # Start/Continue recording if object is detected
            if object_detected and video_writer is None: ## start it up there's something on this frame ! 
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                temp_video_path = f"temp_event-{timestamp}.mp4"
                try:
                    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (frame.shape[1], frame.shape[0]))
                except: 
                    # Try using the MJPG codec instead
                    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame.shape[1], frame.shape[0]))

            if video_writer: # was itiated 
                video_writer.write(frame)
                frame_counter += 1
                if frame_counter > MAX_FRAMES: ## check for length 
                    video_writer.release()
                    video_writer = None
                    frame_counter = 0
                    output_video_path = f"{VIDEO_PATH}/event-{timestamp}.mp4"
                    detect_and_annotate_objects(temp_video_path, yolo_network, output_layers, output_video_path)
                    os.remove(temp_video_path)

            # Stop recording if the object is no longer detected for a set duration
            if not object_detected and video_writer is not None:
                video_writer.release()
                video_writer = None
                output_video_path = f"{VIDEO_PATH}/event-{timestamp}.mp4"
                detect_and_annotate_objects(temp_video_path, yolo_network, output_layers, output_video_path)
                os.remove(temp_video_path)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

        video_capture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f" {str(os.sys.exc_info()[-1].tb_lineno)} - An error occurred: {e}")



if __name__ == "__main__":
    main()
