import cv2
import numpy as np
import imutils
import logging
from datetime import datetime
import urllib.request
import os
import threading
import time


# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='ip-camera.log', filemode='w')

def load_labels(url):
    try:
        with urllib.request.urlopen(url) as response:
            labels = response.read().decode('utf-8').strip().split('\n')
        return labels
    except Exception as e:
        logging.error(f"Error loading labels from {url}.", exc_info=True)
        return []

class Config:
    ANDROID_IP_PORT = "****"
    PI_CAM_IP_PORT = "****"
    IP_PORT = ANDROID_IP_PORT
    URL_VID_STREAM = f'{IP_PORT}/video' if IP_PORT == ANDROID_IP_PORT else IP_PORT
    VIDEO_PATH = {
        'PC': "\\\\guillermonas.local\\Fotos\\IPcam\\events",
        'MAC': "/Volumes/Fotos/IPcam/events",
        'PI': "/mnt/nas/Fotos/IPcam/events"
    }['MAC']
    # TODO: Add OS detection logic to select appropriate VIDEO_PATH
    OBJECT_PERSISTENCE = 80
    MAX_FRAMES = 600
    OBJECTS_OF_INTEREST = [
        'person', 'cat', 'dog', 'bird', 'teddy bear',
        'backpack', 'handbag', 'skateboard', 'knife',
        'surfboard', 'bicycle'
    ]
    LABELS_URL = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'

class FlashlightControl:
    def __init__(self):
        self.flashlight_on = False
        self.thread = None

    def toggle_flashlight(self, interval=0.8):
        try:
            while self.flashlight_on:
                urllib.request.urlopen(f'{Config.IP_PORT}/enabletorch')
                time.sleep(interval)
                urllib.request.urlopen(f'{Config.IP_PORT}/disabletorch')
                time.sleep(interval)
            urllib.request.urlopen(f'{Config.IP_PORT}/disabletorch')
        except:
            logging.error("Error toggling flashlight.", exc_info=True)

    def start_flashlight(self, interval=0.2):
        self.flashlight_on = True
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.toggle_flashlight, args=(interval,))
            self.thread.start()

    def stop_flashlight(self):
        self.flashlight_on = False

class YoloModel:
    def __init__(self):
        self.network, self.output_layers = self.load_model()

    @staticmethod
    def load_model():
        try:
            path = os.getcwd()
            yolo_network = cv2.dnn.readNet(os.path.join(path, "yolov3.weights"), 
                                           os.path.join(path, "yolov3.cfg"))
            layer_names = yolo_network.getLayerNames()
            
            output_layers_indices = yolo_network.getUnconnectedOutLayers()
            output_layers = [layer_names[i[0] - 1] for i in output_layers_indices.reshape(-1, 1)]

            return yolo_network, output_layers
        except:
            logging.error("Error loading YOLO model.", exc_info=True)

class VideoProcessor:
    def __init__(self, yolo):
        self.yolo = yolo
        self.video_writer = None

    def process_frame(self, frame):
        try:
            resized_frame = imutils.resize(frame, width=500)
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            return blurred_frame
        except Exception as e:
            logging.error("Error processing frame.", exc_info=True)

    def detect_motion(self, first_frame, current_frame):
        try:
            frame_delta = cv2.absdiff(first_frame, current_frame)
            threshold_frame = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
            dilated_frame = cv2.dilate(threshold_frame, None, iterations=2)
            frame_contours = cv2.findContours(dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_contours = imutils.grab_contours(frame_contours)
            return len(frame_contours) > 0
        except Exception as e:
            logging.error("Error detecting motion.", exc_info=True)

    def detect_and_annotate_objects(self, video_path, output_path, labels, confidence_threshold=0.5):
        video_capture = cv2.VideoCapture(video_path)
        video_writer = None

        while True:
            read_success, frame = video_capture.read()
            if not read_success:
                break

            object_detections, frame_width, frame_height = self.detect_objects(frame)

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


class ObjectDetector:
    def __init__(self, yolo):
        self.yolo = yolo

    def detect_objects(self, frame, confidence_threshold=0.5):
        try:
            frame_height, frame_width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.yolo.network.setInput(blob)
            object_detections_raw = self.yolo.network.forward(self.yolo.output_layers)
            
            # Filter out detections below the confidence threshold
            object_detections = [detection for detection in object_detections_raw[0] if detection[4] > confidence_threshold]
            return object_detections, frame_width, frame_height
        except Exception as e:
            logging.error("Error detecting objects.", exc_info=True)

    def annotate_objects(self, frame, object_detections, labels, confidence_threshold=0.5):
        for obj in object_detections:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and labels[class_id] in Config.OBJECTS_OF_INTEREST:
                center_x, center_y, width, height = (
                    obj[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                ).astype('int')
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                label = f"{labels[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    flashlight = FlashlightControl()
    yolo = YoloModel()
    processor = VideoProcessor(yolo)
    detector = ObjectDetector(yolo)

    video_capture = cv2.VideoCapture(Config.URL_VID_STREAM)
    if not video_capture.isOpened():
        logging.error("Could not open video stream.")
        exit()

    labels = load_labels(Config.LABELS_URL)
    first_frame = None
    object_detected = False
    object_counter = 0
    frame_counter = 0

    while True:
        read_success, frame_ = video_capture.read()
        frame = cv2.flip(frame_, -1)

        if not read_success:
            logging.error("Failed to read frame from the camera.")
            break

        # Object detection
        object_detections, _, _ = detector.detect_objects(frame)
        objects_in_frame = list(set([labels[np.argmax(detection[5:])] for detection in object_detections]))

        if 'person' in objects_in_frame:
            flashlight.start_flashlight()
        else:
            flashlight.stop_flashlight()
        
        print(objects_in_frame)

        if any(obj in Config.OBJECTS_OF_INTEREST for obj in objects_in_frame):
            object_detected = True
            object_counter = Config.OBJECT_PERSISTENCE
        elif object_counter > 0:
            object_detected = True
            object_counter -= 1
        else:
            object_detected = False

        # Video processing
        if object_detected and processor.video_writer is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            temp_video_path = f"temp_event-{timestamp}.mp4"
            try:
                processor.video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (frame.shape[1], frame.shape[0]))
            except:
                processor.video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame.shape[1], frame.shape[0]))

        if processor.video_writer:
            processor.video_writer.write(frame)
            frame_counter += 1
            if frame_counter > Config.MAX_FRAMES:
                processor.video_writer.release()
                processor.video_writer = None
                frame_counter = 0
                output_video_path = f"{Config.VIDEO_PATH}/event-{timestamp}.mp4"
                detector.detect_and_annotate_objects(temp_video_path, output_video_path, labels)
                os.remove(temp_video_path)

        if not object_detected and processor.video_writer is not None:
            processor.video_writer.release()
            processor.video_writer = None
            output_video_path = f"{Config.VIDEO_PATH}/event-{timestamp}.mp4"
            detector.detect_and_annotate_objects(temp_video_path, output_video_path, labels)
            os.remove(temp_video_path)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

