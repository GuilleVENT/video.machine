import cv2
import numpy as np
import imutils
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(filename='app.log', filemode='w', 
                    format='%(name)s - %(levelname)s - %(message)s')


def load_yolo_model():
    try:
        yolo_network = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = yolo_network.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in yolo_network.getUnconnectedOutLayers()]
        return yolo_network, output_layers
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        exit()


def process_video_frame(frame):
    try:
        resized_frame = imutils.resize(frame, width=500)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        return blurred_frame
    except Exception as e:
        logging.error(f"Error processing frame: {e}")


def detect_frame_motion(first_frame, current_frame):
    try:
        frame_delta = cv2.absdiff(first_frame, current_frame)
        threshold_frame = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
        dilated_frame = cv2.dilate(threshold_frame, None, iterations=2)
        frame_contours = cv2.findContours(dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_contours = imutils.grab_contours(frame_contours)
        return frame_contours
    except Exception as e:
        logging.error(f"Error detecting motion: {e}")


def detect_objects_in_frame(frame, yolo_network, output_layers):
    try:
        frame_height, frame_width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_network.setInput(blob)
        object_detections = yolo_network.forward(output_layers)
        return object_detections, frame_width, frame_height
    except Exception as e:
        logging.error(f"Error detecting objects: {e}")


def main():
    try:
        video_capture = cv2.VideoCapture('http://your_ip_camera_address:port/video')
        if not video_capture.isOpened():
            logging.error("Error: Could not connect to camera.")
            exit()

        yolo_network, output_layers = load_yolo_model()
        read_success, first_frame = video_capture.read()
        if not read_success:
            logging.error("Error: Could not read from camera.")
            exit()

        first_frame = process_video_frame(first_frame)
        video_codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format

        while True:
            read_success, current_frame = video_capture.read()
            if not read_success:
                logging.error("Error: Could not read from camera.")
                break

            processed_frame = process_video_frame(current_frame)
            frame_contours = detect_frame_motion(first_frame, processed_frame)

            if frame_contours:
                object_detections, frame_width, frame_height = detect_objects_in_frame(
                    current_frame, yolo_network, output_layers
                )
                for detection in object_detections:
                    for obj in detection:
                        scores = obj[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x, center_y, width, height = (
                                obj[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                            ).astype('int')
                            x = int(center_x - width / 2)
                            y = int(center_y - height / 2)
                            cv2.rectangle(current_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_path = f"/Volumes/Fotos/IPcam/events/event-{timestamp}.mp4"
                video_writer = cv2.VideoWriter(output_path, video_codec, 20.0, (1280, 720))
                video_writer.write(current_frame)
                video_writer.release()

            first_frame = processed_frame  # Update the first frame for motion detection

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
