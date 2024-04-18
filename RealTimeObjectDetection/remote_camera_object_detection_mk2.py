import os
import cv2
import time
import shutil
import logging
import argparse
import traceback
import threading
import RPi.GPIO as GPIO
from flask import Flask, Response
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor

class LEDRGB:
    colors = {
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "yellow": (1, 1, 0),
        "magenta": (1, 0, 1),
        "cyan": (0, 1, 1),
        "white": (1, 1, 1),
        "off": (0, 0, 0)
    }

    def __init__(self, red_led=33, green_led=35, blue_led=37):
        GPIO.setmode(GPIO.BOARD)
        self.red_led = red_led
        self.green_led = green_led
        self.blue_led = blue_led
        GPIO.setup(self.red_led, GPIO.OUT)
        GPIO.setup(self.green_led, GPIO.OUT)
        GPIO.setup(self.blue_led, GPIO.OUT)
    
    def _set_color(self, color_name):
        color = self.colors.get(color_name.lower(), self.colors["off"])
        GPIO.output(self.red_led, color[0])
        GPIO.output(self.green_led, color[1])
        GPIO.output(self.blue_led, color[2])

    def __getattr__(self, color_name):
        return lambda: self._set_color(color_name.lower())

class ObjectDetector:
    def __init__(self, model_name="efficientdet_lite0.tflite", num_threads=4, score_threshold=0.5, max_results=1, category_name_allowlist=["person"]):
        base_options = core.BaseOptions(file_name=model_name, use_coral=False, num_threads=num_threads)
        detection_options = processor.DetectionOptions(max_results=max_results, score_threshold=score_threshold, category_name_allowlist=category_name_allowlist)
        options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detections(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.detector.detect(vision.TensorImage.create_from_array(rgb_image)).detections

class Camera:
    def __init__(self, frame_width=1280, frame_height=720, camera_number=0):
        self.video_capture = cv2.VideoCapture(camera_number)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    def frame(self):
        _, frame = self.video_capture.read()
        return frame

class RealTimeObjectDetection:
    def __init__(self, frame_width=1280, frame_height=720, camera_number=0, model_name="efficientdet_lite0.tflite", num_threads=4, score_threshold=0.5, 
                 max_results=1, category_name_allowlist=["person"], folder_name="events", storage_capacity=21):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.led_rgb = LEDRGB(33, 35, 37)
        self.camera = Camera(frame_width, frame_height, camera_number)
        self.object_detector = ObjectDetector(model_name, num_threads, score_threshold, max_results, category_name_allowlist)
        self.storage_manager = StorageManager(folder_name, storage_capacity)
        self.storage_manager.supervise_folder_capacity()
        self.last_detection_timestamp = None
        self.frame = self.camera.frame()
        self.folder_name = folder_name
        self.output_path = None
        self.frame_buffer = []
        self.events = 0

    def guard(self, fps=24, max_detection_delay=30, event_check_interval=50):
        try:
            self.led_rgb.blue()
            while self.isOpened():
                detections, time_localtime = self.process_frame((0, 0, 255), 1, 2, cv2.FONT_HERSHEY_SIMPLEX)
                if detections:
                    if not self.frame_buffer:
                        hour, mins, day = time.strftime("%Hhr_%Mmin%Ssec_%B%d", time_localtime).split("_")
                        self.output_path = os.path.join(self.folder_name, day, hour, f"{hour}{mins}{day}.mp4")
                    self.last_detection_timestamp = time.time()
                    self.frame_buffer.append(self.frame)
                else:
                    if self.last_detection_timestamp and ((time.time() - self.last_detection_timestamp) >= max_detection_delay):
                        if len(self.frame_buffer) >= fps:
                            self.events += 1
                            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                            out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.frame_width, self.frame_height))
                            print(f"SAVE EVENT #{self.events}: {int(len(self.frame_buffer)/24)} seconds {self.output_path}")
                            for frame in self.frame_buffer:
                                out.write(frame)
                            out.release()
                            if self.events >= event_check_interval:
                                self.storage_manager.supervise_folder_capacity()
                        self.last_detection_timestamp = None
                        self.output_path = None
                        self.frame_buffer = []
                        self.led_rgb.blue()
                if len(self.frame_buffer) >= fps:
                    self.led_rgb.red()
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            self.close()

    def process_frame(self, color=(0, 0, 255), font_size=1, font_thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
        frame = self.camera.frame()
        time_localtime = time.localtime()
        detections = self.object_detector.detections(frame)
        for detection in detections:
            box = detection.bounding_box
            start_point = box.origin_x, box.origin_y
            end_point = box.origin_x+box.width, box.origin_y+box.height
            category_name = detection.categories[0].category_name
            text_position = (7+box.origin_x, 21+box.origin_y)
            cv2.rectangle(frame, start_point, end_point, color, font_thickness)
            cv2.putText(frame, category_name, text_position, font, font_size, color, font_thickness)
        cv2.putText(frame, time.strftime("%B%d/%Y %H:%M:%S", time_localtime), (21, 42), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness)
        self.frame = frame
        return detections, time_localtime

    def isOpened(self):
        return self.camera.video_capture.isOpened()
    
    def close(self):
        self.camera.video_capture.release()

class StorageManager:
    def __init__(self, events_folder="events", storage_capacity=21):
        self.events_folder = events_folder
        self.storage_capacity = storage_capacity

    @staticmethod
    def folder_size_gb(folder_path):
        total_size_bytes = 0
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size_bytes += os.path.getsize(file_path)
        return total_size_bytes / (1024 ** 3)
    
    @staticmethod
    def delete_folder(folder_path):
        folder_size = StorageManager.folder_size_gb(folder_path)
        shutil.rmtree(folder_path)
        print(f"STORAGE: '{folder_path}' was deleted (-{folder_size:.4f} GB)")
        return folder_size

    def supervise_folder_capacity(self):
        events_folder_size = StorageManager.folder_size_gb(self.events_folder)
        print(f"STORAGE: '{self.events_folder}' is ({events_folder_size:.4f} GB)")
        while events_folder_size > self.storage_capacity:
            folder_to_delete = os.path.join(self.events_folder, min(os.listdir(self.events_folder)))
            events_folder_size -= StorageManager.delete_folder(folder_to_delete)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-name", default="events", help="Name of the folder to store events (default: 'events')")
    parser.add_argument("--reset-events", action="store_true", help="Reset events folder")
    args = parser.parse_args()
    try:
        folder_name = args.folder_name
        if args.reset_events:
            StorageManager.delete_folder(folder_name)
            print(f"STORAGE: 'events' was deleted")
        os.makedirs("events", exist_ok=True)

        remote_camera = RealTimeObjectDetection(1280, 720, 0, "efficientdet_lite0.tflite", 4, 0.5, 3, ["person", "dog", "cat"], folder_name, 21)
        guard_thread = threading.Thread(target=remote_camera.guard, args=(24, 30, 50))
        guard_thread.start()

        app = Flask(__name__)

        def real_time_transmission():
            while remote_camera.isOpened():
                frame = cv2.imencode(".jpg", remote_camera.frame)[1].tobytes()
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        @app.route("/")
        def stream_video():
            return Response(real_time_transmission(), mimetype="multipart/x-mixed-replace; boundary=frame")
        
        app.run(host="0.0.0.0", port=80, threaded=True)   
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        remote_camera.close()
        GPIO.cleanup()
    finally:
        remote_camera.close()
        GPIO.cleanup()
