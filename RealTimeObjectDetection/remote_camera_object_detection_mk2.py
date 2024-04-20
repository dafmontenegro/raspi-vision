import os
import cv2
import time
import shutil
import logging
import argparse
import threading
import RPi.GPIO as GPIO
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor
from flask import Flask, Response, render_template, send_file

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

    def __init__(self, red_led, green_led, blue_led):
        GPIO.setmode(GPIO.BCM)
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
        return lambda: self._set_color(color_name)

class LEDSRGB:
    def __init__(self, leds):
        self.leds = [LEDRGB(*led) for led in leds]
    
    def set_color(self, color_names):
        if isinstance(color_names, str):
            for i, led in enumerate(self.leds):
                getattr(led, color_names)()
        elif isinstance(color_names, list) and len(color_names) == len(self.leds):
            for (led, color) in zip(self.leds, color_names):
                getattr(led, color)()
    
    def __getattr__(self, color_name):
        return lambda: self.set_color(color_name)

class ObjectDetector:
    def __init__(self, model_name="efficientdet_lite0.tflite", num_threads=4, score_threshold=0.3, max_results=1, category_name_allowlist=["person"]):
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
    def __init__(self, frame_width=1280, frame_height=720, camera_number=0, model_name="efficientdet_lite0.tflite", num_threads=4, score_threshold=0.3, max_results=1, category_name_allowlist=["person"], 
                 folder_name="events", storage_capacity=32, led_pines=[(13, 19, 26), (21, 20, 16)], fps_frame_count= 30, safe_zone=((0, 0), (1280, 720))):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera = Camera(frame_width, frame_height, camera_number)
        self.frame = self.camera.frame()
        self.object_detector = ObjectDetector(model_name, num_threads, score_threshold, max_results, category_name_allowlist)
        self.folder_name = folder_name
        self.storage_manager = StorageManager(folder_name, storage_capacity)
        self.storage_manager.supervise_folder_capacity()
        self.leds_rgb = LEDSRGB(led_pines)
        self.safe_zone_start, self.safe_zone_end = safe_zone
        self.fps_frame_count = fps_frame_count
        self.last_detection_timestamp = None
        self.frame_buffer = []
        self.frame_times = []
        self.output = {}
        self.events = 0
        self.fps = 24

    def guard(self, min_video_duration=3, max_detection_delay=10, event_check_interval=10, safe_zone=False):
        try:
            self.leds_rgb.set_color(["off", "green"])
            while self.isOpened():
                security_breach, time_localtime = self.process_frame((0, 0, 255), 1, 2, cv2.FONT_HERSHEY_SIMPLEX, safe_zone)
                if security_breach:
                    if not self.frame_buffer:
                        self.output["file_name"] = time.strftime("%B%d_%Hhr_%Mmin%Ssec", time_localtime)
                        self.output["day"], self.output["hours"], self.output["mins"] = self.output["file_name"].split("_")
                        self.output["path"] = os.path.join(self.folder_name, self.output["day"], self.output["hours"], f"{self.output['file_name']}.mp4")
                    self.last_detection_timestamp = time.time()
                    self.frame_buffer.append(self.frame)
                else:
                    if self.last_detection_timestamp and ((time.time() - self.last_detection_timestamp) >= max_detection_delay):
                        if len(self.frame_buffer) >= self.fps*min_video_duration:
                            self.save_frame_buffer(self.output["path"], event_check_interval)
                        self.leds_rgb.set_color(["off", "green"])
                        self.last_detection_timestamp = None
                        self.frame_buffer = []
                        self.output = {}
                if len(self.frame_buffer) >= self.fps:
                    self.leds_rgb.red()
        except Exception as e:
            logging.error(e, exc_info=True)
            GPIO.cleanup()
            self.close()
            os._exit(0)
    
    def save_frame_buffer(self, path, event_check_interval=10):
        output_seconds = int(len(self.frame_buffer)/self.fps)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), self.fps, (self.frame_width, self.frame_height))
        logging.warning(f"EVENT: {output_seconds} seconds {path}")
        for frame in self.frame_buffer:
            out.write(frame)
        out.release()
        self.events += 1
        if self.events % event_check_interval == 0:
            storage_thread = threading.Thread(target=self.storage_manager.supervise_folder_capacity)
            storage_thread.start()
    
    def _safe_zone_invasion(self, rect_start, rect_end):
        if self.safe_zone_start[0] > rect_end[0] or self.safe_zone_end[0] < rect_start[0]:
            return False
        if self.safe_zone_start[1] > rect_end[1] or self.safe_zone_end[1] < rect_start[1]:
            return False
        return True

    def process_frame(self, color=(0, 0, 255), font_size=1, font_thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, safe_zone=False):
        security_breach = False
        start_time = time.time()
        frame = self.camera.frame()
        time_localtime = time.localtime()
        detections = self.object_detector.detections(frame)
        for detection in detections:
            box = detection.bounding_box
            rect_start = (box.origin_x, box.origin_y)
            rect_end = (box.origin_x+box.width, box.origin_y+box.height)
            category_name = detection.categories[0].category_name
            text_position = (7+box.origin_x, 21+box.origin_y)
            cv2.putText(frame, category_name, text_position, font, font_size, color, font_thickness)
            cv2.rectangle(frame, rect_start, rect_end, color, font_thickness)
            security_breach = self._safe_zone_invasion(rect_start, rect_end)
        cv2.putText(frame, time.strftime("%B%d/%Y %H:%M:%S", time_localtime), (21, 42), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness)
        if safe_zone:
            cv2.rectangle(frame, self.safe_zone_start, self.safe_zone_end, (0, 255, 255), font_thickness)
        self.frame = frame
        self.frame_times.append(time.time() - start_time)
        if self.fps_frame_count == len(self.frame_times):
            average_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = round(1/average_frame_time, 2)
            self.frame_times = []
        return security_breach, time_localtime

    def isOpened(self):
        return self.camera.video_capture.isOpened()
    
    def close(self):
        self.camera.video_capture.release()

class StorageManager:
    def __init__(self, events_folder="events", storage_capacity=32):
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
        logging.warning(f"STORAGE: '{folder_path}' was deleted (-{folder_size:.4f} GB)")
        return folder_size

    def supervise_folder_capacity(self):
        events_folder_size = StorageManager.folder_size_gb(self.events_folder)
        logging.info(f"STORAGE: '{self.events_folder}' is ({events_folder_size:.4f} GB)")
        while events_folder_size > self.storage_capacity:
            folder_to_delete = os.path.join(self.events_folder, min(os.listdir(self.events_folder)))
            events_folder_size -= StorageManager.delete_folder(folder_to_delete)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-name", default="events", help="Name of the folder to store events (default: 'events')")
    parser.add_argument("--log-file", default="logfile.log", help="Name of the log file (default: 'logfile.log')")
    parser.add_argument("--reset-events", action="store_true", help="Reset events folder")
    parser.add_argument("--reset-logs", action="store_true", help="Reset log file")
    args = parser.parse_args()
    try:
        log_file = args.log_file
        if args.reset_logs:
            with open(log_file, "w") as file:
                file.write(f"{log_file.upper()}\n")
        logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%B%d/%Y %H:%M:%S")

        folder_name = args.folder_name
        if args.reset_events:
            StorageManager.delete_folder(folder_name)
        os.makedirs("events", exist_ok=True)

        remote_camera = RealTimeObjectDetection(
            frame_width=1280,
            frame_height=720,
            camera_number=0,
            model_name="efficientdet_lite0.tflite",
            num_threads=4,
            score_threshold=0.3,
            max_results=3, 
            category_name_allowlist=["person", "umbrella"],
            folder_name=folder_name,
            storage_capacity=32,
            led_pines=[(13, 19, 26), (21, 20, 16)],
            fps_frame_count=30,
            safe_zone=((0, 360), (1280, 720))
        )

        guard_thread = threading.Thread(target=remote_camera.guard, kwargs={
            "min_video_duration": 1,
            "max_detection_delay": 10,
            "event_check_interval": 10,
            "safe_zone": True
        })
        guard_thread.start()

        app = Flask(__name__)

        def real_time_transmission(duration=300):
            start_time = time.time()
            time_seconds = int(time.time())
            while time_seconds - start_time < duration:
                if time_seconds % 2:
                    cv2.circle(remote_camera.frame, (1238, 21), 12, (0, 255, 0), -1)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + cv2.imencode(".jpg", remote_camera.frame)[1].tobytes() + b"\r\n")
                time_seconds = int(time.time())
            cv2.circle(remote_camera.frame, (1238, 21), 12, (0, 0, 255), -1)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + cv2.imencode(".jpg", remote_camera.frame)[1].tobytes() + b"\r\n")

        @app.route("/")
        def stream_video():
            return Response(real_time_transmission(), mimetype="multipart/x-mixed-replace; boundary=frame")
        
        @app.route("/logs/")
        def get_logs():
            with open(log_file, "r") as file:
                return Response(file.read(), mimetype="text/plain")
        
        @app.route("/events/")
        def get_events():
            days = []
            h1 = "EVENTS"
            for day in sorted(os.listdir(folder_name)):
                day_path = os.path.join(folder_name, day)
                day_info = {"date": day, "hours": []}
                for hour in sorted(os.listdir(day_path)):
                    hour_path = os.path.join(day_path, hour)
                    hour_info  = {"time": hour, "videos": []}
                    for video in sorted(os.listdir(hour_path)):
                        video_name = "".join(video.split("_")[1:]).replace(".mp4", "")
                        hour_info["videos"].append({"name": video_name, "path": video})
                    day_info["hours"].append(hour_info )
                days.append(day_info)
            if not days:
                h1 = "NO EVENTS AVAIBLE"
            return render_template("events.html", events=days, h1=h1)
        
        @app.route("/play/<path:video_path>")
        def get_video(video_path):
            video_path = os.path.join(folder_name, video_path)
            if os.path.exists(video_path):
                return send_file(video_path, mimetype="video/mp4")
            else:
                return "Video not found."

        app.run(host="0.0.0.0", port=80, threaded=True)   
    except Exception as e:
        logging.error(e, exc_info=True)
        remote_camera.close()
        GPIO.cleanup()
    finally:
        remote_camera.close()
        GPIO.cleanup()
