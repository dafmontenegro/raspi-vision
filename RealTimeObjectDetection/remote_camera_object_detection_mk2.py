import os
import cv2
import time
import shutil
import argparse
import threading
import RPi.GPIO as GPIO
from flask import Flask, Response
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor

red_led = 33
green_led = 35
blue_led = 37

GPIO.setmode(GPIO.BOARD)
GPIO.setup(red_led, GPIO.OUT)
GPIO.setup(green_led, GPIO.OUT)
GPIO.setup(blue_led, GPIO.OUT)

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

def led_rgb(color):
    color = colors[color]
    GPIO.output(red_led, color[0])
    GPIO.output(green_led, color[1])
    GPIO.output(blue_led, color[2])

events = 0
event_frames = []
event_path = None
last_detection = None
app = Flask(__name__)
score_threshold = 0.5
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
base_options = core.BaseOptions(file_name="efficientdet_lite0.tflite", use_coral=False, num_threads=4)
detection_options = processor.DetectionOptions(max_results=1, score_threshold=score_threshold, category_name_allowlist=["person"])
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)
_, last_frame = cap.read()

def visualize(image, detection_result):
    global events
    global event_path
    global event_frames
    global last_detection
    time_localtime = time.localtime()
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)
        category_name = detection.categories[0].category_name
        text_location = (7 + bbox.origin_x, 21 + bbox.origin_y)
        cv2.putText(image, category_name, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, time.strftime("%B%d/%Y %H:%M:%S", time_localtime), (21, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if detection_result.detections:
        if not event_frames:
            hour, mins, day = time.strftime("%Hhr_%Mmin%Ssec_%B%d", time_localtime).split("_")
            event_path = os.path.join("events", day, hour, f"{hour}{mins}{day}.mp4")
        last_detection = time.time()
        event_frames.append(image)
    else:
        if last_detection and ((time.time() - last_detection) >= 30):
            if len(event_frames) >= 24:
                events += 1
                os.makedirs(os.path.dirname(event_path), exist_ok=True)
                out = cv2.VideoWriter(event_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (1280, 720))
                print(f"SAVE EVENT #{events} ({int(len(event_frames)/24)} secs): {event_path}")
                for event_frame in event_frames:
                    out.write(event_frame)
                out.release()
            last_detection = None
            event_path = None
            event_frames = []
            led_rgb("blue")
    return image

def main():
    led_rgb("blue")
    global last_frame
    global event_frames
    while cap.isOpened():
        _, image = cap.read()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_result = detector.detect(vision.TensorImage.create_from_array(rgb_image))
        last_frame = visualize(image, detection_result)
        if len(event_frames) >= 24:
            led_rgb("red")
    cap.release()

def generate_frames():
    while cap.isOpened():
        frame = cv2.imencode(".jpg", last_frame)[1].tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size / 1024 / 1024 / 1024

def delete_oldest_folder(folder_path):
    oldest_folder = min(os.listdir(folder_path))
    folder_to_delete = os.path.join(folder_path, oldest_folder)
    folder_deleted_size = get_folder_size(folder_to_delete)
    shutil.rmtree(folder_to_delete)
    print(f"STORAGE: '{folder_to_delete}' was deleted (-{folder_deleted_size:.4f} GB)")
    return folder_deleted_size

def folder_verification_thread(events_folder):
    while cap.isOpened():
        events_folder_size = get_folder_size(events_folder)
        print(f"STORAGE: '{events_folder}' is ({events_folder_size:.4f} GB)")
        while events_folder_size > 21:
            events_folder_size -= delete_oldest_folder(events_folder)
        time.sleep(1800)

@app.route("/")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

parser = argparse.ArgumentParser()
parser.add_argument("--reset-events", action="store_true", help="Reset events folder")
args = parser.parse_args()

if __name__ == "__main__":
    try:
        main_thread = threading.Thread(target=main)
        main_thread.start()
        if args.reset_events:
            shutil.rmtree("events")
            print(f"STORAGE: 'events' was deleted")
        os.makedirs("events", exist_ok=True)
        folder_thread = threading.Thread(target=folder_verification_thread, args=("events",))
        folder_thread.start()
        app.run(host="0.0.0.0", port=80, threaded=True)
    finally:
        GPIO.cleanup()
