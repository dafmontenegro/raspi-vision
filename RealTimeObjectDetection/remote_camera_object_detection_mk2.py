import cv2
import time
import threading
from flask import Flask, Response
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor

app = Flask(__name__)
score_threshold = 0.6
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
base_options = core.BaseOptions(file_name="efficientdet_lite0.tflite", use_coral=False, num_threads=4)
detection_options = processor.DetectionOptions(max_results=1, score_threshold=score_threshold, category_name_allowlist=["person"])
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)
_, last_frame = cap.read()

def visualize(image, detection_result, datetime_str):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)
    cv2.putText(image, datetime_str , (21, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image

def generate_frames():
    while cap.isOpened():
        frame = cv2.imencode(".jpg", last_frame)[1].tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    cv2.destroyAllWindows()
    cap.release()

def main():
    global last_frame
    while cap.isOpened():
        _, image = cap.read()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_result = detector.detect(vision.TensorImage.create_from_array(rgb_image))
        datetime_str = time.strftime("%H:%M:%S %d/%m/%y", time.localtime())
        last_frame = visualize(image, detection_result, datetime_str)
    cv2.destroyAllWindows()
    cap.release()

@app.route("/")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    thread = threading.Thread(target=main)
    thread.start()
    app.run(host="0.0.0.0", port=80)
