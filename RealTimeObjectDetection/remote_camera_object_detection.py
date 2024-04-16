import cv2
import time
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

def visualize(image, detection_result, fps, datetime_str):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        cv2.rectangle(image, start_point, 3, (0, 0, 255), 3)
        #category = detection.categories[0]
        #result_text = f"{category.category_name} ({round(category.score, 2)})"
        #text_location = (12 + bbox.origin_x, 21 + bbox.origin_y)
        #cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1)
    #cv2.putText(image, f"{fps:.1f}", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, _TEXT_COLOR, 2)
    cv2.putText(image, datetime_str , (21, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image

def generate_frames():
    fps = 0.0
    counter = 0
    start_time = time.time()
    fps_avg_frame_count = 10
    while cap.isOpened():
        _, image = cap.read()
        counter += 1
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_result = detector.detect(vision.TensorImage.create_from_array(rgb_image))
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()
        datetime_str = time.strftime("%H:%M:%S %d/%m/%y", time.localtime())
        image = visualize(image, detection_result, fps, datetime_str)
        frame = cv2.imencode(".jpg", image)[1].tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    cv2.destroyAllWindows()
    cap.release()

@app.route("/")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
