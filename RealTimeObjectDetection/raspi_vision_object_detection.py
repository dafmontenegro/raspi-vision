
import cv2
import time
from tflite_support.task import core
from tflite_support.task import vision
from tflite_support.task import processor

score_threshold = 0.7
_TEXT_COLOR = (0, 0, 255)
_FONT_SIZE = 1

def visualize(image, detection_result, fps):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)
        category = detection.categories[0]
        result_text = f"{category.category_name} ({round(category.score, 2)})"
        text_location = (12 + bbox.origin_x, 21 + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, 1)
    cv2.putText(image, f"{fps:.1f}", (21, 21), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, 1)
    return image

def run(model, width, height):
    fps = 0.0
    counter = 0
    start_time = time.time()
    fps_avg_frame_count = 10
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    base_options = core.BaseOptions(file_name=model, use_coral=False, num_threads=4)
    detection_options = processor.DetectionOptions(max_results=1, score_threshold=score_threshold, category_name_allowlist=["person"])
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
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
        image = visualize(image, detection_result, fps)
        cv2.imshow("Raspberry Pi Object Detector", image)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    run("efficientdet_lite0.tflite", 1280, 720)
