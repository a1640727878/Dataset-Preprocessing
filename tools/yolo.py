import cv2
import json
from yolov8_onnx import DetectEngine


class YOLO:
    def __init__(self, model: tuple[str, str], confidence_threshold: float = 0.25, iou_threshold: float = 0.7) -> None:
        self.model_path, self.label_path = model

        self.classes = self.__parse_json(self.label_path)
        self.max_size = 640

        self.detector = DetectEngine(self.model_path, image_size=self.max_size, conf_thres=confidence_threshold, iou_thres=iou_threshold)

    def __parse_json(self, json_path: str) -> list:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def detect(self, image_path: str):
        image = cv2.imread(image_path)
        output = self.detector(image)
        results = []
        for box, score, class_id in zip(output[0], output[1], output[2]):
            x1, y1, x2, y2 = map(int, box)
            results.append({"box": [x1, y1, x2, y2], "score": score, "class_id": class_id, "class_name": self.classes[class_id]})
        return results


from download_model import Yolo_Downloader


class ImageCropper:
    def __init__(self, model_name: str, confidence_threshold: float = 0.5, iou_threshold: float = 0.5):
        self.model_name = model_name
        self.downloader = Yolo_Downloader()
        self.yolo = YOLO(self.downloader.get_model(self.model_name), confidence_threshold, iou_threshold)

    def get_image_results(self, image_path: str):
        results = self.yolo.detect(image_path)
        return results
