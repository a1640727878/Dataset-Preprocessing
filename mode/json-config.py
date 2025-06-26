import json
import sys
import argparse
import subprocess

from mode_utils import WDTagger, ImageProcessing, Json2TxtProcessing, Txt2JsonProcessing, YoloCroppr


class Default_Config:
    def __init__(self):
        self.in_path = "./in"
        self.out_path = "./out"

    def get_image_pro_config(self, config: dict) -> tuple:
        input_dir = config.get("input_dir", self.in_path)
        output_dir = config.get("output_dir", self.out_path)
        image_size = config.get("image_size", 1024)
        ratio = config.get("ratio", True)
        return input_dir, output_dir, image_size, ratio

    def get_wd_tagger_config(self, config: dict) -> tuple:
        input_dir = config.get("input_dir", self.in_path)
        output_dir = config.get("output_dir", self.out_path)
        confidence_threshold = config.get("confidence_threshold", 0.3)
        thread_count = config.get("thread_count", 1)
        wd_model = config.get("wd_model", "wd-eva02-large-tagger-v3")
        return input_dir, output_dir, confidence_threshold, thread_count, wd_model

    def get_json2txt_config(self, config: dict) -> tuple:
        input_dir = config.get("input_dir", self.in_path)
        output_dir = config.get("output_dir", self.out_path)
        processing_py = config.get("processing_py", "./default_pro_json.py")
        return input_dir, output_dir, processing_py

    def get_txt2json_config(self, config: dict) -> tuple:
        input_dir = config.get("input_dir", self.in_path)
        output_dir = config.get("output_dir", self.out_path)
        return input_dir, output_dir

    def get_yolo_croppr_config(self, config: dict) -> tuple:
        input_dir = config.get("input_dir", self.in_path)
        output_dir = config.get("output_dir", self.out_path)
        confidence_threshold = config.get("confidence_threshold", 0.35)
        iou_threshold = config.get("iou_threshold", 0.7)
        cropers_names = config.get("cropers_names", ["person", "halfbody", "head", "face"])
        return input_dir, output_dir, confidence_threshold, iou_threshold, cropers_names

    def get_run_py_config(self, config: dict) -> str or None:
        py_path = config.get("py_path", None)
        return py_path

    def get_full_config(self, json_path: str) -> list:
        json_data = []
        with open(json_path, "r") as f:
            json_data = json.load(f)
        if not isinstance(json_data, list):
            return []
        return json_data


def parse_args():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--json_config", type=str, default="./default_config.json", help="配置文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    json_config = args.json_config

    config = Default_Config()
    config_data = config.get_full_config(json_config)
    if len(config_data) == 0:
        print("配置文件错误")
        return

    print(config_data)

    for item in config_data:
        if item["mode"] == "image_pro":
            input_dir, output_dir, image_size, ratio = config.get_image_pro_config(item)
            image_pro = ImageProcessing(input_dir, output_dir, image_size, ratio)
            image_pro.run()
        elif item["mode"] == "wd_tagger":
            input_dir, output_dir, confidence_threshold, thread_count, wd_model = config.get_wd_tagger_config(item)
            wd_tagger = WDTagger(wd_model, (input_dir, output_dir), confidence_threshold, thread_count)
            wd_tagger.run()
        elif item["mode"] == "json2txt":
            input_dir, output_dir, processing_py = config.get_json2txt_config(item)
            json2txt = Json2TxtProcessing(input_dir, output_dir, processing_py)
            json2txt.run()
        elif item["mode"] == "txt2json":
            input_dir, output_dir = config.get_txt2json_config(item)
            txt2json = Txt2JsonProcessing(input_dir, output_dir)
            txt2json.run()
        elif item["mode"] == "yolo_croppr":
            input_dir, output_dir, confidence_threshold, iou_threshold, cropers_names = config.get_yolo_croppr_config(item)
            yolo_croppr = YoloCroppr(input_dir, output_dir, cropers_names, confidence_threshold, iou_threshold)
            yolo_croppr.run()
        elif item["mode"] == "run_py":
            py_path = config.get_run_py_config(item)
            if py_path is None:
                continue
            subprocess.run([sys.executable, py_path])


if __name__ == "__main__":
    main()
