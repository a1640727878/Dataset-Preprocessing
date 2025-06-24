import json
from mode_utils import WDTagger, ImageProcessing, Json2TxtProcessing, Txt2JsonProcessing, YoloCroppr


class Default_Config:
    def __init__(self):
        self.in_path = "./in"
        self.out_path = "./out"

    def get_image_pro_config(self, config: dict):
        config["input_dir"] = config.get("input_dir", self.in_path)
        config["output_dir"] = config.get("output_dir", self.out_path)
        config["image_size"] = config.get("image_size", 1024)
        config["ratio"] = config.get("ratio", True)
        return config

    def get_wd_tagger_config(self, config: dict):
        config["input_dir"] = config.get("input_dir", self.in_path)
        config["output_dir"] = config.get("output_dir", self.out_path)
        config["confidence_threshold"] = config.get("confidence_threshold", 0.3)
        config["thread_count"] = config.get("thread_count", 1)
        config["wd_model"] = config.get("wd_model", "wd-eva02-large-tagger-v3")
        return config

    def get_json2txt_config(self, config: dict):
        config["input_dir"] = config.get("input_dir", self.in_path)
        config["output_dir"] = config.get("output_dir", self.out_path)
        config["processing_py"] = config.get("processing_py", "./default_pro_json.py")
        return config

    def get_txt2json_config(self, config: dict):
        config["input_dir"] = config.get("input_dir", self.in_path)
        config["output_dir"] = config.get("output_dir", self.out_path)
        return config

    def get_yolo_croppr_config(self, config: dict):
        config["input_dir"] = config.get("input_dir", self.in_path)
        config["output_dir"] = config.get("output_dir", self.out_path)
        config["confidence_threshold"] = config.get("confidence_threshold", 0.35)
        config["iou_threshold"] = config.get("iou_threshold", 0.7)
        config["cropers_names"] = config.get("cropers_names", ["person", "halfbody", "head", "face"])
        return config

    def get_full_config(self, json_path: str) -> list:
        json_data = []
        with open(json_path, "r") as f:
            json_data = json.load(f)
        if not isinstance(json_data, list):
            return []
        for item in json_data:
            if "mode" not in item:
                continue
            if item["mode"] == "image_pro":
                item = self.get_image_pro_config(item)
            elif item["mode"] == "wd_tagger":
                item = self.get_wd_tagger_config(item)
            elif item["mode"] == "json2txt":
                item = self.get_json2txt_config(item)
            elif item["mode"] == "txt2json":
                item = self.get_txt2json_config(item)
            elif item["mode"] == "yolo_croppr":
                item = self.get_yolo_croppr_config(item)
        return json_data


def parse_args():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--json_config", type=str, default="./default_config.json", help="配置文件路径")
    return parser


def main():
    args = parse_args()
    json_config = args.json_config

    config = Default_Config()
    config_data = config.get_full_config(json_config)
    if len(config_data) == 0:
        print("配置文件错误")
        return

    for item in config_data:
        if item["mode"] == "image_pro":
            image_pro = ImageProcessing(item)
            image_pro.run()
        elif item["mode"] == "wd_tagger":
            wd_tagger = WDTagger(item)
            wd_tagger.run()
        elif item["mode"] == "json2txt":
            json2txt = Json2TxtProcessing(item)
            json2txt.run()
        elif item["mode"] == "txt2json":
            txt2json = Txt2JsonProcessing(item)
            txt2json.run()
        elif item["mode"] == "yolo_croppr":
            yolo_croppr = YoloCroppr(item)
            yolo_croppr.run()
