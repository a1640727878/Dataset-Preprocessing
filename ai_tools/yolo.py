from .download_tools import *
from yolov8_onnx import DetectEngine
from PIL import Image
import numpy as np
from PIL import ImageDraw


class YoloData:
    def __init__(self, repo_id: str, model_path: str, tags: list[str]) -> None:
        self.repo_id = repo_id
        self.model_path = model_path
        self.tags = tags

    def download_yolo(self):
        repo_id_dir_name = self.repo_id.replace("/", "_")
        model_hf_data = huggingface_url_data(self.repo_id, self.model_path)
        output_path = f"./download_data/yolo/{repo_id_dir_name}/model.onnx"

        try:
            isdownload_model, model_path = download(model_hf_data, output_path=output_path)
            if not isdownload_model:
                print("模型下载失败，请检查网络连接或资源路径是否正确")
                return None
        except Exception as e:
            print(f"下载模型时发生错误: {str(e)}")
            return None

        return model_path

    def get_tags(self):
        return self.tags


Models = {
    "face": YoloData(
        "deepghs/anime_face_detection",
        "face_detect_v1.4_s/model.onnx",
        ["脸"],
    ),
    "head": YoloData(
        "deepghs/anime_head_detection",
        "head_detect_v0.5_s_pruned/model.onnx",
        ["头"],
    ),
    "halfbody": YoloData(
        "deepghs/anime_halfbody_detection",
        "halfbody_detect_v1.0_s/model.onnx",
        ["上半身"],
    ),
    "person": YoloData(
        "deepghs/anime_person_detection",
        "person_detect_v1.3_s/model.onnx",
        ["角色"],
    ),
    "censor": YoloData(
        "deepghs/anime_censor_detection",
        "censor_detect_v1.0_s/model.onnx",
        ["乳头", "棍", "洞"],
    ),
    "booru_yolo_aa": YoloData(
        "deepghs/booru_yolo",
        "yolov8s_aa11/model.onnx",
        ["头", "胸", "球", "侧球", "无遮侧球", "肚子", "无内裤", "屁股", "无遮屁股", "劈腿", "类劈腿", "色劈腿", "侧屁股", "翅膀", "兽躯", "龙头", "马头", "狐狸头", "兔头", "猫头", "熊头", "Jack-O动作", "色Jack-O动作", "马头", "鸟头"],
    ),
    "booru_yolo_pp": YoloData(
        "deepghs/booru_yolo",
        "yolov8s_pp13/model.onnx",
        ["棍", "洞", "棍_洞", "指+洞", "口+洞", "球+棍", "手+棍", "口+棍", "洞+洞"],
    ),
    "nudenet": YoloData(
        "deepghs/nudenet_onnx",
        "320n.onnx",
        ["遮洞", "女脸", "露臀", "露球", "露洞", "露胸肌", "露肛", "露脚", "遮肚", "遮脚", "遮腋", "露腋", "男脸", "露肚", "露棍", "遮肛", "遮球", "遮臀"],
    ),
}


class Yolo:

    def __init__(self) -> None:
        self._models_cache = {}
        self.one_tags = None

    def get_model_cache(self, source=None):
        if source not in Models:
            return None
        if source not in self._models_cache:
            path = Models[source].download_yolo()
            tags = Models[source].get_tags()
            run_detector = lambda image, conf_thres, iou_thres: DetectEngine(path, conf_thres=conf_thres, iou_thres=iou_thres)(np.array(image))
            self._models_cache[source] = (run_detector, tags)
        return self._models_cache.get(source)

    def run_models_all(self, image: Image.Image, source: str = "face", conf_thres=0.25, iou_thres=0.7):
        run_detector, tags = self.get_model_cache(source)
        detector_output = run_detector(image, conf_thres, iou_thres)
        data = [(np.round(box).astype(int).tolist(), float(conf), int(cls)) for box, conf, cls in zip(detector_output[0], detector_output[1], detector_output[2])]
        return data, tags

    def run_models(self, image: Image.Image, source: str = "", conf_thres=0.25, iou_thres=0.7):
        return self.run_models_all(image, source, conf_thres, iou_thres)


if __name__ == "__main__":
    yolo = Yolo()
    input_image_paths = [
        "input1.png",
        "input2.png",
        "input3.png",
        "input4.png",
        "input5.png",
    ]
    input_images = []
    for input_image_path in input_image_paths:
        input_images.append(Image.open(f"./z_input/{input_image_path}"))
    for idx, image in enumerate(input_images):
        output, tags = yolo.run_models(image, "booru_yolo_aa")
        print(output)
        draw = ImageDraw.Draw(image)
        for box, conf, cls in output:
            tag = tags[cls]
            draw.rectangle(box, outline="red", width=2)  # 在图片上画红色矩形框
            draw.text((box[0], box[1]), f"  {tag}  ", fill="red")  # 在边框左上角添加 tag 文本
        save_path = f"./z_output/output_{idx}.png"
        image.save(save_path)  # 保存画框后的图片
        print(f"已保存画框后的图片至: {save_path}")
