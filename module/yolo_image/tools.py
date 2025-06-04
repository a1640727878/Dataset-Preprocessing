from pathlib import Path
from ai_tools.yolo import Yolo
from PIL import Image


YOLO = None


tag_name = {
    "脸(特写)": "face",
    "头": "head",
    "上半身": "halfbody",
    "角色": "person",
    "头+": "head_p",
    "胸腔": "chest",
    "腹部": "belly",
    "臀部": "hips",
}

_TAGS_CACHE = list(tag_name.keys())


def get_tags():
    return _TAGS_CACHE


def get_tags_name(zh_tag: str):
    return tag_name.get(zh_tag, "None")


def get_yolo():
    global YOLO
    YOLO = YOLO or Yolo()
    return YOLO


tag_models = {
    "脸(特写)": ("face", [0]),
    "头": ("head", [0]),
    "上半身": ("halfbody", [0]),
    "角色": ("person", [0]),
    "胸腔": ("booru_yolo_aa", [1, 2, 3, 4]),
    "腹部": ("booru_yolo_aa", [5, 6]),
    "臀部": ("booru_yolo_aa", [7, 8, 9, 10, 11, 12, 13]),
    "头+": ("booru_yolo_aa", [0, 16, 17, 18, 19, 20, 21, 24, 25]),
}


def yolo_data_to_models(yolo_data: list):
    models = []
    for tag in yolo_data:
        tag_name = get_tags_name(tag)
        models.append((tag_name, tag_models[tag]))
    return models


def run(yolo_data: list, input_dir: str, output_dir: str):

    models = yolo_data_to_models(yolo_data)

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"输入路径 {input_path} 不存在")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_image_paths = [p for p in input_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    for image_path in input_image_paths:
        models_temp = {}
        input_image = Image.open(str(image_path))
        for tag, (model_name, index_list) in models:
            if model_name not in models_temp:
                outputs, _ = get_yolo().run_models(image=input_image, source=model_name)
                models_temp[model_name] = outputs
            for idx, (box, conf, cls) in enumerate(models_temp[model_name]):
                if cls in index_list:
                    cropped_image = input_image.crop(box)
                    output_image_path = output_path / f"{image_path.stem}_{tag}_{idx+1}{image_path.suffix}"
                    cropped_image.save(output_image_path)
                    print(f"成功将YOLO裁剪后的图片保存至: {output_image_path}")

    print("所有YOLO裁剪任务已完成")
