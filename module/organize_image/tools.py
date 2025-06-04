from pathlib import Path
from PIL import Image
from ai_tools.image_upscaler import Upscaler

RATIOS = {
    "1_1": (1, 1),
    "1_2": (1, 2),
    "2_1": (2, 1),
    "2_3": (2, 3),
    "3_2": (3, 2),
    "3_4": (3, 4),
    "4_3": (4, 3),
    "3_7": (3, 7),
    "7_3": (7, 3),
    "9_16": (9, 16),
    "16_9": (16, 9),
}

UPSCALER = None


def get_upscaler():
    global UPSCALER
    if UPSCALER is None:
        UPSCALER = Upscaler()
    return UPSCALER


def _calculate_closest_ratio(width, height):

    closest_ratio = None
    min_difference = float("inf")

    for ratio_name, (ratio_w, ratio_h) in RATIOS.items():
        diff = abs(width * ratio_h - height * ratio_w)
        normalized_diff = diff / ((width * ratio_h * height * ratio_w) ** 0.5 + 1e-10)

        if normalized_diff < min_difference:
            min_difference = normalized_diff
            closest_ratio = ratio_name

    return closest_ratio


def _calculate_shrunk_image_size(width, height, image_size):
    max_side = max(width, height)
    if max_side <= image_size:
        return (width, height)

    scale_ratio = image_size / max_side

    if width == max_side:
        new_width = image_size
        new_height = round(height * scale_ratio)
    else:
        new_height = image_size
        new_width = round(width * scale_ratio)

    original_ratio = width / height
    new_ratio = new_width / new_height if new_height != 0 else 0

    if abs(original_ratio - new_ratio) > 0.001:
        if width == max_side:
            new_height_alt = new_height + 1 if new_height + 1 <= image_size else new_height - 1
            new_ratio_alt = new_width / new_height_alt
            if abs(original_ratio - new_ratio_alt) < abs(original_ratio - new_ratio):
                new_height = new_height_alt
        else:

            new_width_alt = new_width + 1 if new_width + 1 <= image_size else new_width - 1
            new_ratio_alt = new_width_alt / new_height
            if abs(original_ratio - new_ratio_alt) < abs(original_ratio - new_ratio):
                new_width = new_width_alt

    new_width = max(1, new_width)
    new_height = max(1, new_height)

    return (new_width, new_height)


def _calculate_cropped_size_by_ratio(width, height, ratio_name):
    if ratio_name not in RATIOS:
        raise ValueError(f"无效的比例名称: {ratio_name}")
    if width <= 0 or height <= 0:
        raise ValueError("宽度和高度必须为正数")

    ratio_w, ratio_h = RATIOS[ratio_name]
    img_ratio = width / height
    target_ratio = ratio_w / ratio_h

    if img_ratio > target_ratio:
        new_width = int(height * target_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / target_ratio)

    return new_width, new_height


def _organize_image(image: Image.Image, image_size: int):
    image_max_size = max(image.size)

    if image_size > image_max_size:
        if image_max_size * 1.8 >= image_size:
            image = get_upscaler().run_models([image], 2, 2)[0]
        elif image_max_size * 3.8 >= image_size:
            image = get_upscaler().run_models([image], 3, 4)[0]
        else:
            return False
    width, height = image.size
    image = image.copy().resize(_calculate_shrunk_image_size(width, height, image_size))
    width, height = image.size
    ratio_name = _calculate_closest_ratio(width, height)

    new_width, new_height = _calculate_cropped_size_by_ratio(width, height, ratio_name)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return ratio_name, image.crop((left, top, right, bottom)).resize((new_width, new_height))


def run(image_size: int, input_dir: str, output_dir: str):

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"输入路径 {input_path} 不存在")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_image_paths = [p for p in input_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    for image_path in input_image_paths:
        input_image = Image.open(str(image_path))
        data = _organize_image(input_image, image_size)
        if not data:
            continue
        ratio_name, image = data
        output_image_path = output_path / f"{image_path.stem}_{ratio_name}{image_path.suffix}"
        image.save(output_image_path)
        print(f"成功将规整后的图片保存至: {output_image_path}")

    print("所有规整任务已完成")
