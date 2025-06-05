from PIL import Image
from tools.upscale import Image_Upscaler

RATIOS = {
    "1_1": 1.0,  # 1:1
    "2_3": 2 / 3,  # 2:3
    "3_2": 3 / 2,  # 3:2
    "4_3": 4 / 3,  # 4:3
    "3_4": 3 / 4,  # 3:4
    "16_9": 16 / 9,  # 16:9
    "9_16": 9 / 16,  # 9:16
}


class Image_Processing:
    def __init__(self):
        self._upscaler = None

    @property
    def upscaler(self) -> Image_Upscaler:
        if self._upscaler is None:
            self._upscaler = Image_Upscaler()
        return self._upscaler

    def __upscale_image(self, image: Image, scale=2, noise=0):
        if scale not in [0, 2, 4]:
            scale = 2
        if noise not in [0, 1, 2, 3]:
            noise = 0
        return self.upscaler.upscale_image(image, scale=scale, noise=noise)

    def __resize_image(self, image: Image, image_max_size=1024):
        # 获取原始尺寸
        width, height = image.size
        max_side = max(width, height)

        # 如果原图最长边小于目标尺寸，需要先放大再缩小
        if max_side < image_max_size:
            # 计算需要的放大倍数
            scale = 2 if image_max_size / max_side <= 2 else 4
            # 先放大
            image = self.__upscale_image(image, scale=scale)
            width, height = image.size

        # 等比例缩放图片
        if width > height:
            height = int(height * image_max_size / width)
            width = image_max_size
        else:
            width = int(width * image_max_size / height)
            height = image_max_size

        return image.resize((width, height))

    def __crop_to_ratio(self, image: Image) -> tuple:
        width, height = image.size
        current_ratio = width / height
        ratios = RATIOS.copy()

        # 找到最接近的标准比例
        closest_ratio_name = min(ratios.keys(), key=lambda x: abs(ratios[x] - current_ratio))
        target_ratio = ratios[closest_ratio_name]

        # 计算裁剪尺寸
        if current_ratio > target_ratio:
            # 需要裁剪宽度
            new_width = int(height * target_ratio)
            crop_left = (width - new_width) // 2
            cropped_image = image.crop((crop_left, 0, crop_left + new_width, height))
        else:
            # 需要裁剪高度
            new_height = int(width / target_ratio)
            crop_top = (height - new_height) // 2
            cropped_image = image.crop((0, crop_top, width, crop_top + new_height))

        return cropped_image, closest_ratio_name

    def pro_image(self, image_path: str, max_size=1024, ratio=True) -> tuple[Image, str]:
        image = Image.open(image_path)
        if ratio:
            image, ratio_name = self.__crop_to_ratio(image)
        image = self.__resize_image(image, max_size)
        return image, ratio_name if ratio else None
