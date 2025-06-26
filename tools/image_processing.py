from PIL import Image
import math
import random

from tools.upscale import Image_Upscaler

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

random_color = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
}


def get_random_color():
    return random.choice(list(random_color.values()))


def getImage(image_path: str):
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, get_random_color())
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            alpha = img.convert("L").split()[0]
            background.paste(img.convert("RGB"), mask=alpha)
        img = background
    return img


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

    def __while_upscale_image(self, image: Image, image_max_size=1024):
        new_image = image
        noise = 0
        key_int = 0
        while max(new_image.size) < image_max_size:
            scale = self.__math_image_scale(new_image, image_max_size)
            if scale == 4:
                noise = 2
            new_new_image = self.__upscale_image(new_image, noise=noise, scale=scale)
            new_image = new_new_image
            if max(new_image.size) < image_max_size:
                key_int += 1
                print(f"开始第{key_int+1}次采样，当前图片大小：{new_image.size}")
        return new_image

    def __math_image_scale(self, image: Image, image_max_size=1024):
        width, height = image.size
        max_side = max(width, height)

        if max_side < image_max_size:
            scale = math.ceil(image_max_size / max_side)
            if scale <= 2:
                scale = 2
            else:
                scale = 4
        return scale

    def __resize_image(self, image: Image, image_max_size=1024):

        if max(image.size) < image_max_size:
            image = self.__while_upscale_image(image, image_max_size)
        width, height = image.size

        if width > height:
            height = int(height * image_max_size / width)
            width = image_max_size
        else:
            width = int(width * image_max_size / height)
            height = image_max_size

        return image.resize((width, height))

    def __crop_to_ratio(self, image: Image) -> tuple:
        width, height = image.size
        ratios = RATIOS.copy()

        image_ratios = {}
        for name, (w_ratio, h_ratio) in ratios.items():
            target_ratio = w_ratio / h_ratio
            current_ratio = width / height
            ratio_error = abs(target_ratio - current_ratio) / current_ratio
            image_ratios[name] = ratio_error
        ratio_name = min(image_ratios, key=image_ratios.get)

        w_ratio, h_ratio = ratios[ratio_name]
        long_ratio = max(w_ratio, h_ratio)
        short_ratio = min(w_ratio, h_ratio)
        long_size = max(width, height)
        short_size = min(width, height)

        new_long_size = long_size - (long_size % long_ratio)
        new_short_size = short_ratio * (new_long_size // long_ratio)

        if width > height:
            new_width, new_height = new_long_size, new_short_size
        else:
            new_width, new_height = new_short_size, new_long_size

        MIN_SIZE = 100
        if new_width < MIN_SIZE or new_height < MIN_SIZE:
            scale = max(MIN_SIZE / new_width, MIN_SIZE / new_height)
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)

        new_image = Image.new("RGBA", (new_width, new_height), color=(255, 255, 255, 0))

        crop_width = min(width, new_width)
        crop_height = min(height, new_height)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        resized = cropped.resize((new_width, new_height), Image.LANCZOS)
        new_image.paste(resized, (0, 0))

        return new_image, ratio_name

    def pro_image(self, image_path: str, max_size=1024, ratio=True) -> tuple[Image, str]:
        image = getImage(image_path)
        try:
            image = self.__resize_image(image, max_size)
            if ratio:
                image, ratio_name = self.__crop_to_ratio(image)
            return image, ratio_name if ratio else None
        except Exception as e:
            print(e)
            return None, None
