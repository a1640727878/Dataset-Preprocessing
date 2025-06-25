from download_model import Upscaler_Downloader
import onnxruntime as ort
import numpy as np
from PIL import Image
import math


class Image_Upscaler:
    def __init__(self) -> None:
        self.upscaler_download = Upscaler_Downloader()
        self.default_model = self.upscaler_download.get_model(0, 2)

        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.providers = ["CPUExecutionProvider"]

        self._sessions_cache = {}
        self._model_path_cache = {}

    def __crop_image(self, image: Image) -> Image:
        mask = image.convert("L")
        pixels = mask.getdata()
        alpha = Image.new("L", mask.size)
        alpha_pixels = []

        for pixel in pixels:
            if pixel <= 60:
                alpha_pixels.append(0)
            else:
                alpha_pixels.append(255)

        alpha.putdata(alpha_pixels)

        new_image = Image.new("RGBA", image.size)
        new_image.paste(image)
        new_image.putalpha(alpha)

        bbox = new_image.getbbox()
        return image.crop(bbox)

    def __pro_image(self, image: Image):
        width, height = image.size
        new_size = max(width, height)
        new_size = (new_size // 32 + 1) * 32
        new_image = Image.new("RGB", (new_size, new_size), (0, 0, 0))
        new_image.paste(image, (0, 0))
        return new_image

    def __get_model_path(self, noise: int = 0, scale: int = 2):
        if (noise, scale) not in self._model_path_cache:
            self._model_path_cache[(noise, scale)] = self.upscaler_download.get_model(noise, scale)
        return self._model_path_cache[(noise, scale)]

    def get_model(self, noise: int = 0, scale: int = 2):
        model_path = self.__get_model_path(noise, scale)
        if not model_path:
            model_path = self.default_model
        if model_path not in self._sessions_cache:
            model = ort.InferenceSession(model_path, sess_options=self.session_options, providers=self.providers)
            input_name = model.get_inputs()[0].name
            self._sessions_cache[model_path] = (model, input_name)
        return self._sessions_cache[model_path]

    def __preprocess_tile(self, image: Image) -> tuple[np.ndarray, tuple[int, int]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_np = np.array(image).astype(np.float32) / 255.0
        image_transpose = np.transpose(image_np, (2, 0, 1))
        image_dims = np.expand_dims(image_transpose, axis=0)

        height, width = image.size
        pad_height = (32 - height % 32) % 32
        pad_width = (32 - width % 32) % 32
        image_dims = np.pad(image_dims, ((0, 0), (0, 0), (0, pad_height), (0, pad_width)), mode="constant")

        return image_dims, (height, width)

    def __postprocess_tile(self, output_np: np.ndarray, origin_size: tuple[int, int], scale: int):
        output_squeeze = np.squeeze(output_np, axis=0)
        output_transpose = np.transpose(output_squeeze, (1, 2, 0))
        output_clip = np.clip(output_transpose, 0.0, 1.0)

        origin_height, origin_width = origin_size
        target_height = origin_height * scale
        target_width = origin_width * scale

        crop_output = output_clip[:target_height, :target_width, :]
        image_array = (crop_output * 255).astype(np.uint8)

        return Image.fromarray(image_array, "RGB")

    def __upscale(self, image: Image, noise: int = 0, scale: int = 2) -> tuple[Image, tuple[int, int], tuple[int, int, int, int]]:
        input_data, (height, width) = self.__preprocess_tile(image)
        model, input_name = self.get_model(noise, scale)
        outputs_data = model.run([], {input_name: input_data})[0]
        result_image = self.__postprocess_tile(outputs_data, (height, width), scale)
        return result_image

    def __upscale_tile(self, image: Image, noise: int = 0, scale: int = 2):
        new_image = image.copy()
        width, height = new_image.size
        output_width = width * scale
        output_height = height * scale
        output_image = Image.new("RGB", (output_width, output_height))
        origin_stride = 32
        while min(width, height) > origin_stride:
            origin_stride += 32
        origin_stride -= 32
        if origin_stride > 512:
            origin_stride = 512
        stride = origin_stride - 32
        num_tiles_x = math.ceil((width - 32) / stride) if width > 32 else 1
        num_tiles_y = math.ceil((height - 32) / stride) if height > 32 else 1

        total_tiles = num_tiles_x * num_tiles_y
        processed_tiles = 0

        print(f"开始分块处理: {num_tiles_x} x {num_tiles_y} = {total_tiles} 块 (块大小 {origin_stride})")

        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                x_start = x * stride
                y_start = y * stride

                if width - x_start < origin_stride:
                    x_end = width
                    x_start = width - origin_stride
                else:
                    x_end = x_start + origin_stride
                if height - y_start < origin_stride:
                    y_end = height
                    y_start = height - origin_stride
                else:
                    y_end = y_start + origin_stride

                tile_image = new_image.crop((x_start, y_start, x_end, y_end))
                result_image = self.__upscale(tile_image, noise, scale)

                paste_x_start = x_start * scale
                paste_y_start = y_start * scale

                output_image.paste(result_image, (paste_x_start, paste_y_start))

                processed_tiles += 1
                print(f"已处理 {processed_tiles} / {total_tiles} 块", end="\r")

        return output_image

    def upscale_image(self, image: Image, noise: int = 0, scale: int = 2, test: bool = False) -> Image:
        width, height = image.size
        if max(width, height) > 32:
            result_image = self.__upscale_tile(image, noise, scale)
        else:
            new_image = self.__pro_image(image)
            result_image = self.__upscale(new_image, noise, scale)
        if test:
            result_image.save("./out/test_2.png")
        return self.__crop_image(result_image)
