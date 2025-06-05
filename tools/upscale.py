from turtle import width
from download_model import Upscaler_Downloader

import onnxruntime as ort
import numpy as np
import math
from typing import Tuple
from PIL import Image


class Image_Upscaler:
    def __init__(self):
        self.upscaler_download = Upscaler_Downloader()
        self.default_model = self.upscaler_download.get_model(0, 2)

        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.providers = []
        self.providers.append("CPUExecutionProvider")

        self._sessions_cache = {}

    def get_model(self, noise: int = 0, scale: int = 1) -> ort.InferenceSession:
        model_path = self.upscaler_download.get_model(noise, scale)
        if not model_path:
            model_path = self.default_model
        if model_path not in self._sessions_cache:
            self._sessions_cache[model_path] = self.__get_session(model_path)
        return self._sessions_cache[model_path]

    def __get_session(self, model_path: str) -> ort.InferenceSession:
        return ort.InferenceSession(model_path, sess_options=self.session_options, providers=self.providers)

    def __pad_to_multiple(self, img_np: np.ndarray, divisor: int = 32) -> Tuple[np.ndarray, Tuple[int, int]]:
        """将图像填充到指定倍数的尺寸"""
        h, w = img_np.shape[-2:]  # 获取 H, W
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        padded_img = np.pad(img_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        original_shape = (h, w)
        return padded_img, original_shape

    def __preprocess_tile(self, tile_image: Image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        预处理单个图像块 (NCHW, float32)，并进行填充。
        返回填充后的图像块和原始形状。
        """
        if tile_image.mode != "RGB":
            tile_image = tile_image.convert("RGB")

        img_np = np.array(tile_image).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)

        # 对单个块进行填充
        padded_img, original_shape = self.__pad_to_multiple(img_np, 32)
        return padded_img, original_shape

    def __postprocess_tile(self, output_np: np.ndarray, original_shape: Tuple[int, int], scale: int) -> Image:
        output_np = np.squeeze(output_np, axis=0)
        output_np = np.transpose(output_np, (1, 2, 0))
        output_np = np.clip(output_np, 0.0, 1.0)

        original_h, original_w = original_shape
        target_h = original_h * scale
        target_w = original_w * scale

        cropped_output = output_np[:target_h, :target_w, :]
        img_array = (cropped_output * 255.0).astype(np.uint8)
        return Image.fromarray(img_array, "RGB")

    def __upscale(self, input_name: str, image: Image, session: ort.InferenceSession, scale: int) -> Image:
        input_data, original_shape = self.__preprocess_tile(image)
        try:
            outputs = session.run(None, {input_name: input_data})
            output_data = outputs[0]
        except Exception as e:
            raise RuntimeError(f"ONNX 模型推理失败")
        result_image = self.__postprocess_tile(output_data, original_shape, scale)
        return result_image

    def __upscale_tile(
        self,
        input_name: str,
        image: Image,
        session: ort.InferenceSession,
        scale: int,
        tile_size: int = 250,
        tile_overlap: int = 32,
        tile_min_size: int = 350,
    ) -> Image:
        width, height = image.size
        output_width = width * scale
        output_height = height * scale
        output_image = Image.new("RGB", (output_width, output_height))
        stride = tile_size - tile_overlap

        if stride <= 0:
            raise ValueError(f"tile_size ({tile_size}) 必须大于 tile_overlap ({tile_overlap})")

        num_tiles_x = math.ceil((width - tile_overlap) / stride) if width > tile_overlap else 1
        num_tiles_y = math.ceil((height - tile_overlap) / stride) if height > tile_overlap else 1

        total_tiles = num_tiles_x * num_tiles_y
        processed_tiles = 0

        print(f"开始分块处理: {num_tiles_x} x {num_tiles_y} = {total_tiles} 块 (阈值 {tile_min_size})")

        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # 计算当前块的坐标 (左上角)
                x_start = x * stride
                y_start = y * stride
                # 计算当前块的坐标 (右下角)，确保不超过图像边界
                x_end = min(x_start + tile_size, width)
                y_end = min(y_start + tile_size, height)

                # 提取块
                tile = image.crop((x_start, y_start, x_end, y_end))

                # 放大块
                upscaled_tile = self.__upscale(input_name, tile, session, scale)

                # 计算粘贴位置 (考虑重叠和缩放)
                paste_x_start = x_start * scale
                paste_y_start = y_start * scale

                # 只粘贴非重叠部分，或者根据需要处理重叠区域（简化：直接粘贴）
                output_image.paste(upscaled_tile, (paste_x_start, paste_y_start))

                processed_tiles += 1
                print(f"已处理块: {processed_tiles}/{total_tiles}", end="\r")

        return output_image

    def upscale_image(
        self,
        image: Image,
        noise: int = 0,
        scale: int = 1,
        tile_size: int = 250,
        tile_overlap: int = 32,
        tile_min_size: int = 350,
    ) -> Image:
        model = self.get_model(noise, scale)
        input_name = model.get_inputs()[0].name
        width, height = image.size
        max_dim = max(width, height)

        use_tiling = scale > 1 and max_dim >= tile_min_size

        if not use_tiling:
            return self.__upscale(input_name, image, model, scale)
        else:
            return self.__upscale_tile(input_name, image, model, scale, tile_size, tile_overlap, tile_min_size)
            # 计算需要的图像块数量
