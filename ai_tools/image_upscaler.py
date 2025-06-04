from .download_tools import *
import onnxruntime as ort
from PIL import Image
import numpy as np


class Upscaler_Models:
    def __init__(self):
        self.repo_id = "deepghs/waifu2x_onnx"
        self.repo_file_dir = "20250502/onnx_models/swin_unet/art"

        self.noise = [0, 1, 2, 3]
        self.scale = [1, 2, 4]

    def get_model_name(self, noise: int = 0, scale: int = 1) -> str:
        noise = noise if noise in self.noise else self.noise[0]
        scale = scale if scale in self.scale else self.scale[0]

        if noise == 0:
            return f"noise{noise}.onnx" if scale == 1 else f"scale{scale}x.onnx"
        return f"noise{noise}_scale{scale}x.onnx"

    def get_model_path(self, noise: int = 0, scale: int = 1) -> str | None:
        try:
            model_name = self.get_model_name(noise, scale)
            model_repo_path = f"{self.repo_file_dir}/{model_name}"
            hf_data = huggingface_url_data(self.repo_id, model_repo_path)
            isDownload, path = download(hf_data, output_path=f"./download_data/waifu2x/{model_name}")
            return path if isDownload else None
        except Exception as e:
            print(f"Error getting model path: {e}")
            return None


class Upscaler:

    def __init__(self):
        self.models = Upscaler_Models()
        self._sessions_cache = {}

    def get_model_sessions(self, noise: int = 0, scale: int = 1) -> ort.InferenceSession | None:
        if (noise, scale) not in self._sessions_cache:
            model_path = self.models.get_model_path(noise, scale)
            if model_path is None:
                return None
            try:
                sessions = ort.InferenceSession(model_path)
                self._sessions_cache[(noise, scale)] = sessions
            except Exception as e:
                print(f"Error creating ONNX inference session: {e}")
                return None
        return self._sessions_cache.get((noise, scale))

    def image2Array(self, images: list[Image.Image]) -> np.ndarray:
        tiles_float = []
        for image in images:
            tile_np = np.array(image, dtype=np.uint8)
            tile_transposed = np.transpose(tile_np, (2, 0, 1))
            tile_float = tile_transposed.astype(np.float32) / 255.0
            tiles_float.append(tile_float)
        batch_tensor = np.stack(tiles_float, axis=0)
        return batch_tensor

    def array2Image(self, batch_output) -> Image.Image:
        tile_images = []
        for tile_array in batch_output:
            tile_denorm = tile_array * 255.0
            tile_clamped = np.clip(tile_denorm, 0, 255)
            tile_uint8 = tile_clamped.astype(np.uint8)
            tile_transposed = np.transpose(tile_uint8, (1, 2, 0))
            tile_image = Image.fromarray(tile_transposed)
            tile_images.append(tile_image)

        return tile_images

    def crop_image_to_multiple_of_four(self, image: Image) -> Image.Image:
        width, height = image.size
        new_width = width // 4 * 4
        new_height = height // 4 * 4
        return image.crop((0, 0, new_width, new_height))

    def run_models(self, images: list[Image.Image], noise: int = 0, scale: int = 1) -> list[Image.Image] | None:
        input_images = []
        for image in images:
            image = image.convert("RGB")
            inference_session = self.get_model_sessions(noise, scale)
            if inference_session is None:
                return None
            processed_image = self.crop_image_to_multiple_of_four(image)
            input_images.append(processed_image)
        input_tensor = self.image2Array([processed_image])
        try:
            model_outputs = inference_session.run(["y"], {"x": input_tensor})
        except Exception as e:
            print(f"处理图像时发生错误: {e}")
            return None
        output_image = []
        for image in self.array2Image(model_outputs[0]):
            output_image.append(image)

        return output_image


if __name__ == "__main__":
    image_upscaler = Upscaler()
    input_image_paths = [
        "input1.png",
        "input2.png",
        "input3.png",
        "input4.png",
        "input5.png",
    ]
    for index, input_image_path in enumerate(input_image_paths):
        input_image = Image.open(f"./z_input/{input_image_path}")
        print(f"正在对图片 ./z_input/{input_image_path} {input_image.size} 进行超分辨率处理...")
        upscaled_image = image_upscaler.run_models([input_image], noise=2, scale=2)[0]
        output_image_path = f"./z_output/output{index}.png"
        print(f"图片已成功放大, 分辨率 {upscaled_image.size}，并保存至 {output_image_path}")
        upscaled_image.save(output_image_path)
