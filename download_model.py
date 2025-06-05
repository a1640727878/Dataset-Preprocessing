from math import e
from huggingface_hub import hf_hub_download

import os
import shutil


class WDTagger_Downloader:
    def __init__(self, models: dict = {}):
        if models:
            self.models = models
        else:
            self.models = {
                "wd-eva02-large-tagger-v3": {
                    "repo_id": "SmilingWolf/wd-eva02-large-tagger-v3",
                    "model": "model.onnx",
                    "csv": "selected_tags.csv",
                },
                "wd-vit-large-tagger-v3": {
                    "repo_id": "SmilingWolf/wd-vit-large-tagger-v3",
                    "model": "model.onnx",
                    "csv": "selected_tags.csv",
                },
                "wd-vit-tagger-v3": {
                    "repo_id": "SmilingWolf/wd-vit-tagger-v3",
                    "model": "model.onnx",
                    "csv": "selected_tags.csv",
                }
            }

    def __get_models(self, model_name: str = "wd-eva02-large-tagger-v3") -> tuple[str, str]:
        models = self.models.get(model_name)
        if not models:
            raise ValueError(f"Model {model_name} not found.")
        model = models["model"]
        csv = models["csv"]
        return (model, csv)

    def __get_path(self, model_name: str = "wd-eva02-large-tagger-v3") -> tuple[str, str]:
        model, csv = self.__get_models(model_name)
        return (
            f"./data/models/{model_name}/{model}",
            f"./data/models/{model_name}/{csv}",
        )

    def __get_temp_path(self, model_name: str = "wd-eva02-large-tagger-v3") -> tuple[str, str]:
        model, csv = self.__get_models(model_name)
        return (
            f"./data/models_temp/{model_name}/{model}",
            f"./data/models_temp/{model_name}/{csv}",
        )

    def __if_exists(self, model_name: str = "wd-eva02-large-tagger-v3") -> tuple[bool, bool]:
        model, csv = self.__get_path(model_name)
        return (
            os.path.exists(model),
            os.path.exists(csv),
        )

    def get_model(self, model_name: str = "wd-eva02-large-tagger-v3") -> tuple[str, str]:
        model_bool, scv_bool = self.__if_exists(model_name)
        self.__download(model_name, model_bool, scv_bool)
        return self.__get_path(model_name)

    def __set_file(self, path: str, model_name: str = "wd-eva02-large-tagger-v3"):
        if os.path.exists(path) and os.path.isdir(path):
            dir_name = os.path.basename(path)
            to_path = f"./data/models/{model_name}"
            if not os.path.exists(to_path):
                os.makedirs(to_path)
            print(f"Copying {dir_name} to ./data/models/{model_name}/{dir_name}")
            shutil.copy2(f"{path}/{dir_name}", f"./data/models/{model_name}/{dir_name}")

    def __download(self, model_name: str = "wd-eva02-large-tagger-v3", dow_model: bool = False, dow_csv: bool = False) -> None:
        model, csv = self.__get_models(model_name)
        model_temp_path, csv_temp_path = self.__get_temp_path(model_name)
        repo_id = self.models[model_name]["repo_id"]
        if not dow_model:
            hf_hub_download(repo_id=repo_id, filename=model, local_dir=model_temp_path)
            self.__set_file(model_temp_path, model_name)
        if not dow_csv:
            hf_hub_download(repo_id=repo_id, filename=csv, local_dir=csv_temp_path)
            self.__set_file(csv_temp_path, model_name)
        temp_path = f"./data/models_temp/{model_name}"
        if os.path.exists(temp_path):
            shutil.rmtree(f"./data/models_temp/{model_name}")


class Upscaler_Downloader:
    def __init__(self):
        self.repo_id = "deepghs/waifu2x_onnx"
        self.hf_file_path = "20250502/onnx_models/swin_unet/art/"
        self.noise = [0, 1, 2, 3]
        self.scale = [1, 2, 4]

    def __get_models_name(self, noise: int = 0, scale: int = 1) -> str:
        if noise not in self.noise or scale not in self.scale:
            raise ValueError(f"Model {noise}_{scale}x not found.")
        if noise == 0 and scale == 1:
            return "noise{n}.onnx"
        elif noise == 0:
            return "scale{s}x.onnx"
        else:
            return "noise{n}_scale{s}x.onnx"

    def __get_path(self, noise: int = 0, scale: int = 1) -> str:
        return f"./data/models/waifu2x/{self.__get_models_name(noise, scale)}"

    def __get_temp_path(self, noise: int = 0, scale: int = 1) -> str:
        return f"./data/models_temp/waifu2x/{self.__get_models_name(noise, scale)}"

    def __if_exists(self, noise: int = 0, scale: int = 1) -> bool:
        return os.path.exists(self.__get_path(noise, scale))

    def get_model(self, noise: int = 0, scale: int = 1) -> str:
        if not self.__if_exists(noise, scale):
            self.__download(noise, scale)
        return self.__get_path(noise, scale)

    def __set_file(self, path: str, noise: int = 0, scale: int = 1) -> None:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.copyfile(path, self.__get_path(noise, scale))

    def __download(self, noise: int = 0, scale: int = 1) -> None:
        model_name = self.__get_models_name(noise, scale)
        model_temp_path = self.__get_temp_path(noise, scale)
        hf_file_path = self.hf_file_path + model_name
        hf_hub_download(repo_id=self.repo_id, filename=hf_file_path, local_dir=model_temp_path)
        self.__set_file(model_temp_path, noise, scale)
        shutil.rmtree(f"./data/models_temp/waifu2x")
