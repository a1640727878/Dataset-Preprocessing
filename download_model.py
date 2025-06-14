from hmac import new
import os
import json
from typing import final
from pySmartDL import SmartDL
import requests


def get_download_url(repo_id: str, repo_model_file: str, commit_name: str = "main") -> str:
    url = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    final_url = f"{url}/{repo_id}/resolve/{commit_name}/{repo_model_file}?download=true"
    print(f"Downloading from {final_url}")
    return final_url


def download_file(url: str, output_dir: str, threads: int = 16) -> bool:
    file_obj = SmartDL(
        url,
        output_dir,
        threads=threads,
        timeout=10,
        progress_bar=True,
        verify=True,
    )
    if file_obj.isFinished():
        return True
    try:
        file_obj.start()
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False
    return file_obj.isSuccessful()


class Models_Data:
    def __init__(self, output_dir: str = "./data/models"):
        self.models = {}
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def add_model(self, repo_id: str, model_name: str, model_file: str, commit_name: str = "main", output_file: str = None) -> None:
        if not output_file:
            output_file = f"{self.output_dir}/{model_name}/{model_file}"
        self.models[model_name] = (repo_id, commit_name, model_file, output_file)

    def get_model(self, model_name: str) -> str:
        if self.download_model(model_name):
            return self.models[model_name][3]
        return None

    def download_model(self, model_name: str, threads: int = 16) -> bool:
        repo_id, commit_name, model_file, output_file = self.models.get(model_name)
        if not os.path.exists(output_file):
            url = get_download_url(repo_id, model_file, commit_name)
            return download_file(url, output_file, threads)
        return True


class WDTagger_Downloader:
    def __init__(self):
        self.models_data = Models_Data("./data/models/wd-tagger")
        self.models = {}
        self.add_models("SmilingWolf/wd-eva02-large-tagger-v3", "wd-eva02-large-tagger-v3")
        self.add_models("SmilingWolf/wd-vit-large-tagger-v3", "wd-vit-large-tagger-v3")
        self.add_models("SmilingWolf/wd-vit-tagger-v3", "wd-vit-tagger-v3")
        for model_type_name, model in self.models.items():
            self.__add_models_data(model_type_name, model)

    def add_models(self, repo_id: str, model_type_name: str, commit_name: str = "main") -> None:
        self.models[model_type_name] = [repo_id, commit_name]

    def __add_models_data(self, model_type_name: str, moedl: tuple[str, str]) -> None:
        repo_id, commit_name = moedl
        csv_type_name = f"{model_type_name}_csv"
        model_name = "model.onnx"
        csv_name = "selected_tags.csv"
        self.models_data.add_model(repo_id, model_type_name, model_name, commit_name)
        self.models_data.add_model(repo_id, csv_type_name, csv_name, commit_name)

    def get_model(self, model_type_name: str) -> tuple[str, str]:
        model_path = self.models_data.get_model(model_type_name)
        csv_path = self.models_data.get_model(f"{model_type_name}_csv")
        return (model_path, csv_path)


class Upscaler_Downloader:
    def __init__(self):
        self.repo_id = "deepghs/waifu2x_onnx"
        self.hf_dir_path = "20250502/onnx_models/swin_unet/art"
        self.path_dir = "./data/models/waifu2x"
        self.models_data = Models_Data(self.path_dir)
        self.noise = [0, 1, 2, 3]
        self.scale = [1, 2, 4]
        self.models = {}
        self.__add_models()
        for model_name, model in self.models.items():
            repo_id, commit_name, model_name = model
            self.models_data.add_model(repo_id, model_name, f"{self.hf_dir_path}/{model_name}", commit_name, f"{self.path_dir}/{model_name}")

    def __get_model_name(self, noise: int = 0, scale: int = 1) -> str:
        if noise not in self.noise:
            noise = 0
        if scale not in self.scale:
            scale = 1
        if noise == 0 and scale == 1:
            return f"noise{noise}.onnx"
        elif noise == 0:
            return f"scale{scale}x.onnx"
        else:
            return f"noise{noise}_scale{scale}x.onnx"

    def __add_models(self, repo_id: str = None, commit_name: str = "main") -> None:
        if not repo_id:
            repo_id = self.repo_id
        for noise in self.noise:
            for scale in self.scale:
                model_name = self.__get_model_name(noise, scale)
                self.models[model_name] = (repo_id, commit_name, f"{model_name}")

    def get_model(self, noise: int = 0, scale: int = 1) -> str:
        model_name = self.__get_model_name(noise, scale)
        return self.models_data.get_model(model_name)


class Yolo_Downloader:
    def __init__(self):
        self.models = {
            "face": {
                "repo_id": "deepghs/anime_face_detection",
                "models": ["face_detect_v1.4_s"],
            },
            "head": {
                "repo_id": "deepghs/anime_head_detection",
                "models": ["head_detect_v0.5_s_pruned"],
            },
            "person": {
                "repo_id": "deepghs/anime_person_detection",
                "models": ["person_detect_v1.3_s"],
            },
            "halfbody": {
                "repo_id": "deepghs/anime_halfbody_detection",
                "models": ["halfbody_detect_v1.0_s"],
            },
            "eye": {
                "repo_id": "deepghs/anime_eye_detection",
                "models": ["eye_detect_v1.0_s"],
            },
            "hand": {
                "repo_id": "deepghs/anime_hand_detection",
                "models": ["hand_detect_v1.0_s"],
            },
            "censor": {
                "repo_id": "deepghs/anime_censor_detection",
                "models": ["censor_detect_v1.0_s"],
            },
            "booru_yolo": {
                "repo_id": "deepghs/booru_yolo",
                "models": ["yolov8s_aa11"],
                "ext": ("model.onnx", "meta.json"),
            },
        }
        self.path_dir = "./data/models/yolo"
        self.models_data = Models_Data(self.path_dir)
        for model_type_name, model in self.models.items():
            repo_id = model["repo_id"]
            models = model["models"]
            for model_name in models:
                model_labels_name = f"{model_name}_labels"
                model_file_name = "model.onnx"
                model_labels_file_name = "labels.json"
                if "ext" in model:
                    model_file_name, model_labels_file_name = model["ext"]
                self.models_data.add_model(repo_id, model_name, f"{model_name}/{model_file_name}", output_file=f"{self.path_dir}/{model_type_name}/{model_name}/{model_file_name}")
                self.models_data.add_model(repo_id, model_labels_name, f"{model_name}/{model_labels_file_name}", output_file=f"{self.path_dir}/{model_type_name}/{model_labels_name}/{model_labels_file_name}")
        self.models["nudenet"] = {
            "repo_id": "deepghs/nudenet",
            "models": ["Default"],
        }

        self.models_data.add_model("deepghs/nudenet", "nudenet", "320n.onnx", output_file=f"{self.path_dir}/nudenet/model.onnx")

    def __get_model_path(self, model_type_name: str) -> tuple[str, str]:
        model_name = self.models[model_type_name]["models"][0]
        model_labels_name = f"{model_name}_labels"
        model = self.models_data.get_model(model_name)
        model_label = self.models_data.get_model(model_labels_name)
        return (model, model_label)

    def __parse_json(self, json_path: str) -> dict:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __save_json(self, json_path: str, data: dict or list) -> None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_model(self, model_type_name: str) -> tuple[str, str]:
        if model_type_name == "booru_yolo":
            new_label = f"{self.path_dir}/booru_yolo/labels.json"
            model, model_label = self.__get_model_path(model_type_name)
            if not os.path.exists(new_label):
                meta_json = self.__parse_json(model_label)
                new_json: list = meta_json["labels"]
                self.__save_json(new_label, new_json)
            return (model, new_label)
        elif model_type_name == "nudenet":
            label = f"{self.path_dir}/nudenet/labels.json"
            if not os.path.exists(label):
                new_json = ["FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED"]
                self.__save_json(label, new_json)
            return (self.models_data.get_model(model_type_name), label)
        else:
            return self.__get_model_path(model_type_name)


class Classification_Downloader:

    def __init__(self):
        self.models = {
            "classification": ["mobilenetv3_v1.5_dist"],
            "completeness": ["mobilenetv3_v2.2_dist"],
            "rating": ["mobilenetv3_sce_dist"],
            "character_sex": ["caformer_s36_v1"],
            "portrait_type": ["mobilenetv3_v0_dist"],
            "is_anime": ["mobilenetv3_v1.2_dist"],
        }
        self.types = {
            "classification": "deepghs/anime_classification",
            "completeness": "deepghs/anime_completeness",
            "rating": "deepghs/anime_rating",
            "character_sex": "deepghs/anime_ch_sex",
            "portrait_type": "deepghs/anime_portrait",
            "is_anime": "deepghs/anime_real_cls",
        }
        self.models_dir = "./data/models/classification"
        self.models_data = Models_Data(self.models_dir)
        for type_name, models in self.models.items():
            for model in models:
                model_meta_name = f"model_meta"
                self.models_data.add_model(self.types[type_name], model, f"{model}/model.onnx")
                self.models_data.add_model(self.types[type_name], model_meta_name, f"{model}/meta.json")
