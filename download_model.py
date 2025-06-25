import os
import json
from pySmartDL import SmartDL


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
    def __init__(self) -> None:
        self.models = {
            "face": self.__get_models_data(
                "deepghs/anime_face_detection",
                "face_detect_v1.4_s/model.onnx",
                "./data/models/yolo/face.onnx",
                ["face"],
            ),
            "head": self.__get_models_data(
                "deepghs/anime_head_detection",
                "head_detect_v0.5_s_pruned/model.onnx",
                "./data/models/yolo/head.onnx",
                ["head"],
            ),
            "person": self.__get_models_data(
                "deepghs/anime_person_detection",
                "person_detect_v1.3_s/model.onnx",
                "./data/models/yolo/person.onnx",
                ["person"],
            ),
            "halfbody": self.__get_models_data(
                "deepghs/anime_halfbody_detection",
                "halfbody_detect_v1.0_s/model.onnx",
                "./data/models/yolo/halfbody.onnx",
                ["halfbody"],
            ),
            "eye": self.__get_models_data(
                "deepghs/anime_eye_detection",
                "eye_detect_v1.0_s/model.onnx",
                "./data/models/yolo/eye.onnx",
                ["eye"],
            ),
            "hand": self.__get_models_data(
                "deepghs/anime_hand_detection",
                "hand_detect_v1.0_s/model.onnx",
                "./data/models/yolo/hand.onnx",
                ["hand"],
            ),
            "censor": self.__get_models_data(
                "deepghs/anime_censor_detection",
                "censor_detect_v1.0_s/model.onnx",
                "./data/models/yolo/censor.onnx",
                ["nipple_f", "penis", "pussy"],
            ),
            "booru_yolo": self.__get_models_data(
                "deepghs/booru_yolo",
                "yolov8s_aa11/model.onnx",
                "./data/models/yolo/booru_yolo.onnx",
                ["head", "bust", "boob", "shld", "sideb", "belly", "nopan", "butt", "ass", "split", "sprd", "vsplt", "vsprd", "hip", "wing", "feral", "hdrago", "hpony", "hfox", "hrabb", "hcat", "hbear", "jacko", "jackx", "hhorse", "hbird"],
            ),
            "nudenet": self.__get_models_data(
                "deepghs/nudenet_onnx",
                "320n.onnx",
                "./data/models/yolo/nudenet.onnx",
                ["female genitalia covered", "face female", "buttocks exposed", "female breast exposed", "female genitalia exposed", "male breast exposed", "anus exposed", "feet exposed", "belly covered", "feet covered", "armpits covered", "armpits exposed", "face male", "belly exposed", "male genitalia exposed", "anus covered", "female breast covered", "buttocks covered"],
            ),
        }
        self.path_dir = "./data/models/yolo"
        self.models_data = Models_Data(self.path_dir)
        for model_type_name, model_data in self.models.items():
            self.models_data.add_model(model_data["repo_id"], model_type_name, model_data["repo_path"], output_file=model_data["to_path"])

    def __get_models_data(self, repo_id: str, repo_path: str, to_path: str, labels=[]):
        return {"repo_id": repo_id, "repo_path": repo_path, "to_path": to_path, "labels": labels}

    def get_model(self, model_type_name: str) -> tuple[str, list]:
        model = self.models_data.get_model(model_type_name)
        labels = self.models[model_type_name]["labels"]
        return (model, labels)
