from .download_tools import *
import onnxruntime as ort
from PIL import Image
import numpy as np
import pandas as pd
import json

SmilingWolf_models = [
    "wd-eva02-large-tagger-v3",
    "wd-vit-large-tagger-v3",
    "wd-v1-4-swinv2-tagger-v2",
    "wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3",
    "wd-v1-4-moat-tagger-v2",
    "wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-vit-tagger-v2",
    "wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-convnext-tagger",
    "wd-v1-4-vit-tagger",
]

models_source = ["SmilingWolf", "Pixai" "Z3D_E621", "Camie"]


class Tagger_Models:

    def _get_model(self, repo_id: str, model_path: str, tags_path: str):
        repo_id_dir_name = repo_id.replace("/", "_")
        if not model_path.endswith(".onnx"):
            print("仅支持 ONNX 模型，请确保模型路径以 '.onnx' 结尾")
            return None
        model_hf_data = huggingface_url_data(repo_id, model_path)
        tags_hf_data = huggingface_url_data(repo_id, tags_path)

        try:
            isdownload_model, model_path = download(model_hf_data, output_path=f"./download_data/tagger/{repo_id_dir_name}/model.onnx")
            if not isdownload_model:
                print("模型下载失败，请检查网络连接或资源路径是否正确")
                return None
            isdownload_tags, tags_path = download(tags_hf_data, output_path=f"./download_data/tagger/{repo_id_dir_name}/selected_tags.csv")
            if not isdownload_tags:
                print("标签文件下载失败，请检查网络连接或资源路径是否正确")
                return None
        except Exception as e:
            print(f"下载模型时发生错误: {str(e)}")
            return None

        return (model_path, tags_path)

    # https://huggingface.co/SmilingWolf
    def get_SmilingWolf_model(self, model_name: str = "wd-eva02-large-tagger-v3"):
        return self._get_model(f"SmilingWolf/{model_name}", "model.onnx", "selected_tags.csv")

    # https://huggingface.co/toynya/Z3D-E621-Convnext/
    def get_Z3D_E621_model(self):
        return self._get_model("toynya/Z3D-E621-Convnext", "model.onnx", "tags-selected.csv")

    # https://huggingface.co/Camais03
    # https://huggingface.co/deepghs/camie_tagger_onnx
    def get_Camie_model(self):
        return self._get_model("deepghs/camie_tagger_onnx", "initial/model.onnx", "initial/selected_tags.csv")

    # https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx
    def get_Pixai_model(self):
        return self._get_model("deepghs/pixai-tagger-v0.9-onnx", "model.onnx", "tags-selected.csv")

    def get_model(self, source=None, name=None):
        # 设置默认源为 SmilingWolf
        source = "SmilingWolf" if not source or source not in models_source else source
        name = "wd-eva02-large-tagger-v3" if not name or name not in SmilingWolf_models else name
        model_getters = {
            "SmilingWolf": lambda: self.get_SmilingWolf_model(name),
            "Z3D_E621": self.get_Z3D_E621_model,
            "Camie": self.get_Camie_model,
            "Pixai": self.get_Pixai_model,
        }

        try:
            result = model_getters[source]()
            if result is None:
                return None
            return (source, *result)
        except Exception as e:
            print(f"获取 {source} 模型时发生错误: {str(e)}")
            return None


class Image2Tagger:
    def __init__(self):
        self.models = Tagger_Models()
        self._sessions_cache = {}
        self.kaomojis = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]

    def image2Array(self, images: list[Image.Image], shape, source="SmilingWolf"):
        if source == "Camie":
            return self.image2Array_Camie(images, shape)
        _, shape_height, shape_width, _ = shape
        np_arrays = []
        for image in images:
            image = image.convert("RGB").resize((shape_height, shape_width))
            np_array_BGR = np.array(image, dtype=np.uint8)[:, :, ::-1]
            np_array_f32 = np_array_BGR.astype(np.float32)
            np_arrays.append(np_array_f32)
        return np.stack(np_arrays, axis=0)

    def image2Array_Camie(self, images: list[Image.Image], shape):
        _, _, shape_height, shape_width = shape

        np_arrays = []
        for image in images:
            image = image.convert("RGB").resize((shape_width, shape_height))
            np_array = np.array(image, dtype=np.uint8)[:, :, ::-1].transpose((2, 0, 1))
            np_array_f32 = np_array.astype(np.float32)
            np_arrays.append(np_array_f32)
        return np.stack(np_arrays, axis=0)

    def image2Array_Z3D_E621(self, image: Image.Image, shape):
        _, shape_height, shape_width, _ = shape
        image = image.convert("RGB").resize((shape_width, shape_height))
        np_array = np.array(image, dtype=np.uint8)[:, :, ::-1]  # 转换为 BGR 格式
        np_array_f32 = np_array.astype(np.float32)
        return np.expand_dims(np_array_f32, axis=0)

    def run_sessions_Z3D_E621(self, sessions, images: list[Image.Image]):
        sessions_input = sessions.get_inputs()[0]
        sessions_output = sessions.get_outputs()[-1]
        outputs = []
        for image in images:
            output = sessions.run([sessions_output.name], {sessions_input.name: self.image2Array_Z3D_E621(image, sessions_input.shape)})[0][0]
            outputs.append(output)
        return outputs

    def get_tag_list(self, tags_path):
        tag_pd = pd.read_csv(tags_path)
        return list(zip(tag_pd["name"].map(lambda x: str(x).replace("_", " ") if str(x) not in self.kaomojis else str(x)).tolist(), tag_pd["category"].tolist()))

    def get_model_sessions(self, model_source: str = "SmilingWolf", model_name: str = "wd-eva02-large-tagger-v3"):
        model_source = "SmilingWolf" if not model_source or model_source not in models_source else model_source
        model_name = "wd-eva02-large-tagger-v3" if not model_name or model_name not in SmilingWolf_models else model_name
        key = f"SmilingWolf_{model_name}" if model_source == "SmilingWolf" else model_source
        if key not in self._sessions_cache:
            source, model_path, tags_path = self.models.get_model(model_source, model_name)
            if model_path is None or tags_path is None:
                return None
            try:
                sessions = ort.InferenceSession(model_path)
                sessions.enable_custom_ops = False
                if model_source == "Z3D_E621":
                    run_sessions = lambda images: self.run_sessions_Z3D_E621(sessions, images)
                else:
                    sessions_input = sessions.get_inputs()[0]
                    sessions_output = sessions.get_outputs()[-1]
                    run_sessions = lambda images: sessions.run(
                        [sessions_output.name],
                        {sessions_input.name: self.image2Array(images, sessions_input.shape, source)},
                    )[0]
                tag_list = self.get_tag_list(tags_path)
                self._sessions_cache[key] = (run_sessions, tag_list)
            except Exception as e:
                print(f"Error creating ONNX inference session: {e}")
                return None
        return self._sessions_cache.get(key)

    def run_models(self, images: list[Image.Image], model_source: str = "SmilingWolf", model_name: str = "wd-eva02-large-tagger-v3"):
        run_sessions, tag_list = self.get_model_sessions(model_source, model_name)

        try:
            model_outputs = run_sessions(images)
        except Exception as e:
            print(f"生成图片标签时发送错误: {e}")
            return None

        output_tagger = []

        output_tagger = [[(name, category, float(f"{model_output[idx]:.10f}")) for idx, (name, category) in enumerate(tag_list)] for model_output in model_outputs]
        return output_tagger


if __name__ == "__main__":
    image2tagger = Image2Tagger()
    input_image_paths = [
        "input1.png",
        "input2.png",
        "input3.png",
    ]
    input_images = []
    for input_image_path in input_image_paths:
        input_images.append(Image.open(f"./z_input/{input_image_path}"))
    output_taggers = image2tagger.run_models(input_images, "Z3D_E621")
    for index, tagger in enumerate(output_taggers):
        output_json_path = f"./z_output/output{index}.json"
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(tagger, f, ensure_ascii=False, indent=4)
            print(f"标签已成功保存至 {output_json_path}")
        except Exception as e:
            print(f"保存标签到 {output_json_path} 时出错: {e}")
