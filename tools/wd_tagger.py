from download_model import WDTagger_Downloader

import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image


class WDTagger:

    kaomojis = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]

    def __init__(self, model_name: str = "wd-eva02-large-tagger-v3"):
        self.wdtagger_dowbload = WDTagger_Downloader()
        self.model_path, self.model_csv_path = self.wdtagger_dowbload.get_model(model_name)
        # 加载标签
        self.__load_tags()

        # 加载模型
        self.model = rt.InferenceSession(self.model_path)

        _, self.height, self.width, _ = self.model.get_inputs()[0].shape
        self.target_size = self.height

    def __load_tags(self):
        tags_df = pd.read_csv(self.model_csv_path)
        name_series = tags_df["name"]
        name_series = name_series.map(lambda x: x.replace("_", " ") if x not in self.kaomojis else x)
        self.tag_names = name_series.tolist()

    def __preprocess_image(self, image_path: str):
        image = Image.open(image_path)
        max_dim = max(image.size)

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        offset = ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2)
        padded_image.paste(image, offset)

        padded_image = padded_image.resize((self.target_size, self.target_size), Image.BICUBIC) if max_dim != self.target_size else padded_image

        image_array = np.asarray(padded_image, dtype=np.float32)[:, :, ::-1]
        return np.expand_dims(image_array, axis=0)

    def predict(self, image_path: str):
        image_ndarray = self.__preprocess_image(image_path)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        preds = self.model.run([output_name], {input_name: image_ndarray})[0]
        preds_confidence = preds[0].astype(float)

        return list(zip(self.tag_names, preds_confidence))

    def predict_confidence(self, image_path: str, confidence_threshold: float = 0.3) -> list[str]:
        labels = self.predict(image_path)
        str_list = []
        for label, confidence in labels:
            if confidence >= confidence_threshold:
                str_list.append(str(label))
        return str_list
