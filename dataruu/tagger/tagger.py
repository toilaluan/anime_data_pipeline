import huggingface_hub
import torch
import onnxruntime as rt
import os
import pandas as pd
import numpy as np
import PIL
from . import dbimutils

# HF_TOKEN = os.environ["HF_TOKEN"]
HF_TOKEN = ""
CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

class Tagger:
    def __init__(self):
        self.model = self.load_model(CONV_MODEL_REPO, MODEL_FILENAME)
        self.tag_names, self.rating_indexes, self.general_indexes, self.character_indexes = self.load_labels()

    def load_model(self, model_repo: str, model_filename: str) -> rt.InferenceSession:
        path = huggingface_hub.hf_hub_download(
            model_repo, model_filename, use_auth_token=HF_TOKEN
        )
        model = rt.InferenceSession(path, providers=["CUDAExecutionProvider"])
        return model

    def load_labels(self) -> list[str]:
        path = huggingface_hub.hf_hub_download(
            CONV_MODEL_REPO, LABEL_FILENAME, use_auth_token=HF_TOKEN
        )
        df = pd.read_csv(path)

        tag_names = df["name"].tolist()
        rating_indexes = list(np.where(df["category"] == 9)[0])
        general_indexes = list(np.where(df["category"] == 0)[0])
        character_indexes = list(np.where(df["category"] == 4)[0])
        return tag_names, rating_indexes, general_indexes, character_indexes

    def predict(
        self,
        image: PIL.Image.Image,
        general_threshold: float = 0.35,
        character_threshold: float = 0.8,
    ):
        rawimage = image
        _, height, width, _ = self.model.get_inputs()[0].shape

        # Alpha to white
        image = image.convert("RGBA")
        new_image = PIL.Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, probs[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]
        general_res = [x for x in general_names if x[1] > general_threshold]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]
        character_res = [x for x in character_names if x[1] > character_threshold]
        character_res = dict(character_res)

        b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
        a = (
            ", ".join(list(b.keys()))
            .replace("_", " ")
            .replace("(", "\(")
            .replace(")", "\)")
        )
        c = ", ".join(list(b.keys()))

        return rating, character_res, general_res

if __name__ == "__main__":
    tagger = Tagger()
    image = PIL.Image.open("test.jpg")
    rating, character_res, general_res = tagger.predict(image)
    print(rating, character_res, general_res)