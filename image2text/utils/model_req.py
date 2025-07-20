from io import BytesIO
from PIL import Image
import base64
from transformers import AutoProcessor, pipeline
import torch
from utils.custom_model import Blip2ForConditionalSeqGeneration


PROCESSOR = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
MODEL = Blip2ForConditionalSeqGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
TRANSLATOR = pipeline("translation_en_to_ru", model="models/model_artifacts", tokenizer="models/model_artifacts")


def decode_base64_image(b64_string):
    if b64_string.startswith("data:image"):
        b64_string = b64_string.split(",")[1]

    try:
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image

    except Exception as e:
        print("Ошибка при декодировании:", e)
        exit()


class Model:
    def __init__(self):
        self.processor = PROCESSOR
        self.model = MODEL
        self.translator = TRANSLATOR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)


    def model_request(self, images):
        processor, model, translator = self.processor, self.model, self.translator

        inputs = []

        for el in images:
            image = decode_base64_image(el)
            inputik = processor(images=image, return_tensors="pt").to(self.device, torch.float16)["pixel_values"]
            inputs.append(inputik)

        out_ids = (model.generate_for_list(**inputs, max_length=64, num_beams=5, early_stopping=True))
        en_caption = processor.decode(out_ids[0], skip_special_tokens=True)

        # Перевод на русский
        ru_caption = translator(en_caption, max_length=128)[0]["translation_text"]

        return ru_caption