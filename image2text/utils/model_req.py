from io import BytesIO
from PIL import Image
import base64
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline


class Model:
    def __init__(
            self,
            processor="Salesforce/blip-image-captioning-base",
            model="Salesforce/blip-image-captioning-base",
            translator_task="translation_en_to_ru",
            translator_model="models/model_artifacts",
            translator_tokenizer="models/model_artifacts"):

        self.processor = BlipProcessor.from_pretrained(processor)
        self.model = BlipForConditionalGeneration.from_pretrained(model)
        self.translator = pipeline(translator_task, model=translator_model, tokenizer=translator_tokenizer)

    def model_request(self, image):
        processor, model, translator = self.processor, self.model, self.translator
        inputs = processor(images=image, return_tensors="pt")
        out_ids = model.generate(**inputs, max_length=64, num_beams=5, early_stopping=True)
        en_caption = processor.decode(out_ids[0], skip_special_tokens=True)

        # Перевод на русский
        ru_caption = translator(en_caption, max_length=128)[0]["translation_text"]

        return ru_caption

    @staticmethod
    def decode_base64_image(b64_string):
        if b64_string.startswith("data:image"):
            b64_string = b64_string.split(",")[1]

        try:
            image_data = base64.b64decode(b64_string)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image

        except Exception as e:
            print("image decoding error:", e)
            exit()
