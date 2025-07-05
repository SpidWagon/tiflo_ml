from io import BytesIO
from PIL import Image
import base64
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

class Model:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    translator = pipeline("translation_en_to_ru", model="model", tokenizer="model")

model_blip = Model()

def model_request(image):
    processor, model, translator = model_blip.processor, model_blip.model, model_blip.translator
    inputs = processor(images=image, return_tensors="pt")
    out_ids = model.generate(**inputs, max_length=64, num_beams=5, early_stopping=True)
    en_caption = processor.decode(out_ids[0], skip_special_tokens=True)

    # Перевод на русский
    ru_caption = translator(en_caption, max_length=128)[0]["translation_text"]

    return ru_caption

def decode_base64_image(b64_string):
    if b64_string.startswith("data:image"):
        b64_string = b64_string.split(",")[1]

    try:
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        print(image)
        return image

    except Exception as e:
        print("Ошибка при декодировании:", e)
        exit()
