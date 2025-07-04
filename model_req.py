import csv, time, json
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from huggingface_hub import snapshot_download
from functools import lru_cache


@lru_cache(maxsize=1)
def get_models():
    processor  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model      = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    translator = pipeline("translation_en_to_ru", model="model", tokenizer="C:/mine/models/opus-mt-en-ru")
    return processor, model, translator


def model_request(image):
    processor, model, translator = get_models()
    inputs = processor(images=image, return_tensors="pt")
    out_ids = model.generate(**inputs, max_length=64, num_beams=5, early_stopping=True)
    en_caption = processor.decode(out_ids[0], skip_special_tokens=True)

    # Перевод на русский
    ru_caption = translator(en_caption, max_length=128)[0]["translation_text"]

    return ru_caption
