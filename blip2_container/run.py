import sys, torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel

BASE_ID  = "Salesforce/blip2-opt-2.7b"
LORA_DIR = "model"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

def load():
    base  = Blip2ForConditionalGeneration.from_pretrained(
                BASE_ID, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(base, LORA_DIR)
    proc  = Blip2Processor.from_pretrained(BASE_ID)
    return model.eval(), proc

def main(img_path):
    model, proc = load()
    img  = Image.open(img_path).convert("RGB")
    inp  = proc(images=img, return_tensors="pt").to(DEVICE)
    out  = model.generate(**inp, max_new_tokens=32)
    text = proc.decode(out[0], skip_special_tokens=True)
    print(text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python run.py <image-path>")
    main(sys.argv[1])
