from fastapi import FastAPI
import base64
from formats import *
from io import BytesIO
from PIL import Image

from model_req import model_request

app = FastAPI()

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



@app.post("/")
def root(req: Req):
    image = decode_base64_image(req.image)

    caption = model_request(image)

    return {
        "comment": caption,
        "timestamp": req.timestamp
    }

@app.get("/")
def prikol():
    return "КУДА ТЫ ЛЕЗЕШЬ??"

