from fastapi import FastAPI
from formats import *
from utils.model_req import decode_base64_image, model_request

app = FastAPI()


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

