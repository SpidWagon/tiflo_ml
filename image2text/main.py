from fastapi import FastAPI
from formats import *
from utils.model_req import decode_base64_image, Model
import uvicorn

app = FastAPI()

model_blip = Model()


@app.post("/")
def root(req: Req):
    comments = []

    images = req.images

    for image in images:
        image_decode = decode_base64_image(image)
        caption = model_blip.model_request(image_decode)
        comments.append(caption)



    return {
        "comments": comments
    }

@app.get("/")
def prikol():
    return "КУДА ТЫ ЛЕЗЕШЬ??"

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=3874, reload=False)