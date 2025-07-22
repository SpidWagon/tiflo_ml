import os

from fastapi import FastAPI
from formats import *
from utils.model_req import Model
import uvicorn

from models.model_download import download_model


app = FastAPI()

current_dir = os.getcwd()
model_artifacts_dir = os.path.join("models", "model_artifacts")

download_model(local_dir=os.path.join(current_dir, model_artifacts_dir))
model_blip = Model(
    translator_model=model_artifacts_dir,
    translator_tokenizer=model_artifacts_dir
)


@app.post("/")
def root(req: Req):

    images = req.images
    print("images aquired")

    caption = model_blip.model_request(images)
    print("captions done")

    return {
        "comments": caption
    }


@app.get("/")
def prikol():
    return "КУДА ТЫ ЛЕЗЕШЬ??"


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=3874, reload=False)