from flask import Flask, render_template, request
from PIL import Image
from model_req import *

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        data = request.files['image']
        image = Image.open(data.stream).convert("RGB")
        image.save('pomogite.jpg')
        result = model_request(image)
        return result

    if request.method == 'GET':
        return 'КУДА ТЫ ЛЕЗЕШЬ???????'

app.run(debug=False)