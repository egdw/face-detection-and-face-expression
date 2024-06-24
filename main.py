from PIL import Image

from model import FacialExpressionRecognition
from flask import Flask, request, jsonify
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1000 * 1000

model = FacialExpressionRecognition(device="mps")


@app.route("/predict", methods=["POST"])
def hello_world():
    pic = request.files['file']
    pil_image = Image.open(pic).convert('RGB')
    result = model.detect_face(pil_image)
    # respone = json.dumps(result)
    # return respone,200,{"Content-Type":"application/json"}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
