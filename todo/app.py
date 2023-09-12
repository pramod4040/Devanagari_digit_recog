import os
from flask import Flask, render_template, request, jsonify
import random
import base64
import re

import identifiers.EnsembleModel as model
# import pickle

app = Flask(__name__)

FLASH_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(FLASH_ROOT, '..')




@app.route('/')
def hello_world():
    """ Print Hello world as the response body.  """
    a = __name__
    # k = pre.predit(a)
    value = {"status": 200, "message": "All Okay"}
    return jsonify(value)


@app.route('/try-it-out')
def hello_world_index():
    """ Print Hello world as the response body.  """
    a = __name__
    # k = pre.predit(a)
    return render_template("index.html")


@app.route('/encode', methods=['POST'])
def encode():
    if request.method == 'POST':
        base64_img = request.json['image']
        a = random.randint(100, 900)
        base64_data = re.sub('^data:image/.+;base64,', '', base64_img)
        base64_img_bytes = base64_data.encode('utf-8')

        current_path = os.getcwd()
        imgName = f"nepchar_{a}.png"
        imagePath = os.path.join(current_path, 'images', 'test_data', imgName)
        # print(os.path())
        # "imagePath"/aa_{}.png".format(a)
        with open(imagePath, "wb") as fh:
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            fh.write(decoded_image_data)

        image_path = os.path.join(ROOT_DIR, "images", "test_data", imgName)

        # loaded_model = pickle.load(open('devnagari-best-model.pkl', 'rb'))
        # predicted_label = loaded_model.predict(imagePath)

        predicted_label = model.magic(image_path)
        value = {"predicted": predicted_label, "percentage": "100%"}
        print("--------------------from prediction----------------")
        print(value)
        return jsonify(value)
