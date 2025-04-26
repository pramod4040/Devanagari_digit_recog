import os
from flask import Flask, render_template, request, jsonify
import random
import base64
import re
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# from ..identifiers.EnsembleModel import EnsembleModelCharacter as model

import identifiers.EnsembleModel as model
from identifiers.predict import predict_character, models_and_weights, get_model_details

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
    # a = __name__
    # k = pre.predit(a)
    return render_template("index.html")


@app.route('/model-list')
def model_list():
    """ Print Hello world as the response body.  """
    modelDetails = models_and_weights()
    value = {"status": 200, "modelDetails": modelDetails}
    return jsonify(value)


@app.route('/encode', methods=['POST'])
def encode():
    try:
        if request.method == 'POST':
            base64_img = request.json['image']
            a = random.randint(100, 900)
            base64_data = re.sub('^data:image/.+;base64,', '', base64_img)
            base64_img_bytes = base64_data.encode('utf-8')

            userReqModelName = request.json['modelName']

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

            # predicted_label = model.magic(image_path)
            print(image_path)

            # weight_path = os.path.join(
            #     os.getcwd(), 
            #     "identifiers", 
            #     "trained_model", 
            #     "nepali_char_moe_model.1.weights.h5"
            # )

# ------ -          Model enhancedMoeModel------
            # weight_path = os.path.join(
            #     os.getcwd(), 
            #     "trainModel", 
            #     "model_checkpoints", 
            #     "moe_optimized_model-23-0.9473.weights.h5"
            # )

            # result = predict_character(image_path, "enhancedMoeModel", weightPath=weight_path)
# ------ -          Model enhancedMoeModel------

            weight_path = os.path.join(
                os.getcwd(), 
                "trainModel", 
                "model_checkpoints", 
                "MoeModelV2_moe_optimized_model-3-expert-20-0.9737.weights.h5"
            )

            print("--------------------weight path is ----------------")
            print(weight_path)

            # img = Image.open(image_path).resize((32, 32)).convert("RGB")
            # img_array = np.array(img)

            # img = Image.fromarray(img_array)
            # img.show()

            modelInfo = get_model_details(userReqModelName)
            
            result = predict_character(image_path, model_type=modelInfo['modelName'], weightPath=modelInfo['weightPath'])
            #  result = predict_character(image_path)
            print(f"Predicted: {result['class']} with {result['confidence']:.2%} confidence")
            
            value = {
                "predicted": result['class'], 
                "percentage": f"{result['confidence']:.2%}"
            }
            print("--------------------from prediction----------------")
            print(value)
            return jsonify(value)

    except Exception as e:
        print(e)
