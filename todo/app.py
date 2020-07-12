from flask import Flask, render_template, request, jsonify
import sys
import random
import base64
import re

import identifiers.predict as pre

import identifiers.number_predict as npre
app = Flask(__name__)


@app.route('/')
def hello_world():
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
        # print(base64_data)
        # print("BYtes utf 8 start")
        base64_img_bytes = base64_data.encode('utf-8')
        # print(base64_img_bytes)
        with open("../images/numbers/aa_{}.png".format(a), "wb") as fh:
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            fh.write(decoded_image_data)

        # image_name = '../images/aa_{}.png'.format(a)
        # value = pre.predict_character(image_name)
        image_name = '../images/numbers/aa_{}.png'.format(a)
        value = npre.predict_number(image_name)
        print("--------------------from prediction----------------")
        print(value)
        return value
