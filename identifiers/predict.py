import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
from skimage import io
import cv2
import numpy as np
from PIL import Image
import random


def prepare_image(raw_image):
    print(raw_image)
    image_in_numpy = io.imread(raw_image)

    # print(image_in_numpy.shape)

    #convert into grayscale image
    image_in_numpy = cv2.cvtColor(image_in_numpy, cv2.COLOR_RGB2GRAY)
    # image_in_numpy = image_in_numpy / 255
    # print(image_in_numpy)

    averaged_image = np.zeros((1024,), dtype='uint8')
    # print(averaged_image)
    for y in range(0,32):
        for x in range(0,32):
            mean = 0
            for h in range(0,8):
                for k in range(0,8):
                    mean += image_in_numpy[y * 8 + h, x * 8 + k]
        mean = mean / 64
        averaged_image[y * 32 + x] = mean
    
    # averaged_image = np.expand_dims(averaged_image, axis=0)
    averaged_image = averaged_image.reshape((32,32))
    # print(averaged_image.shape)
    gg = cv2.bitwise_not(averaged_image)
    return gg


def load_model():
    path_for_json = '/home/pranil/learning/machine_learning/devanagari_character/identifiers/trained_model/acc_97_Ver_1/model_97acc .json'
    path_for_weights = '/home/pranil/learning/machine_learning/devanagari_character/identifiers/trained_model/acc_97_Ver_1/model_97acc_weights.h5'
    
    with open(path_for_json) as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        json_file.close()
    
    loaded_model.load_weights(path_for_weights)
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return loaded_model


def map_index_with_result(result_index):
    r_indexes = [i for i in range(0,46)]

    result_coll = ['ka','kha','ga','gha','ŋa','cha','chha','ja','jha','ña','ta','tha','da','dha','ṇa','ṭa','ṭha','ḍa','ḍha','na','pa','pha','ba','bha','ma','ya','ra','la','wa','sa','sa','sa','ha','chhya','ṭra','gya', '1', '2', '3', '4', '5', '6', '7', '8', '9','0']
    
    dict_result =  dict(zip(r_indexes, result_coll))

    return dict_result[result_index]



def predict_character(raw_image):
    a = random.randint(100,900)
    im = Image.open(raw_image)
    bg = Image.new("RGB", im.size, (255,255,255))
    bg.paste(im, (0,0), im)
    # rgb_im = im.convert('RGB')
    bg.save('../images/aaa{}.jpg'.format(a), quality=95)

    #prepare image
    jpg_img_path = "../images/aaa{}.jpg".format(a)
    clean_image = prepare_image(jpg_img_path)

    #load model
    loaded_model = load_model()

    #making perfect input for model
    clean_image = np.expand_dims(clean_image, axis=0)
    clean_image = np.expand_dims(clean_image, axis=3)

    #predict
    results = loaded_model.predict(clean_image)
    # print(results)
    # print(results.shape)

    result = np.argmax(results)

   

    #return appropriate result
    return map_index_with_result(result)
    

def check_argmax():
    collection = np.array([[34,45,76,78,45,90]])
    r = np.argmax(collection)
    return r

if __name__ == "__main__":
    pass
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # print("its main")
    # load_model()