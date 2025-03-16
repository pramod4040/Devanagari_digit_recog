import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
from skimage import io, color
import numpy as np
from PIL import Image
import random
from typing import Dict, Union
from .ModelFactory import ModelFactory


def prepare_image(raw_image):
    print(raw_image)
    image_in_numpy = io.imread(raw_image)

    # Convert into grayscale image using scikit-image
    image_in_numpy = color.rgb2gray(image_in_numpy)

    averaged_image = np.zeros((1024,), dtype='float64')
    for y in range(0,32):
        for x in range(0,32):
            mean = 0
            for h in range(0,8):
                for k in range(0,8):
                    mean += image_in_numpy[y * 8 + h, x * 8 + k]
        mean = mean / 64
        averaged_image[y * 32 + x] = mean
    
    averaged_image = averaged_image.reshape((32,32))
    gg = 1 - averaged_image
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



def predict_character(image_path: str, model_type: str = 'ensemble', weightPath: str = None) -> Dict[str, Union[str, float]]:
    """
    Predict the character in the given image using the specified model type.
    
    Args:
        image_path (str): Path to the image file to predict
        model_type (str): Type of model to use (default: 'ensemble')
        
    Returns:
        Dict containing the predicted class and confidence score
    """
    # Create model using factory
    model = ModelFactory.create_model(model_type, weightPath)
    
    # Preprocess the image and get prediction
    # preprocessed_image = model.preprocess_image(image_path)

    result = model.predict(image_path)
    
    return result


def check_argmax():
    collection = np.array([[34,45,76,78,45,90]])
    r = np.argmax(collection)
    return r

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    result = predict_character(image_path)
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")