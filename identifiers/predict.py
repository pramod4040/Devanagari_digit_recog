import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json
from skimage import io, color
import numpy as np
from PIL import Image
import random
from typing import Dict, Union
# from .ModelFactory import ModelFactory
from identifiers.ModelFactory import ModelFactory


current_file = os.path.abspath(__file__)

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
    
    # print(model.model.summary())
    
    # Preprocess the image and get prediction
    # preprocessed_image = model.preprocess_image(image_path)

    result = model.predict(image_path)
    
    return result


def check_argmax():
    collection = np.array([[34,45,76,78,45,90]])
    r = np.argmax(collection)
    return r

# Traverse up until you find the top-level folder you expect
def get_project_root(target_folder_name: str = "Devanagari_digit_recog"):
    path = current_file
    while True:
        path, folder = os.path.split(path)
        if folder == target_folder_name:
            return os.path.join(path, folder)
        if folder == "":
            raise Exception(f"Could not find folder named {target_folder_name}")

project_root = get_project_root()

def models_and_weights():
    models = ModelFactory.get_models()
    modelDetails = []
    for model in models:
        if model == 'enhancedMoeModel':
            weight_path = os.path.join(
                project_root,
                "trainModel", 
                "model_checkpoints", 
                "moe_optimized_model-23-0.9473.weights.h5"
            )
            details = { "modelName": 'enhancedMoeModel',
                        "weightPath": weight_path,
                        "description": "Enhanced mode model with 3 experts!"
                       }
            modelDetails.append(details)
        elif model == 'moeModelV2Good':
            weight_path = os.path.join(
                project_root,
                "trainModel", 
                "model_checkpoints", 
                "MoeModelV2_moe_optimized_model-3-expert-20-0.9737.weights.h5"
            )
            details = { "modelName": 'moeModelV2Good',
                        "weightPath": weight_path,
                        "description": "Enhanced mode model with 4 experts and trained 25 epochs best model so far!"
                       }
            modelDetails.append(details)

        
    return modelDetails   

def get_model_details(modelName):
    modelDetails = models_and_weights()
    for model in modelDetails:
        if model['modelName'] == modelName:
            return model
        


if __name__ == "__main__":
    # Example usage
    # image_path = "path/to/your/image.jpg"
    # result = predict_character(image_path)
    # print(f"Predicted class: {result['class']}")
    # print(f"Confidence: {result['confidence']:.2%}")

    result = models_and_weights()
    print(result)