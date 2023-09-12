import os
import tensorflow as tf
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.models import load_model
# from tensorflow.keras import Sequential
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import model_from_json
from skimage import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from resizeimage import resizeimage
import random
# import all_class_names 
from .CreateEnsModel import getEfficientModel, get_class_names
import pickle


class EnsembleModelCharacter:
    def __init__(self, weightPath, class_names):
        self.weightPath = weightPath
        self.model = None
        self.class_names = class_names

        self.create_model()
        self.load_weights()

    def create_model(self):
        self.model = getEfficientModel()
    
    def load_weights(self):
        self.model.load_weights(self.weightPath)
        return self.model
    

    def predict(self, imagePath):
        imgpath = imagePath
        img = Image.open(imgpath).resize((32,32)).convert("RGB")
        # img = ImageOps.invert(img)

        imgNumpy = np.array(img)
        imgNumpy = tf.expand_dims(imgNumpy, 0)
        
        predictions = self.model.predict(imgNumpy)
        scores = tf.nn.softmax(predictions)

        predictedLabel = self.class_names[np.argmax(scores)]
        return predictedLabel



def magic(imagePath):
    weightPath= os.path.join(os.getcwd(), "identifiers", "trained_model", "model_weight_ensemble.h5")
    # jsonModelPath= os.path.join(os.getcwd(), "trained_model", "ensemble_json_model.json")
    label_names = get_class_names()

    model = EnsembleModelCharacter(weightPath, label_names)

    # pickle.dump(model, open('devnagari-best-model.pkl', 'wb'))
    result = model.predict(imagePath)
    return result


def get_class_names():
    class_names = ['character_10_yna',
    'character_11_taamatar',
    'character_12_thaa',
    'character_13_daa',
    'character_14_dhaa',
    'character_15_adna',
    'character_16_tabala',
    'character_17_tha',
    'character_18_da',
    'character_19_dha',
    'character_1_ka',
    'character_20_na',
    'character_21_pa',
    'character_22_pha',
    'character_23_ba',
    'character_24_bha',
    'character_25_ma',
    'character_26_yaw',
    'character_27_ra',
    'character_28_la',
    'character_29_waw',
    'character_2_kha',
    'character_30_motosaw',
    'character_31_petchiryakha',
    'character_32_patalosaw',
    'character_33_ha',
    'character_34_chhya',
    'character_35_tra',
    'character_36_gya',
    'character_3_ga',
    'character_4_gha',
    'character_5_kna',
    'character_6_cha',
    'character_7_chha',
    'character_8_ja',
    'character_9_jha',
    'digit_0',
    'digit_1',
    'digit_2',
    'digit_3',
    'digit_4',
    'digit_5',
    'digit_6',
    'digit_7',
    'digit_8',
    'digit_9']

    return class_names




# class NumberPredict:
#     def __init__(self):
#         # self.image_path = image_path
#         pass

#     def resize_image_into_32_32(self, img_path):
#         a = random.randint(100,900)
#         img = Image.open(img_path)
#         img = resizeimage.resize_contain(img, [32,32])
#         current_path = os.getcwd()
#         imgName = f'aaa_32_32_{a}.png'
#         imagePath = os.path.join(current_path, 'images', 'numbers', imgName)
#         img.save(imagePath, img.format)
#         return imagePath


#     def average_pixels_value_in_8_by_8(self, image_in_numpy):
#         averaged_image = np.zeros((1024,), dtype='uint8')
#         for y in range(0,32):
#             for x in range(0,32):
#                 mean = 0
#                 for h in range(0,8):
#                     for k in range(0,8):
#                         mean += image_in_numpy[y * 8 + h, x * 8 + k]
#             mean = mean / 64
#             averaged_image[y * 32 + x] = mean
#         return averaged_image


#     def prepare_image(self, raw_image, method=None):
#         if method == 'resize_32':
#             img_32_path = self.resize_image_into_32_32(raw_image)
#             # image_in_numpy = cv2.imread(img_32_path, cv2.IMREAD_GRAYSCALE)
#             image_in_numpy = cv2.imread(img_32_path)

#             return image_in_numpy

#         print(raw_image)
#         image_in_numpy = io.imread(raw_image)
#         print(image_in_numpy.shape)

#         #convert into grayscale image
#         # image_in_numpy = cv2.cvtColor(image_in_numpy, cv2.COLOR_RGB2GRAY)
#         image_in_numpy = cv2.cvtColor(image_in_numpy)
#         print("gray shape image")
#         print(image_in_numpy.shape)
    
#         # image_in_numpy = image_in_numpy / 255
#         # print(image_in_numpy)

#         # averaged_image = np.zeros((1024,), dtype='uint8')
#         # print(averaged_image)
#         averaged_image = self.average_pixels_value_in_8_by_8(image_in_numpy)
        
#         #reshaping the averaged_image into 32 / 32
#         averaged_image = averaged_image.reshape((32,32))

#         # print(averaged_image.shape)
#         gg = cv2.bitwise_not(averaged_image)
#         return gg


#     def load_model(self, modelName=None):
#         # currentPath = os.getcwd()
#         # path_for_json = os.path.join(currentPath, 'identifiers', 'trained_model', 'nepali_digit_model', 'model-0-9-98-precision.json')
#         # path_for_weights = os.path.join(currentPath, 'identifiers', 'trained_model', 'nepali_digit_model', 'weights-0-9-98-precision.h5')
    
#         # with open(path_for_json) as json_file:
#         #     loaded_model_json = json_file.read()
#         #     loaded_model = model_from_json(loaded_model_json)
#         #     json_file.close()
#         # loaded_model.load_weights(path_for_weights)
#         # loaded_model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
#         # return loaded_model
    

#         currentPath = os.getcwd()
#         weightPath = os.path.join(currentPath, 'identifiers', 'trained_model', 'consonants_model.h5')
#         model = DevnagariCharacter(weightPath).load_trained_model()
#         return model
    
    
#     def map_index_with_result(self, result, percent):
#         # label = [i for i in range(0,10)]
#         # result_list = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
#         # dict_result = dict(zip(label, result_list))


#         label = [i for i in range(0, 36)]
#         result_list = ['ka','kha','ga','gha','ŋa','cha','chha','ja','jha','ña','ta','tha','da','dha','ṇa','ṭa','ṭha','ḍa','ḍha','na','pa','pha','ba','bha','ma','ya','ra','la','wa','sa','sa','sa','ha','chhya','ṭra','gya']
#         dict_result = dict(zip(label, result_list))

#         a = {"predicted":dict_result[result], "percentage": percent * 100 }
#         return a


#     def predict_character(self, raw_image):
#         a = random.randint(100,900)

#         #convert .png into .jpg

#         # im = Image.open(raw_image)
#         # bg = Image.new("RGB", im.size, (255,255,255))
#         # bg.paste(im, (0,0), im)
#         # bg.save('../images/numbers/aaa{}.jpg'.format(a), quality=95)

#         #prepare image
#         # jpg_img_path = "../images/numbers/aaa{}.jpg".format(a)
#         clean_image = self.prepare_image(raw_image, method="resize_32")

#         #load model
#         loaded_model = self.load_model()

#         #making perfect input for model
#         clean_image = np.expand_dims(clean_image, axis=0)
#         clean_image = np.expand_dims(clean_image, axis=3)

#         #predict
#         results = loaded_model.predict(clean_image)
#         # print(results.shape)

#         result = np.argmax(results)
#         percent = results.max()
#         # self.print_percentage_of_result(results)

    

#         #return appropriate result
#         return self.map_index_with_result(result, percent)


#     def print_percentage_of_result(self,results):
#         print(results / 100)
#         return
    


# def create_object():
#     # if 'number_predict' in locals:
#     #     print("number_predict found in locals scope")
#     #     return number_predict
#     # if 'number_predict' in globals:
#     #     print("number_predict found in global scope")
#     #     return number_predict
    
#     number_predict = NumberPredict()
#     return number_predict

# def predict_number(image_path):
#     # num_predict = create_object()
#     print(image_path)
#     num_predict = NumberPredict()
#     # print(num_predict)
#     result = num_predict.predict_character(image_path)
#     return result





# /home/ubuntu/Thesis/Coding/Devanagari_digit_recog/identifiers/trained_model/model_weight_ensemble.h5