"""Implements a Classifier to use CNN model."""

import os
import cv2
import json

import numpy as np
# import tensorflow as tf

from pathlib import Path
from tensorflow.keras.models import load_model # type: ignore

# PATH_MODEL = "/Users/machofv/Projects/road_signs_binary_classification/application/resources/models/CNN/cnn _v6_ep_9.h5"
# PATH_MODEL = "/Users/machofv/Projects/road_signs_binary_classification/application/resources/models/CNN/TSR.h5"
# PATH_JSON_CLASSES = "/Users/machofv/Projects/road_signs_binary_classification/application/resources/configs/classes_gtsrb.json"

PATH_MODEL = f"{os.path.dirname(__file__)}/../resources/models/CNN/TSR.h5"
PATH_JSON_CLASSES = f"{os.path.dirname(__file__)}/../resources/configs/classes_gtsrb.json"

DEFAULT_IMAGE_SHAPE = (None, 64, 64, 3)

def resize_and_pad(image: np.ndarray, target_size: tuple[int, int]=(64, 64)) -> np.ndarray:
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate the scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image with the scaling factor
    resized_image = cv2.resize(image, (new_w, new_h))

    # Create a new image and fill it with the padding color (e.g., black)
    padded_image = np.full((target_h, target_w, 3), (0, 0, 0), dtype=np.uint8)
    
    # Calculate padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # Place the resized image onto the padded image
    padded_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_image

    return padded_image

class Classifier:
    def __init__(self):
        self.PATH_MODEL = {"CNN": Path(PATH_MODEL)}

        # Load classes from json
        with open(PATH_JSON_CLASSES, 'r') as file:
            self.classes = json.loads(file.read())
    
    def __get_model(self, model_name: str):
        return load_model(self.PATH_MODEL.get(model_name))
    
    def __preprocess_img(self, img):
        if img.shape == DEFAULT_IMAGE_SHAPE:
            return img

        # return np.expand_dims(resize_and_pad(img), axis=0)
        # return np.expand_dims(cv2.resize(img, (64, 64)), axis=0)
        return np.expand_dims(cv2.resize(img, (30, 30)), axis=0)    # TSR.h5
    
    
    def __get_class_name(self, id):
        return self.classes[id].get("SignName")


    def predict(self, img):
        model = self.__get_model("CNN")

        # img_prep = self.__preprocess_img(img)
        # predictions = model.predict(self.__preprocess_img(img))

        # pred_classes = np.argmax(predictions, axis=1)

        # prediction = np.argmax(model.predict(self.__preprocess_img(img)), axis=1)
        probabilities = model.predict(self.__preprocess_img(img))
        # print(probabilities)
        prediction = np.argmax(probabilities, axis=1)
        confidence = probabilities[0][prediction]
        # print(confidence)
        return self.__get_class_name(prediction[0])
