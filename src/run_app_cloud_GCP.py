import os
import cv2
import json
import requests

import numpy as np
import matplotlib.pyplot as plt

from detect import Detector, NoDetectionsException
from classify import Classifier

HTTP_FUNCTION = "https://europe-central2-aesthetic-fiber-428811-m0.cloudfunctions.net/rs-recognize"

# 0
# PATH_IMAGE = "/Users/machofv/Downloads/GTSDB-archive-2/TrainIJC NN2013/TrainIJCNN2013/00171.ppm"

# 1
# PATH_IMAGE = "/Users/machofv/Downloads/GTSDB-archive-2/TrainIJCNN2013/TrainIJCNN2013/00169.ppm"

# 2
PATH_IMAGE = "/Users/machofv/Downloads/GTSDB-archive-2/TrainIJCNN2013/TrainIJCNN2013/00159.ppm"

def __img_to_json(img: np.ndarray) -> dict:
    """Jsonifies given array."""
    # return json.dumps(img.tolist())
    return {"ndarray_data": json.dumps(img.tolist())}

def __send_request(message: dict, url: str):
    response = requests.post(url, json=message)
    if response.status_code == 200:
        print("response:", response.text)
    else:
        print(f"Error: {response.status_code}, {response.text}")

def run(detector: Detector, img_path: str) -> None:
    try:
        imgs_detected = detector.detect_yolo(img_path)
    except NoDetectionsException:
        return
    
    predictions = []
    if imgs_detected:
        for img_detected in imgs_detected:
            # predictions.append(classifier.predict(img_detected))

            # Preprocess the image
            img_resized =  np.expand_dims(cv2.resize(img_detected, (64, 64)), axis=0)
            message_json = __img_to_json(img_resized)

            response = __send_request(message_json, HTTP_FUNCTION)

            # Check the response
            # print(response)
            # if response.status_code == 200:
            #     print(response.text)
            # else:
            #     print(f"Error: {response.status_code}, {response.text}")


# def run_from_folder(DIR_IMAGES, detector, classifier):
#     test_imgs = os.listdir(DIR_IMAGES)
#     for IMG_PATH in test_imgs[:30]:

#         imgs_detected = []
#         if IMG_PATH == ".DS_Store":
#             continue
#         try:
#             imgs_detected = detector.detect_yolo(f"{DIR_IMAGES}/{IMG_PATH}")
#         except NoDetectionsException:
#             pass

#         predictions = []
#         if imgs_detected:
#             for img_detected in imgs_detected:
#                 predictions.append(classifier.predict(img_detected))

#             print(len(predictions))

#             NUM_OF_PREDS = len(predictions)
#             img = cv2.imread(f"{DIR_IMAGES}/{IMG_PATH}")

#             # Visualize the data
#             fig, axs = plt.subplots(1, 1+NUM_OF_PREDS, figsize=(15, 5))
#             # Show original image

#             axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             axs[0].set_title('Original image')
#             axs[0].axis('off')

#             for i, img_disp in enumerate(imgs_detected):
#                 axs[i+1].imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
#                 axs[i+1].set_title(predictions[i])
#                 axs[i+1].axis("off")
#             plt.tight_layout()
#             plt.show()

detector = Detector()
run(detector, PATH_IMAGE)