import os
import cv2

import matplotlib.pyplot as plt

from detect import Detector, NoDetectionsException
from classify import Classifier


# PATH_IMAGE = "/Users/machofv/Projects/road_signs_binary_classification/YOLOv8/dataset/images/test/7c08f32689c9e2e1.jpg"
# PATH_IMAGE = "/Users/machofv/Projects/road_signs_binary_classification/application/resources/austria_1.png"

# 1
PATH_IMAGE = "/Users/machofv/Downloads/GTSDB-archive-2/TrainIJCNN2013/TrainIJCNN2013/00169.ppm"
# 0
# PATH_IMAGE = "/Users/machofv/Downloads/GTSDB-archive-2/TrainIJC NN2013/TrainIJCNN2013/00171.ppm"

# 2
# PATH_IMAGE = "/Users/machofv/Downloads/GTSDB-archive-2/TrainIJCNN2013/TrainIJCNN2013/00159.ppm"

# DIR_TEST_IMAGES = "/Users/machofv/Projects/road_signs_binary_classification/YOLOv8/dataset/images/test"

# DIR_TEST_IMAGES = "/Users/machofv/Downloads/GTSDB-archive-2/TestIJCNN2013/TestIJCNN2013Download"


# DIR_TEST_IMAGES = "/Users/machofv/Downloads/archive/traffic_Data/TEST"
DIR_TEST_IMAGES = "/Users/machofv/Downloads/GTSDB-archive-2/TrainIJCNN2013/TrainIJCNN2013"
DIR_TEST_IMAGES = "/Users/machofv/Downloads/GTSDB-archive-2/TestIJCNN2013/TestIJCNN2013Download"

# DIR_TEST_IMAGES = "/Users/machofv/Projects/road_signs_binary_classification/YOLOv8/dataset/images/test"

# DIR_TEST_IMAGES = "/Users/machofv/Projects/road_signs_binary_classification/application/resources/datasets/custom_google_street_view"

detector = Detector()
classifier = Classifier()

def run_solo_img(PATH_IMAGE, detector, classifier):
    try:
        imgs_detected = detector.detect_yolo(PATH_IMAGE)
    except NoDetectionsException:
        pass

    predictions = []
    for img_detected in imgs_detected:
        predictions.append(classifier.predict(img_detected))

    NUM_OF_PREDS = len(predictions)
    img = cv2.imread(f"{PATH_IMAGE}")

    # Visualize the data
    fig, axs = plt.subplots(1, 1+NUM_OF_PREDS, figsize=(15, 5))
    # Show original image

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original image')
    axs[0].axis('off')

    for i, img_disp in enumerate(imgs_detected):
        axs[i+1].imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
        axs[i+1].set_title(predictions[i])
        axs[i+1].axis("off")
    plt.tight_layout()
    plt.show()

""" ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----"""

def run_from_folder(DIR_IMAGES, detector, classifier):
    test_imgs = os.listdir(DIR_IMAGES)
    for IMG_PATH in test_imgs[:30]:

        imgs_detected = []
        if IMG_PATH == ".DS_Store":
            continue
        try:
            imgs_detected = detector.detect_yolo(f"{DIR_IMAGES}/{IMG_PATH}")
        except NoDetectionsException:
            pass

        predictions = []
        if imgs_detected:
            for img_detected in imgs_detected:
                predictions.append(classifier.predict(img_detected))

            print(len(predictions))

            NUM_OF_PREDS = len(predictions)
            img = cv2.imread(f"{DIR_IMAGES}/{IMG_PATH}")

            # Visualize the data
            fig, axs = plt.subplots(1, 1+NUM_OF_PREDS, figsize=(15, 5))
            # Show original image

            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Original image')
            axs[0].axis('off')

            for i, img_disp in enumerate(imgs_detected):
                axs[i+1].imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
                axs[i+1].set_title(predictions[i])
                axs[i+1].axis("off")
            plt.tight_layout()
            plt.show()

run_from_folder(DIR_TEST_IMAGES, detector, classifier)
# run_from_folder("/Users/machofv/Downloads/GTSDB-archive-2/TestIJCNN2013/TestIJCNN2013Download", detector, classifier)

# run_solo_img(PATH_IMAGE, detector, classifier)

# Add filtering if image is too small - 16x16?