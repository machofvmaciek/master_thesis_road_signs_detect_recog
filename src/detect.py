"""Implements a detector class to use YOLO / Faster-RCNN models."""
import os
import cv2

from PIL import Image
from pathlib import Path
from ultralytics import YOLO

MODEL_TYPE = "YOLO"
# PATH_MODEL = "/Users/machofv/Projects/road_signs_binary_classification/application/resources/models/YOLO/best_kuba.pt"
PATH_MODEL = f"{os.path.dirname(__file__)}/../resources/models/YOLO/best_kuba.pt"

THRESHOLD_CONFIDENCE = 0.6

class NoDetectionsException(Exception):
    """Raised where nothing was detected."""

class Detector:
    def __init__(self, ignore_tiny = True, ignore_unconfident = True):
        self.PATH_MODEL = {"YOLO": Path(PATH_MODEL)}
        self.ignore_tiny = ignore_tiny
        self.ignore_unconfident = ignore_unconfident
    
    def __get_model(self, model_name: str):
        return YOLO(self.PATH_MODEL.get(model_name))
    
    def detect_yolo(self, PATH_IMAGE):
        model = self.__get_model("YOLO")
        
        img = cv2.imread(f"{PATH_IMAGE}")
        if img is None:
            raise IOError(f"Failed to open '{PATH_IMAGE}' image.")
        
        # [0] as we're passing only one image at the time
        results = model(img, device="mps")[0]

        if not results:
            raise NoDetectionsException

        cropped_imgs = []

        for result in results:
            for box in result.boxes:
                # Filter out unconfident detections
                if self.ignore_unconfident:
                    if box.conf.item() < THRESHOLD_CONFIDENCE:
                        continue

                # Extract Bounding Box Coordinates
                x1, y1, x2, y2 = box.xyxy[0]

                if self.ignore_tiny:
                    # Filter out very small detections
                    if x2-x1 < 15 or y2-y1 < 15:
                        continue

                # Convert to Integer for Cropping
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Crop
                cropped_imgs.append(img[y1:y2, x1:x2])

        return cropped_imgs
