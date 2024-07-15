
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


class ObjectDetector:
    def __init__(self):
        self.load_model()
        self.model = null
    
    def load_model(self):
        try:
            self.model =  YOLO("yolov8n.pt")
        except Exception as e:
            raise Exception(f"Error loading YOLO model: {str(e)}")
    
    def detect_objects(self, image_path):
        try:
            image = cv2.imread(image_path)
            results = self.model(image)
            annotated_image = results[0].plot()  # results[0] contains the results for the first image
            return annotated_image
        except Exception as e:
            raise Exception(f"Error detecting objects: {str(e)}")
    
    def save_annotated_image(self, image_path, output_path):
        try:
            annotated_image = self.detect_objects(image_path)
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved successfully at: {output_path}")
        except Exception as e:
            raise Exception(f"Error saving annotated image: {str(e)}")
