import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image
import extcolors

class ImageProcessor:
    def __init__(self, rpath):
        self.rpath = rpath
    
    def load_image(self, image_path):
        try:
            image = plt.imread(image_path)
            return image
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File not found at {image_path}")
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def grayscale_conversion(self, image):
        try:
            gray_image = rgb2gray(image)
            return gray_image
        except Exception as e:
            raise Exception(f"Error converting image to grayscale: {str(e)}")
    
    def binary_thresholding(self, gray_image, threshold=0.5):
        try:
            binary_image = np.where(gray_image > threshold, 1, 0)
            return binary_image
        except Exception as e:
            raise Exception(f"Error in binary thresholding: {str(e)}")
    
    def template_matching(self, main_image_path, template_image_path, threshold=0.8):
        try:
            main_image = cv2.imread(main_image_path, cv2.IMREAD_COLOR)
            template_image = cv2.imread(template_image_path, cv2.IMREAD_COLOR)
            
            if main_image is None or template_image is None:
                raise FileNotFoundError("Error: Image file not found or cannot be read")
            
            main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            
            w, h = template_gray.shape[::-1]
            res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            
            loc = np.where(res >= threshold)
            
            if len(loc[0]) == 0:
                print("No match found. Try lowering the threshold.")
                return None
            else:
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(main_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                
                return main_image
        except FileNotFoundError as fnf_error:
            raise FileNotFoundError(f"Error: {fnf_error}")
        except Exception as e:
            raise Exception(f"Error in template matching: {str(e)}")
    
    def extract_colors(self, image_path, tolerance=33, limit=10):
        try:
            img = Image.open(image_path).convert("RGBA")
            colors = extcolors.extract_from_image(img, tolerance=tolerance, limit=limit)
            return colors
        except FileNotFoundError as fnf_error:
            raise FileNotFoundError(f"Error: {fnf_error}")
        except Exception as e:
            raise Exception(f"Error extracting colors: {str(e)}")
    
    def plot_image(self, image):
        try:
                plt.imshow(image)
                plt.axis('off')
                plt.show()
        except Exception as e:
                raise Exception(f"Error plotting image: {str(e)}")
