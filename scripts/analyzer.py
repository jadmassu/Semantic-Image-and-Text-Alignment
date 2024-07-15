import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ImageAnalyzer:
    def __init__(self, image_path):
        try:
            self.image_path = image_path
            self.image = Image.open(image_path)
            self.image_array = np.array(self.image)
            self.height, self.width, self.channels = self.image_array.shape
            self.data_type = self.image_array.dtype
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file '{image_path}' not found.")
        except Exception as e:
            raise Exception(f"Error loading image '{image_path}': {str(e)}")
    
    def display_image(self):
        try:
            plt.figure(figsize=(8, 8))
            plt.imshow(self.image)
            plt.axis('off')
            plt.show()
        except Exception as e:
            raise Exception(f"Error displaying image: {str(e)}")

    def plot_histogram(self):
        try:
            if self.channels == 1:
                plt.hist(self.image_array.ravel(), bins=256, range=(0, 255), density=True)
                plt.title('Histogram of Pixel Intensities')
                plt.xlabel('Pixel Intensity')
                plt.ylabel('Normalized Frequency')
                plt.show()
            else:
                colors = ('r', 'g', 'b')
                for i, color in enumerate(colors):
                    hist, bins = np.histogram(self.image_array[:,:,i].ravel(), bins=256, range=(0, 255), density=True)
                    plt.plot(hist, color=color, alpha=0.7, label=color)
                plt.title('Histogram of Color Channels')
                plt.xlabel('Pixel Intensity')
                plt.ylabel('Normalized Frequency')
                plt.legend()
                plt.show()
        except Exception as e:
            raise Exception(f"Error plotting histogram: {str(e)}")

