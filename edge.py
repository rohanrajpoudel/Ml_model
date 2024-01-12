import cv2
from preProcess import visualize_images, preprocess_image
from colorThreshold import localize_vehicle
import numpy as np

def edge_detection(image):
    # preprocessing the image
    reprocessedp_image = preprocess_image(image)
    # Using color thresholding
    colorThreshold_image = localize_vehicle(reprocessedp_image)
    # colorThreshold_image = preprocess_image(image)

    # Apply Canny edge detection
    edges = cv2.Canny(colorThreshold_image, 80, 220)  # You can adjust the thresholds based on your image

    # Optional: Apply morphological operations for edge refinement
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    return edges

# Display the results
def main():
    original_image = cv2.imread('2.jpg')
    # visualize_images(original_image, edge_detection(original_image))

main()