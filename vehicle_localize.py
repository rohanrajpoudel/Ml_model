import cv2
import numpy as np
from preProcess import visualize_images, preprocess_image
# from colorThreshold import localize_vehicle
from edge import edge_detection

def vehicle_localization(image, edges):
    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area, aspect ratio, etc.
    potential_vehicle_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        aspect_ratio = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0

        # Add additional filters based on your requirements
        if area > 100 and 0.2 < aspect_ratio < 4:
            potential_vehicle_contours.append(contour)

    # Draw the potential contours on a copy of the original image
    localized_image = cv2.resize(image, (500, 500))
    cv2.drawContours(localized_image, potential_vehicle_contours, -1, (0, 255, 0), 2)

    # Display the results
    # cv2.imshow('Localized Image', localized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    visualize_images(cv2.resize(image, (500, 500)), preprocess_image(image))
    visualize_images(edge_detection(preprocess_image(image)), localized_image)

# Assuming 'edges_image' is the result from edge detection
image = cv2.imread('2.jpg')  # Replace with the path to your actual image
edges_image = edge_detection(preprocess_image(image))
vehicle_localization(image, edges_image)
