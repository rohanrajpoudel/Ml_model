import cv2
import numpy as np
from preProcess import visualize_images, preprocess_image

def localize_vehicle(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for vehicles (you need to adjust these based on your scenario)
    lower_color = np.array([0, 0, 140])  # Example: lower bound for white color
    upper_color = np.array([255, 20, 255])  # Example: upper bound for white color

    # Create a binary mask using color thresholding
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=color_mask)

    return result

# Load an example image (replace 'your_image.jpg' with the actual image file)
image_path = "1.jfif"
original_image = cv2.imread(image_path)
preprocessed_image = preprocess_image(original_image)

# Localize the vehicle using color thresholding
localized_image = localize_vehicle(preprocessed_image)
# localized_image = localize_vehicle(original_image)

# Display the results
# visualize_images(original_image, localized_image)
# cv2.imshow('Original Image', original_image)
# cv2.imshow('Localized Vehicle', localized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
