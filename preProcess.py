import cv2

def resize_image(image):
    return cv2.resize(image, (500, 500))
def remove_noise(image):
    return cv2.GaussianBlur(image, (3, 7), 0)
def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)
def preprocess_image(image):
    resized = resize_image(image)
    denoised = remove_noise(resized)
    # contrast_enhanced = enhance_contrast(denoised)
    # return contrast_enhanced
    return denoised
def visualize_images(original, preprocessed):
    cv2.imshow("Original Image", original)
    cv2.imshow("Pre-processed Image", preprocessed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    orginal = cv2.imread("OIP.jfif")
    preprocessed_image = preprocess_image(orginal)
    visualize_images(orginal,preprocessed_image)


# if __name__ == "__main__":
    # main()
