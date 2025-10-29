import cv2

# try 3:4 image
def load_image(image_path, target_size=(600, 800)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    return img
