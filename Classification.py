import keras
import cv2
import numpy as np

# call the model
model = keras.models.load_model('./model/CNN_20epochBG.h5')

def upload_image(image_path):
    img = cv2.imread(image_path)
    return img

def classify_image(img):
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    result = model.predict(img)
    if result[0][0] > 0.5:
        result = 'Machine-printed'
    else:
        result = 'Handwritten'
    return result
