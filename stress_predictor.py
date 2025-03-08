import tensorflow as tf
import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image


interpreter = tf.lite.Interpreter(model_path="stress_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def stress_predictor(img):
    # img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)

    return output_data[0][0]


def read_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = np.array(img)

    if img is None or img.size == 0:
        raise ValueError("Failed to load image from URL.")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


if __name__ == "__main__":
    a, b = stress_predictor("test.jpg")
    print(b[0][0])
    a, b = stress_predictor("test1.jpg")
    print(b[0][0])
