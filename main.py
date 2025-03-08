import tensorflow as tf
import numpy as np
import cv2


model = tf.keras.models.load_model("stress_detector.h5", compile=False)
model.summary()
