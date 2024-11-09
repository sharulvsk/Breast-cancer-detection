
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image


def validate_user(username, password):
    if username == "admin@gmail.com" and password == "pass":
        return True
    elif username == "user@gmail.com" and password == "pass":
        return True
    return False

def preprocess_image(file_path):
    img = load_img(file_path, target_size=(224, 224)) 
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0  
    return img_array

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


