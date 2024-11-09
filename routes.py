from flask import Flask, request, jsonify, session, render_template, send_file, redirect, url_for, flash, send_from_directory
import io
from func import validate_user, preprocess_image ,allowed_file
from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "your_secret_key"


IMAGE_HEIGHT = 150  
IMAGE_WIDTH = 150  


#change the model path
MODEL_PATH = r"C:\Users\shankaripriya s\OneDrive\Desktop\python difference\Breast Cancer vs. Fibroadenoma An AI-Driven Differentiation\breast_ultrasound_model.h5"
model = load_model(MODEL_PATH)

@app.route('/ultrasound', methods=['GET', 'POST'])
def ultrasound():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file', 400
        
        if file:
            filename = secure_filename(file.filename)
            upload_folder = app.config['UPLOAD_FOLDER']
            file_path = os.path.join(upload_folder, filename)
            
            file.save(file_path)

            session['file_path'] = file_path 
            session['filename'] = filename
            
            return render_template('ultrasound.html',filename=filename)
    
    return render_template('ultrasound.html')

@app.route('/result', methods=['GET'])
def result():
    uploaded_filename = session.get('filename')
    file_path = session.get('file_path')

    img = image.load_img(file_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) 

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    class_mapping = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
    predicted_label = class_mapping.get(predicted_class, 'Unknown')
    print(predicted_label)

    return render_template('result.html',
                           filename=uploaded_filename,
                           predicted_label=predicted_label,
                           confidence=confidence)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/bloodtest')
def bloodtest():
    return render_template('bloodtest.html')



@app.route('/biopsy')
def biopsy():
    return render_template('biopsy.html')




@app.route('/logout',methods = ['GET'])
def logout():
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if validate_user(username, password):
            session['username'] = username  
            return render_template('home.html')
        else:
            return render_template('login.html', error="Invalid Username or Password.")
    
    return render_template('login.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)