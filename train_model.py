import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


IMAGE_HEIGHT = 150  
IMAGE_WIDTH = 150  
CHANNELS = 3       
BATCH_SIZE = 32
EPOCHS = 10       

#path to the dataset
DATASET_DIR = r'C:\Users\shankaripriya s\OneDrive\Desktop\Original Breast Cancer vs. Fibroadenoma An AI-Driven Differentiation\Dataset_BUSI_with_GT'

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax')) 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data(dataset_dir):
    benign_dir = os.path.join(dataset_dir, 'benign')
    malignant_dir = os.path.join(dataset_dir, 'malignant')
    normal_dir = os.path.join(dataset_dir, 'normal')

    images = []
    labels = []

    for label, class_dir in enumerate([benign_dir, malignant_dir, normal_dir]):
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist. Skipping.")
            continue
        
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)

            try:
                img = image.load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
                img_array = image.img_to_array(img) / 255.0  # Normalize
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    
    labels = to_categorical(labels, num_classes=3)
    
    return images, labels

def main():
    print("Loading data...")
    images, labels = load_data(DATASET_DIR)

    train_data, val_data, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = create_model()
    print("Model created.")

    print("Training the model...")
    model.fit(train_data, train_labels, epochs=EPOCHS, validation_data=(val_data, val_labels), batch_size=BATCH_SIZE)

    model.save('breast_ultrasound_model.h5')
    print("Model saved as 'breast_ultrasound_model.h5'.")
    
if __name__ == '__main__':
    main()
    