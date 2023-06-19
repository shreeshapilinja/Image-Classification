import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Load the trained model
model = load_model('vehicle_classification_model_ad.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes_ad.npy')

app = Flask(__name__)

# Preprocess the uploaded image
def preprocess_image(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict the class of the uploaded image
def predict_class(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    vehicle_class = label_encoder.classes_[class_index]
    return vehicle_class

@app.route('/', methods=['GET', 'POST'])
def classify_vehicle():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', message='No image file selected.')
        
        image = request.files['image']
        
        if image.filename == '':
            return render_template('index.html', message='No image file selected.')
        
        vehicle_class = predict_class(image)
        
        return render_template('index.html', message='Predicted class: {}'.format(vehicle_class))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)