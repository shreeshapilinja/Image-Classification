import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from werkzeug.utils import secure_filename

# Set the path to the model and label encoder classes
model_path = 'vehicle_classification_model.h5'
label_encoder_path = 'label_encoder_classes.npy'

# Load the model and label encoder
model = load_model(model_path)
label_encoder = np.load(label_encoder_path, allow_pickle=True)

# Create Flask app
app = Flask(__name__)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded.')
    
    file = request.files['file']

    # Check if a file with allowed extension was uploaded
    if file.filename == '':
        return render_template('index.html', message='No file selected.')

    if not allowed_file(file.filename):
        return render_template('index.html', message='Invalid file extension. Only PNG, JPG, and JPEG files are allowed.')

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read the uploaded image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))  # Resize image for model input
    img = img.astype('float32') / 255.0  # Normalize image

    # Perform prediction
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_class_index = np.argmax(prediction)
    predicted_class = label_encoder[predicted_class_index]


    # Render the result page
    return render_template('result.html', image_file=filepath, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
