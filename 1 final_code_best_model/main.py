import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
import io

# Load the Keras model
model = tf.keras.models.load_model("model_keras.h5")
classes = ["bike", "car"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((300, int(300 * img.size[1] / img.size[0])), Image.ANTIALIAS)
    inp_numpy = np.array(img)[None]
    return inp_numpy

# Predict the class of the uploaded image
def predict_class(image):
    img = preprocess_image(image)
    class_scores = model.predict(img)[0]
    predicted_class = classes[class_scores.argmax()]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', message='No image file selected.')

        image = request.files['image']
        
        if image.filename == '':
            return render_template('index.html', message='No image file selected.')
        
        # Save the uploaded image to the static folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        
        predicted_class = predict_class(image)
        
        return render_template('index.html', predicted_class=predicted_class, image_url=image_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
