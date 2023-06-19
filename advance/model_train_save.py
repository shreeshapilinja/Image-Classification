import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set the path to the dataset directories
dataset_dir = '../datasets'
two_wheeler_dir = os.path.join(dataset_dir, 'bike')
four_wheeler_dir = os.path.join(dataset_dir, 'car')

# Load and preprocess the dataset
def load_dataset():
    images = []
    labels = []

    # Load images of two-wheelers
    for filename in os.listdir(two_wheeler_dir):
        img = cv2.imread(os.path.join(two_wheeler_dir, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize image for VGG16
            images.append(img)
            labels.append('bike')

    # Load images of four-wheelers
    for filename in os.listdir(four_wheeler_dir):
        img = cv2.imread(os.path.join(four_wheeler_dir, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize image for VGG16
            images.append(img)
            labels.append('car')

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Split the dataset into training and testing sets
def split_dataset(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.01, random_state=42)
    return X_train, X_test, y_train, y_test

# Preprocess and normalize the images
def preprocess_images(X_train, X_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, X_test

# Convert labels to one-hot encoding
def encode_labels(y_train, y_test):
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return y_train, y_test, label_encoder

# Build the CNN model with VGG16 backbone
def build_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess the dataset
images, labels = load_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_dataset(images, labels)

# Preprocess and normalize the images
X_train, X_test = preprocess_images(X_train, X_test)

# Convert labels to one-hot encoding
y_train, y_test, label_encoder = encode_labels(y_train, y_test)

# Build the CNN model
model = build_model(num_classes=len(label_encoder.classes_))

# Define callbacks for early stopping and model checkpoint
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1),
             ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)]

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# Save the model
model.save('vehicle_classification_model.h5')

# Save the label encoder classes
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('\nTest Loss:', loss)
print('Test Accuracy:', accuracy)
