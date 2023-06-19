import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.models import save_model

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
            img = cv2.resize(img, (64, 64))  # Resize image to a fixed size
            images.append(img)
            labels.append('bike')

    # Load images of four-wheelers
    for filename in os.listdir(four_wheeler_dir):
        img = cv2.imread(os.path.join(four_wheeler_dir, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize image to a fixed size
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

# Build the CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
model = build_model()

# Train the model
model.fit(X_train, y_train, epochs=16, batch_size=12, validation_data=(X_test, y_test))

# Save the model
save_model(model, 'vehicle_classification_model_ad.h5')

# Save the label encoder classes
np.save('label_encoder_classes_ad.npy', label_encoder.classes_)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('\nTest Loss:', loss)
print('Test Accuracy:', accuracy)
