import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.applications import EfficientNetB0
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import scipy.io as sio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import logging

# Load the CSV file
image_labels = pd.read_csv(r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\csv\MAFA_training_data.csv')
path_to_images = r"C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Datasets\MAFA\MAFA-Label-Train\train-images"

# Add mask_label column
image_labels['mask_label'] = np.where((image_labels['occluder_type'] == 1) | (image_labels['occluder_type'] == 2), 1, 0)

# Define image size
img_size = 100  # Using a larger image size for better feature extraction

# Load and preprocess images
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize pixel values
    return img

# Load images and labels
X = np.array([preprocess_image(os.path.join(path_to_images, img_name)) for img_name in image_labels['imgName']])
y = to_categorical(image_labels['mask_label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model by adding custom layers on top of the base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 output nodes for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the split test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes)
recall = recall_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes)
print(f'Accuracy on split test data: {accuracy}')
print(f'Precision on split test data: {precision}')
print(f'Recall on split test data: {recall}')
print(f'F1 Score on split test data: {f1}')

# Load the actual test data
test_data = pd.read_csv(r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\csv\MAFA_test_data.csv')
test_images_path = r"C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Datasets\MAFA\MAFA-Label-Test\test-images"

# Preprocess the test images
X_test_actual = np.array([preprocess_image(os.path.join(test_images_path, img_name)) for img_name in test_data['imgName']])
y_test_actual = to_categorical(test_data['mask_label'])

# Get the predicted labels for the actual test data
y_pred_actual = model.predict(X_test_actual)
y_pred_actual_classes = np.argmax(y_pred_actual, axis=1)
y_test_actual_classes = np.argmax(y_test_actual, axis=1)

# Calculate the metrics on the actual test data
accuracy_actual = accuracy_score(y_test_actual_classes, y_pred_actual_classes)
precision_actual = precision_score(y_test_actual_classes, y_pred_actual_classes)
recall_actual = recall_score(y_test_actual_classes, y_pred_actual_classes)
f1_actual = f1_score(y_test_actual_classes, y_pred_actual_classes)

# Print the metrics on the actual test data
print(f'\nAccuracy on actual test data: {accuracy_actual}')
print(f'Precision on actual test data: {precision_actual}')
print(f'Recall on actual test data: {recall_actual}')
print(f'F1 Score on actual test data: {f1_actual}')

# Display test images with aggregated face mask information
# ... (code for displaying test images remains the same)

# Configure logging
log_filename = r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\logs\EfficientNetB0.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Log the evaluation metrics on the split test data
logging.info(f'Accuracy on split test data: {accuracy}')
logging.info(f'Precision on split test data: {precision}')
logging.info(f'Recall on split test data: {recall}')
logging.info(f'F1 Score on split test data: {f1}')

# Log the evaluation metrics on the actual test data
logging.info(f'Accuracy on actual test data: {accuracy_actual}')
logging.info(f'Precision on actual test data: {precision_actual}')
logging.info(f'Recall on actual test data: {recall_actual}')
logging.info(f'F1 Score on actual test data: {f1_actual}')