import pandas as pd
import numpy as np
import cv2
import os
from keras.applications import EfficientNetB0
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, con
import matplotlib.pyplot as plt
import random
import logging
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

# Load the training data
train_data = pd.read_csv(r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\csv\MAFA_training_data_unstructured.csv')
train_images_path = r"C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Datasets\MAFA\MAFA-Label-Train\train-images"

# Add mask_label column
train_data['mask_label'] = np.where((train_data['occluder_type'] == 1) | (train_data['occluder_type'] == 2), 1, 0)

# Define image size
img_size = 100  # Using a larger image size for better feature extraction

# Load and preprocess images
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize pixel values
    return img

# Load training images and labels
X_train = np.array([preprocess_image(os.path.join(train_images_path, img_name)) for img_name in train_data['imgName']])
y_train = to_categorical(train_data['mask_label'])

# Data augmentation
data_generator = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    zoom_range=[0.9, 1.1]
)

# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze some layers of the pre-trained model
for layer in base_model.layers[:100]:  # Freeze the first 100 layers
    layer.trainable = False

# Create a new model by adding custom layers on top of the base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 output nodes for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
model.fit(data_generator.flow(X_train, y_train, batch_size=32), epochs=5, steps_per_epoch=len(X_train) // 32)

# Load the test data
test_data = pd.read_csv(r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\csv\MAFA_test_data.csv')
test_images_path = r"C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Datasets\MAFA\MAFA-Label-Test\test-images"

# Preprocess the test images
X_test = []
y_test = []
for _, row in test_data.iterrows():
    img_name = row['imgName']
    occluder_type = row['occluder_type']

    # Check if the occluder type is a mask
    is_mask = occluder_type == 1 or occluder_type == 2

    # Preprocess the image
    img_path = os.path.join(test_images_path, img_name)
    img = preprocess_image(img_path)
    X_test.append(img)
    y_test.append(1 if is_mask else 0)

X_test = np.array(X_test)
y_test = to_categorical(y_test)

# Get the predicted labels for the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate the metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes)
recall = recall_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes)

# Print the metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Display test images with aggregated face mask information
# ... (code for displaying test images remains the same)

# Configure logging
log_filename = r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\logs\EfficientNetB0.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Log the evaluation metrics
logging.info(f'Accuracy: {accuracy}')
logging.info(f'Precision: {precision}')
logging.info(f'Recall: {recall}')
logging.info(f'F1 Score: {f1}')