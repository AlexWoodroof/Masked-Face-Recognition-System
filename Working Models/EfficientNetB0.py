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

# Evaluate the model on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes)
recall = recall_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Load the test data
test_data = pd.read_csv(r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\csv\MAFA_test_data.csv')
test_images_path = r"C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Datasets\MAFA\MAFA-Label-Test\test-images"



# Display test images with aggregated face mask information
plt.figure(figsize=(20, 16))

for i in range(5):
    selected_row = random.choice(range(len(test_data)))

    row = test_data.iloc[selected_row]
    img_name = row['imgName']
    img_path = os.path.join(test_images_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Check if there are multiple faces for this image
    faces = test_data[test_data['imgName'] == img_name]
    num_faces_with_mask = 0
    num_faces_without_mask = 0
    combined_mask_probability = 0.0
    combined_no_mask_probability = 0.0

    for _, face_row in faces.iterrows():
        face_x, face_y, face_w, face_h = face_row[['face_x', 'face_y', 'face_w', 'face_h']]
        occluder_type = face_row['occluder_type']

        # Check if the occluder type is a mask
        is_mask = occluder_type == 1 or occluder_type == 2

        # Preprocess the face image
        face_img = img[int(face_y):int(face_y + face_h), int(face_x):int(face_x + face_w)]
        face_img = cv2.resize(face_img, (img_size, img_size))
        face_img = face_img / 255.0  # Normalize pixel values
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(face_img)
        mask_probability = prediction[0][1]  # Probability of wearing a mask

        if is_mask:
            num_faces_with_mask += 1
            combined_mask_probability += mask_probability
        else:
            num_faces_without_mask += 1
            combined_no_mask_probability += mask_probability

        # Draw a bounding box around the face
        cv2.rectangle(img, (int(face_x), int(face_y)), (int(face_x + face_w), int(face_y + face_h)), (0, 255, 0), 2)

    # Display the image with aggregated face mask information
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Image {i+1}')

    plt.subplot(2, 5, i+6)
    plt.text(0.5, 1.1, f'Faces with mask: {num_faces_with_mask}, Probability: {combined_mask_probability / num_faces_with_mask if num_faces_with_mask > 0 else 0:.2f}', color='green', fontsize=10, ha='center')
    plt.text(0.5, 1.05, f'Faces without mask: {num_faces_without_mask}, Probability: {combined_no_mask_probability / num_faces_without_mask if num_faces_without_mask > 0 else 0:.2f}', color='red', fontsize=10, ha='center')
    plt.axis('off')

plt.tight_layout()
plt.show()

import logging

# Configure logging
log_filename = r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\logs\EfficientNetB0.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Log the evaluation metrics
logging.info(f'Accuracy: {accuracy}')
logging.info(f'Precision: {precision}')
logging.info(f'Recall: {recall}')
logging.info(f'F1 Score: {f1}')