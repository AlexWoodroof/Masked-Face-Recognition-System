import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.applications import EfficientNetB0
from keras.models import Model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
y = image_labels['mask_label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Create a new model by removing the last classification layer
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Freeze the base model's layers
for layer in feature_extractor.layers:
    layer.trainable = False

# Extract features from the training data
X_train_features = feature_extractor.predict(X_train)
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)

# Extract features from the test data
X_test_features = feature_extractor.predict(X_test)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# Train the SVM model with the extracted features
svm_model = SVC(kernel='linear', max_iter=1000)
svm_model.fit(X_train_features, y_train)

# Evaluate the hybrid model on test data
y_pred = svm_model.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')