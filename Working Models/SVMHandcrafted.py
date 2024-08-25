import pandas as pd
import numpy as np
import cv2
import os
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the CSV file
image_labels = pd.read_csv(r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\csv\MAFA_training_data.csv')
path_to_images = r"C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Datasets\MAFA\MAFA-Label-Train\train-images"

# Add mask_label column
image_labels['mask_label'] = np.where((image_labels['occluder_type'] == 1) | (image_labels['occluder_type'] == 2), 1, 0)

# Define image size
img_size = 100

# Load and preprocess images
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    # Extract handcrafted features (e.g., color histograms, texture features)
    features = extract_features(img)
    return features

# Example handcrafted feature extraction function
def extract_features(img):
    # Compute color histograms
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    # Concatenate color histograms into a single feature vector
    features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])

    return features

# Load images and labels
X = np.array([preprocess_image(os.path.join(path_to_images, img_name)) for img_name in image_labels['imgName']])
y = image_labels['mask_label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model with a limit on the number of iterations
max_iterations = 1000  # Set the maximum number of iterations
svm_model = SVC(kernel='linear', max_iter=max_iterations)
svm_model.fit(X_train, y_train)

# Evaluate the model on test data
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# for i in range(5):
#     selected_row = random.choice(range(len(test_data)))

#     row = test_data.iloc[selected_row]
#     img_name = row['imgName']
#     img_path = os.path.join(test_images_path, img_name)
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Check if there are multiple faces for this image
#     faces = test_data[test_data['imgName'] == img_name]
#     num_faces_with_mask = 0
#     num_faces_without_mask = 0
#     combined_mask_probability = 0.0
#     combined_no_mask_probability = 0.0

#     for _, face_row in faces.iterrows():
#         face_x, face_y, face_w, face_h = face_row[['face_x', 'face_y', 'face_w', 'face_h']]
#         occluder_type = face_row['occluder_type']

#         # Check if the occluder type is a mask
#         is_mask = occluder_type == 1 or occluder_type == 2

#         # Preprocess the face image
#         face_img = img[int(face_y):int(face_y + face_h), int(face_x):int(face_x + face_w)]
#         face_img = cv2.resize(face_img, (img_size, img_size))
#         face_img = face_img / 255.0  # Normalize pixel values
#         face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

#         # Predict using the model
#         prediction = svm_model.predict(face_img)
#         mask_probability = prediction[0][1]  # Probability of wearing a mask

#         if is_mask:
#             num_faces_with_mask += 1
#             combined_mask_probability += mask_probability
#         else:
#             num_faces_without_mask += 1
#             combined_no_mask_probability += mask_probability

#         # Draw a bounding box around the face
#         cv2.rectangle(img, (int(face_x), int(face_y)), (int(face_x + face_w), int(face_y + face_h)), (0, 255, 0), 2)

#     # Display the image with aggregated face mask information
#     plt.subplot(2, 5, i+1)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(f'Image {i+1}')

#     plt.subplot(2, 5, i+6)
#     plt.text(0.5, 1.1, f'Faces with mask: {num_faces_with_mask}, Probability: {combined_mask_probability / num_faces_with_mask if num_faces_with_mask > 0 else 0:.2f}', color='green', fontsize=10, ha='center')
#     plt.text(0.5, 1.05, f'Faces without mask: {num_faces_without_mask}, Probability: {combined_no_mask_probability / num_faces_without_mask if num_faces_without_mask > 0 else 0:.2f}', color='red', fontsize=10, ha='center')
#     plt.axis('off')

# plt.tight_layout()
# plt.show()