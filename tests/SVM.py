import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
    img = img.flatten() / 255.0  # Flatten and normalize pixel values
    return img

# Load images and labels
X = np.array([preprocess_image(os.path.join(path_to_images, img_name)) for img_name in image_labels['imgName']])
y = image_labels['mask_label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model with a limit on the number of iterations
max_iterations = 10  # Set the maximum number of iterations
svm_model = SVC(kernel='linear', max_iter=max_iterations)
svm_model.fit(X_train, y_train)

print(svm_model.score(X_test,y_test))
#accuracy
y_predict = svm_model.predict(X_test)
# print("y-predict:" + y_predict)

# Evaluate the model on test data
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# def confusionM(y_true,y_predict,target_names):
# #function for visualisation
#     cMatrix = confusion_matrix(y_true,y_predict)
#     df_cm = pd.DataFrame(cMatrix,index=target_names,columns=target_names)
#     plt.figure(figsize = (6,4))
#     cm = sns.heatmap(df_cm,annot=True,fmt="d")
#     cm.yaxis.set_ticklabels(cm.yaxis.get_ticklabels(),rotation=90)
#     cm.xaxis.set_ticklabels(cm.xaxis.get_ticklabels(),rotation=0)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
    
# confusionM(y_test,y_predict,image_labels)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Model has an score of 0.755 when done with train_unstructured.
# Model has an score of 0.68 when done with train_data.