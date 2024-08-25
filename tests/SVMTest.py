import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
image_labels = pd.read_csv(r'C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Masked_Face\csv\MAFA_Training_Data_Unstructured.csv')

# Add mask_label column
image_labels['mask_label'] = np.where((image_labels['occluder_type'] == 1) | (image_labels['occluder_type'] == 2), 1, 0)

# Define image size
img_size = 100

# Define data augmentation transformations
data_generator = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.5],
    # contrast_range=[0.5, 1.5]
)

# Load and preprocess images
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize pixel values
    return img

# Extract handcrafted features (e.g., color histograms, texture features)
def extract_features(img):
    # Compute color histograms
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    # Concatenate color histograms into a single feature vector
    features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
    return features

# Load images, apply augmentations, and extract features
path_to_images = r"C:\Users\alexw\OneDrive\Documents\03_Education\University_Programming\Python\Big_Data\Coursework\Datasets\MAFA\MAFA-Label-Train\train-images"
X, y = [], []
for img_name in image_labels['imgName']:
    img_path = os.path.join(path_to_images, img_name)
    img = preprocess_image(img_path)
    img_label = image_labels.loc[image_labels['imgName'] == img_name, 'mask_label'].values[0]

    # Apply data augmentations
    augmented_images = data_generator.flow(np.expand_dims(img, axis=0), batch_size=1, shuffle=False)
    for augmented_img in augmented_images:
        X.append(extract_features(augmented_img[0]))
        y.append(img_label)

X = np.array(X)
y = np.array(y)

# Dimensionality reduction using PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X = pca.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model with a limit on the number of iterations
max_iterations = 100
svm_model = SVC(kernel='linear', max_iter=max_iterations)
svm_model.fit(X_train, y_train)

print(svm_model.score(X_test,y_test))
#accuracy
y_predict = svm_model.predict(X_test)
print(y_predict)

# Evaluate the model on test data
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

def confusionM(y_true,y_predict,target_names):
#function for visualisation
    cMatrix = confusion_matrix(y_true,y_predict)
    df_cm = pd.DataFrame(cMatrix,index=target_names,columns=target_names)
    plt.figure(figsize = (6,4))
    cm = sns.heatmap(df_cm,annot=True,fmt="d")
    cm.yaxis.set_ticklabels(cm.yaxis.get_ticklabels(),rotation=90)
    cm.xaxis.set_ticklabels(cm.xaxis.get_ticklabels(),rotation=0)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
confusionM(y_test,y_predict,image_labels)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')