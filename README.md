# MAFA Dataset Exploration and Analysis

This repository contains a series of Python scripts and Jupyter notebooks for exploring, visualizing, and analyzing the MAFA dataset. The MAFA dataset includes images with various occluders and annotations, and this project demonstrates how to work with such data using different techniques, including visualization, data cleaning, augmentation, dimensionality reduction, and machine learning.

## Table of Contents

- [1. Dataset Exploration and Visualization](#1-dataset-exploration-and-visualization)
- [2. Data Preparation](#2-data-preparation)
- [3. Image Augmentation](#3-image-augmentation)
- [4. Dimensionality Reduction and Clustering](#4-dimensionality-reduction-and-clustering)
- [5. VGG16 Model for Classification](#5-vgg16-model-for-classification)
- [6. Logging and Duplicate Image Detection](#6-logging-and-duplicate-image-detection)
- [7. Additional Notes](#7-additional-notes)
- [8. Running the Project](#8-running-the-project)

## 1. Dataset Exploration and Visualization

This section involves loading and exploring the MAFA dataset, which consists of images with various annotations. The key tasks include:

- Loading .mat files containing the dataset annotations.
- Converting the annotations into a pandas DataFrame.
- Displaying images with bounding boxes for faces, occluders, glasses, and eye positions.

### Key Functions

- `display_image_with_features(row)`: Displays an image with annotated features.
- `show_random_faces()`: Shows random images from the dataset with annotations.
- `show_selected_faces(rows)`: Shows specific images from the dataset with annotations.

## 2. Data Preparation

This section covers the preparation of the dataset for further analysis:

- Converting training and test data from .mat files to CSV files.
- Ensuring that data is properly formatted and contains no missing values.
- Removing duplicates from the dataset.

### Key Functions

- `find_duplicate_images(image_dir)`: Finds and logs duplicate images based on perceptual hashes.

## 3. Image Augmentation

In this section, basic image augmentation techniques are demonstrated:

- Rotation
- Blurring
- Flipping
- Brightness adjustment

These augmentations are used to improve the robustness of models trained on the dataset.

### Key Functions

- `augment_and_plot(row)`: Applies and plots various augmentation techniques on an image.

## 4. Dimensionality Reduction and Clustering

This section involves dimensionality reduction and clustering of the dataset:

- Using PCA (Principal Component Analysis) to reduce dimensionality.
- Performing K-means clustering on the PCA-transformed data.

### Key Functions

- `PCA and K-means clustering`: Transform data using PCA and apply K-means clustering to the reduced data.

## 5. VGG16 Model for Classification

In this section, a VGG16-based deep learning model is used to classify images:

- Training a model to distinguish between masked and unmasked faces.
- Evaluating the model on test data and calculating performance metrics such as accuracy, precision, recall, and F1 score.

### Key Steps

- Loading and preprocessing images.
- Training a VGG16 model with custom classification layers.
- Evaluating the model and printing performance metrics.

## 6. Logging and Duplicate Image Detection

This section deals with identifying and logging duplicate images:

- Using perceptual hashing to find duplicate images.
- Logging duplicate images for further inspection.

### Key Functions

- `find_duplicate_images(image_dir)`: Detects and logs duplicate images based on perceptual hashes.

## 7. Additional Notes

- Ensure that paths to datasets and image directories are correctly specified in the code.
- The code assumes that the necessary Python libraries are installed and available in your environment.
- Adjust file paths and parameters as needed for your specific setup.

## 8. Running the Project

1. **Setup**:
   - Ensure all required libraries are installed: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`.
  
```python
pip install pandas numpy seaborn matplotlib scikit-learn
```

2. **Execution**:
   - Run the provided Jupyter Notebook to process data, generate visualizations, and use the recommendation function to find movie suggestions based on titles.

For any questions or further assistance, please feel free to reach out.

## Contributing

Contributions are welcome! You can improve the recommendation algorithm, enhance data visualizations, or add new features. Please submit pull requests for any changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

