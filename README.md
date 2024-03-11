# Image Classification: Sedan vs Pickup Truck

This project focuses on image classification, specifically distinguishing between sedan and pickup truck images using machine learning techniques. The classification model is implemented in Python, leveraging the following libraries: pandas, numpy, matplotlib, scikit-learn (for Random Forest Classifier), and randomized search for hyperparameter tuning.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- opendatasets

Install the required packages using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn opendatasets
```

## Dataset

The dataset used for this project was obtained from Kaggle using the `opendatasets` library. The dataset likely contains a collection of labeled images, with each image belonging to either the sedan or pickup truck category.

To download the dataset, you can use the following code snippet:

```python
import opendatasets as od

# Kaggle dataset URL
dataset_url = 'your_kaggle_dataset_url_here'

# Download the dataset
od.download(dataset_url)
```

Replace `'your_kaggle_dataset_url_here'` with the actual URL of the dataset on Kaggle.

## Overview

**Data Loading and Preprocessing:**

- The project reads image data containing sedan and pickup truck images.
- It preprocesses the images and extracts relevant features.
- Utilizes pandas and numpy for data manipulation.

**Random Forest Model:**

- Implements a Random Forest Classifier for image classification.
- Utilizes scikit-learn for machine learning functionality.

**Hyperparameter Tuning:**

- Employs randomized search for hyperparameter tuning to enhance model performance.
- Fine-tunes Random Forest hyperparameters for optimal results.

**Model Evaluation:**

- Evaluates the model performance using various metrics such as accuracy, precision, recall, and F1-score.
- Plots the confusion matrix to visualize classification results.

**Prediction:**

- Demonstrates the model's ability to predict whether an input image represents a sedan or a pickup truck.
- Provides an example of using the trained model for predictions.

**Note:**

This readme provides an overview of the image classification project, highlighting key steps, techniques, and tools used. Ensure you have the required dependencies installed before running the code. Additionally, download the dataset from Kaggle using the provided code snippet before executing the project. Feel free to explore and customize the project based on your specific requirements and dataset.
