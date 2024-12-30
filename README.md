# Binary Image Classification Models Comparison

This repository aims to compare different binary classification models on an image dataset of cells.  
The two categories of cell images are parasitized cells (infected) and healthy (uninfected) cells.

## Description

As I continue the exciting journey of machine learning, I started learning about Convolutional Neural Networks (CNNs) and how the concept of convolution extracts information from an image.  
Before diving deep into CNNs, I decided to visualize how effective they are by applying them to a binary image classification task.

In this small project, we work with a dataset of images of infected (parasitized) and healthy cells. The images are of shape 64x64x3 (64 pixels width, 64 pixels height, 3 color channels).

We train and test five different models, comparing their training and testing/validation results:

### 1. Custom Logistic Regression (from scratch)

This model is built from scratch using numpy, without relying on libraries like scikit-learn (except for the shuffle function).  
The dataset is shuffled to avoid biases, as the images are ordered by class (healthy cells first, then infected cells).

### 2. Scikit-learn Logistic Regression Model

The second model uses Scikit-learn's Logistic Regression. This acts as a benchmark to ensure our custom implementation produces similar results.

### 3. Dense Neural Network (Fully Connected)

A Dense Neural Network (DNN) is also known as a fully connected neural network.  
- **Architecture**:  
  - Two hidden layers activated by the ReLU function.  
  - An output layer with a sigmoid activation function to generate probabilities.  
- **Features**:  
  - Dropout layers are added to prevent overfitting.  
  - The goal is to compare the performance improvement of a simple neural network over logistic regression and understand the difference between a Dense Neural Network and a Convolutional Neural Network.

### 4. Convolutional Neural Network (CNN)

A simple CNN designed specifically for image-related tasks.  
- **Architecture**:  
  - Two convolutional layers followed by max-pooling layers for down-sampling.  
  - Dropout is added to prevent overfitting.  
- CNNs are expected to outperform other models due to their ability to extract spatial features effectively.

### 5. Enhanced CNN

An advanced version of the previous CNN with improvements aimed at reducing the loss function and increasing accuracy.  
**Key Features**:  
- An additional convolutional layer.  
- Doubling the number of convolutions in the first block.  
- L2 regularization applied only to dense layers.  
- Increased neurons in dense layers (512 vs. 128).  
- Adjusted dropout rates for better regularization.  
- "Same" padding instead of "valid."  
- Learning rate adjustments using ReduceLROnPlate.

---

## Model Comparison: First CNN vs. Enhanced CNN

| Feature                    | First CNN                  | Enhanced CNN                     |
|----------------------------|----------------------------|-----------------------------------|
| **Convolutional Layers**   | 2                          | 3                                 |
| **Convolutions per Block** | Single                     | Double in the first block         |
| **L2 Regularization**      | Yes (Conv layers)          | Yes (Dense layer only)            |
| **Dense Neurons**          | 128                        | 512                               |
| **Dropout Rates**          | 0.25, 0.5                  | 0.25, 0.3, 0.5                   |
| **Padding**                | Valid                      | Same                              |
| **Learning Rate Adjustment** | No                       | ReduceLROnPlataeu                  |

---

## Contact

**Konstantinos Tsolakidis**  
Machine Learning Engineer  
Hatzakis Lab - University of Copenhagen  
📧 kontsolakidis25@gmail.com  
