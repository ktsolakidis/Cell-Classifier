# Binary Image Classification models comparison
This repository aims to compare different binary classification models on an image dataset of cells.
the two cateogries of cell images are parasitized cells (infected), and healthy (uninfected) cells.

# Description

As i continue the exciting journey of machine learning, i started learning about Convolutional Neaural Networks and how the concept of convolution extracts information from an image.
Before I dive deep into CNNs, I decided to visualise how effective they are by applying them on a binary images classification task.

In this small project we will work with a dataset of images of infected (parasitized) and healthy cells. The images are of shape 64x64x3 (64 pixels width - 64 pixels height - 3 channels).

With this dataset we will try to train and test 4 different models and compare the training and testing / validation results.

1) Custom Logistic Regression from scratch.
    The first one is a binary logistic regression model which I have built from scratch. This means no library like scikit-learn was used, just pure nice oldschool math with numpy. We will only use the shuffle function from scikit-learn to help us "mix" our dataset. This is important because otherwise the images we are using are in order of "Healthy - uninfected cells, Parasitized - infected cells".

2) Scikit - learn Logistic Regression Model
    The second model will be Scikit-learn's Logistic Regression. It's not that different from our first model, we only use it to make sure that we get similar - if not the same - results with the previous model. We use it as a "check" to confirm that our previous from-scratch Logistic Regression model will produce similar results as this one.

3) Dense Neural Netork (Fully connected) 
    The third model is a Dense Neural Network (DNN), also known as a fully connected neural network.
    For this project, we will be using two hidden layers, both activated by the ReLU function. 
    The output layer uses a sigmoid activation function to produce probabilities for the binary classification. We will also add dropout layers to prevent overfitting. The goal here is to see how much improvement a simple neural network can offer over logistic regression, and aftet that, the difference between a Dense Neural Network, and a Convolutional Neural Network.

4) Convolutional Neural Network - CNN 
    The fourth model is a simple Convolutional Neural Network (CNN). CNNs are specifically designed for image-related tasks, as they extract spatial hierarchies of features through convolutional layers. In this project, the CNN uses two convolutional layers followed by max-pooling layers for down-sampling.Dropout is again added to prevent overfitting. This model is expected to outperform the others, as CNNs are particularly well-suited for tasks like this, where the data is image-based.

5) "Enhanced" CNN
    A similar CNN with the previous one, with some changes that aim to reduce our loss function and get our accuracy up. More specifically:

    Feature                         First CNN                   Second CNN

    Convolutional Layers                 2                           3
    Convolutions per block            Single               Double in the first block
    L2 Regularization              Yes (Conv layers)        Yes(Dense layer only)
    Dense Neurons                      128                         512
    Dropout Rates                    0.25,0.5                  0.25,0.3,0.5
    Padding                           Valid                       Same
    Learning Rate Adjustment            No                   ReduceLROnPlateau
            


Contact:

Konstantinos Tsolakidis
Machine Learning Engineer
Hatzakis Lab - University of Copenhagen
kontsolakidis25@gmail.com



