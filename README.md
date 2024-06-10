# EMOTION DETECTION USING DEEP LEARNING

## Introduction

This Model implements an emotion detection system using a convolutional neural network. It covers data preprocessing, model training, and evaluation, and employs OpenCV for real-time face detection and emotion classification from webcam feed. The model predicts six emotions: angry, fear, happy, neutral, sad, and surprise, drawing bounding boxes and annotations on detected faces.

## Datasets

please download dataset from [this link](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

## Algorithm 
The model uses the following algorithms and techniques:

 **Convolutional Neural Network (CNN)**: A deep learning algorithm that can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image, and differentiate one from the other. In this notebook, a CNN is used for emotion classification from facial images.

 **OpenCV's Haar Cascade Classifier**: A machine learning object detection method used to identify faces in the images. This is a pre-trained model provided by OpenCV to detect faces.

 **Batch Normalization**: This technique is used to normalize the inputs of each layer so that the network can converge faster and perform better. It is applied to the outputs of convolutional and dense layers.

 **Max Pooling**: A down-sampling strategy used in CNNs to reduce the dimensionality of feature maps and retain important features. It helps in reducing computational cost and controlling overfitting.

 **Dropout**: A regularization technique used to prevent overfitting in neural networks. It randomly drops a set of activations to zero during training.

 **Activation Functions (ReLU and Softmax)**:
   - **ReLU (Rectified Linear Unit)**: An activation function used in hidden layers of the CNN to introduce non-linearity.
   - **Softmax**: Used in the output layer to convert the raw model outputs into probabilities, which is useful for multi-class classification.

 **Adam Optimizer**: An optimization algorithm that adjusts the learning rate based on moments of the gradients, used to minimize the loss function during training.

 **Categorical Cross-Entropy Loss**: A loss function used for multi-class classification problems, which measures the performance of the classification model whose output is a probability value between 0 and 1.

 ## Dependencies
 - install python (3.10.9)
 - install numpy (1.26.4)
 - install pandas (2.2.2)
 - install tensorflow (2.16.1)
 - install keras (3.3.3)
 - install opencv-python (4.9.0.80)

## Results

![Screenshot 2024-06-10 023727](https://github.com/aditisolanki0313/emotion_detection/assets/143034653/bd290907-d0b2-4df4-9356-ce4b0a5d8a45)




![Screenshot 2024-06-10 023843](https://github.com/aditisolanki0313/emotion_detection/assets/143034653/c5530e04-32c6-4f48-a245-396ec4d72440)





