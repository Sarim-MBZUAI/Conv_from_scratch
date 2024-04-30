# Convolutional Neural Network (CNN) from scratch

## Introduction

A Convolutional Neural Network (CNN) is a type of Feed-Forward Neural Network (FFNN) that employs convolution operations to significantly reduce the number of parameters in deep neural networks, thereby maintaining the quality of the model. CNNs are widely used in various applications such as image classification, object detection, semantic segmentation, image captioning, natural language processing, and more.

### What is a Convolution?

The core concept behind CNNs is the convolution operation. Understanding this operation is crucial to grasp the inspiration behind CNNs. For a more in-depth explanation, refer to the 3Blue1Brown video on convolution [here](https://youtu.be/KuXjwB4LzSA?si=iFf6zbMQY-0KC4RR).

<p align="center">
  <img src="https://github.com/Sarim-MBZUAI/Conv_from_scratch/blob/main/conv_image.png" alt=" Standard 2d CNN for image classification" width="75%"/>
  <br>
  <strong>Figure 1:</strong>  Standard 2d CNN for image classification
</p>

## CNN Structure

The structure of a CNN typically consists of the following components:

- **Convolutional Layer**: This is the primary layer that uses various filters to capture different patterns from the input. For example, in image processing, filters may detect edges, textures, or other significant features.

- **Resampling**: This includes both upsampling and downsampling to modify the spatial dimensions of the input data.

- **Activation Function**: Typically a ReLU function, this layer introduces non-linearities into the model, helping it learn more complex patterns.

- **Pooling Layer**: Reduces the dimensionality of each feature map while retaining the most important information.

- **Flatten Layer**: Converts the 2D feature maps into a 1D feature vector, preparing it for the final classification layer.

- **Classification Layer**: Usually consists of one or more dense layers followed by a softmax layer for classification.

### Additional Components

1. **Stride Configurations**: Adjusting the stride affects how densely a filter scans the input.

2. **Padding**: Adding padding to the input allows the convolutional filters more flexibility in covering the edge regions.

## Implementation Details

This project involves the practical implementation of various types of convolutional layers, resampling methods, and integration into a fully functional CNN model capable of processing both 1D and 2D data. Specific focus is given to manipulating the dimensions of data through operations like upsampling and downsampling, essential for building efficient and scalable CNNs.

## Usage

To use this CNN framework, import the necessary classes and instantiate the model with the desired configuration of layers and parameters. Train the model using a suitable dataset by adjusting the hyperparameters based on the complexity of the task and the computational resources available.

### Example Code

```python
from cnn_model import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)
