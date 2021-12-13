# George Washington Neural Network
This repository contains a neural network library built from scratch for use in Neural Network courses at
The George Washington University. It is a basic implementation that provides a number of layers, activation
functions, and loss functions for students to explore how networks work at a low level. 

# Convolutional Neural Network
This branch adds CNN functionality to the GWU NN library for the final course project. Features implemented in 
this branch by Jonathan Lee. 

This branch includes code for the implementation of convolutional layers, pooling layers, and a flattening layer (implemented in `layers.py`). The new layers support forward and backward propagation as structured in the GWU NN. I trained a CNN binary classifier on the MNIST handwritten digits dataset and compared the model performance against a dense network. On a small set of input data, the CNN scored an 89% accuracy compared to the 79% accuracy of the Dense network.

## Running the Code
- `demo.ipynb` - run as a Jupyter Notebook. Contains the CNN and Dense network trained on MNIST dataset.
- `test.ipynb` - Contains some initial tests and visualizations of forward and backward propagation for the Convolutional layer.
- `one_hot_encoding.ipynb` - Contains my attempt at training a CNN multi-classifier on the 10 different digits. Did not work due to issues with the softmax activation function back propagation.


## Resources
- Deep Learning from Scratch with Python from First Principles by Seth Weidman, Chapter 5 CNNs
  - Explains the math of forward and backward propagation as well as some python code
