#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:50:13 2025

@author: darios
"""

# Neural network to approximate a function on [-5,5]
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 1.- Define the function and get the set of inputs and outputs.
# Sine function.
def target_function(x):
    return torch.sin(x)

# Polynomial
def polynomial(x):
    return x**3 - 4*x**2 + x - 2


# Training data
x_train = torch.linspace(-5, 5, 100).view(-1, 1)  # Inputs
y_train = target_function(x_train)  # Outputs


# 2.- Declare the neural network
# Neural network
class FunctionApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential( # This allows us to declare each step
            nn.Linear(1, 64),  # Input layer (1 neuron) → Hidden layer (64 neurons)
            nn.ReLU(),         # Activation function (ReLU)
            nn.Linear(64, 64), # Hidden layer (64 neurons)
            nn.ReLU(),         # Activation function (ReLU)
            nn.Linear(64, 1)   # Output layer (1 neuron)
        )

    def forward(self, x):
        return self.model(x)

# Initialize the neural network
net = FunctionApproximator()


# 3.- Define how the model is optimized
# Define the loss function and optimizer
criterion = nn.MSELoss() # Median square error loss
optimizer = optim.Adam(net.parameters(), lr=0.01) # Adaptative Moment Estimation (Adam) is a powerful gradient descent optimization method.
# Parameters gives all of the trainable weights and biases of the affine transformations.


# 4.- Train the data
# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    y_pred = net(x_train)  # Forward pass
    loss = criterion(y_pred, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    if epoch % 100 == 0:  # Print progress
        print(f'Epoch {epoch}, Loss: {loss.item()}')


# 5.- Test
# Test the trained network
x_test = torch.linspace(-5, 5, 100).view(-1, 1)  # Test points
y_pred = net(x_test).detach().numpy()  # Predictions
y_true = target_function(x_test).numpy()  # True function values

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x_test.numpy(), y_true, label="True function", linestyle="dashed")
plt.plot(x_test.numpy(), y_pred, label="NN Approximation", linewidth=2)
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Neural Network Approximation of a Function")
plt.show()
