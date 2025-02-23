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

#%%

# 1.- Code the target function and get the data
# Target function to approximate
def target_function(x):
    return torch.sin(2 * np.pi * x[:, 0]) * torch.cos(2 * np.pi * x[:, 1])

# Generate training data in R^2
num_samples = 500
x_train = 2 * torch.rand((num_samples, 2)) - 1  # Random points in [-1,1]²
y_train = target_function(x_train).view(-1, 1)  # Compute true function values

# 2.- Tent activation function (Fixed version)
class TentActivation(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon  # Small value to avoid vanishing gradients

    def forward(self, x):
        return torch.maximum(1 - torch.abs(x), torch.tensor(self.epsilon))  # Ensuring non-zero values

# 3.- Declare the neural network using the tent activation function.
class TentNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)   # Input layer (2D → 64 neurons)
        self.fc2 = nn.Linear(64, 64)  # Hidden Layer (64 → 64 neurons)
        self.fc3 = nn.Linear(64, 1)   # Output layer (64 → 1 neuron)
        self.tent = TentActivation()  # Tent activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.tent(x)  # Apply Tent Activation
        x = self.fc2(x)
        x = self.tent(x)  # Apply Tent Activation Again
        x = self.fc3(x)   # Final output layer
        return x

# 4.- Train the neural network
# Initialize the Tent Neural Network
net = TentNeuralNetwork()

# Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Training Loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    y_pred = net(x_train)  # Forward pass
    loss = criterion(y_pred, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 5.- Test
# Generate test grid
grid_size = 50
x1_test, x2_test = torch.meshgrid(torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size))
x_test = torch.stack([x1_test.flatten(), x2_test.flatten()], dim=1)

# Predict values using the trained neural network
y_pred = net(x_test).detach().numpy()
y_true = target_function(x_test).numpy()

# Reshape for plotting
y_pred = y_pred.reshape(grid_size, grid_size)
y_true = y_true.reshape(grid_size, grid_size)

# Plot True Function vs. Neural Network Approximation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# True Function
axes[0].imshow(y_true, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
axes[0].set_title("True Function f(x₁, x₂)")
axes[0].set_xlabel("x₁")
axes[0].set_ylabel("x₂")

# Neural Network Approximation
axes[1].imshow(y_pred, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
axes[1].set_title("Tent Neural Network Approximation")
axes[1].set_xlabel("x₁")
axes[1].set_ylabel("x₂")

plt.show()







