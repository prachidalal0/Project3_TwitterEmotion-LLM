# -*- coding: utf-8 -*-
"""Prachi_Dalal_HW_3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1I0x_iKTjHZ7HJx0CT6FVovNsvHu-vh-b

# <font color = 'indianred'>**HW3 - 20 Points** </font>
- **You have to submit two files for this part of the HW**
  >(1) ipynb (colab notebook) and<br>
  >(2) pdf file (pdf version of the colab file).**
- **Files should be named as follows**:
>FirstName_LastName_HW_3**

# <font color = 'indianred'>**Task 1 - Autodiff - 5 Points**
"""

import torch
import torch.nn as nn

"""##  <font color = 'indianred'>**Q1 -Normalize Function (1 Points)**<font>

Write the function that normalizes the columns of a matrix. You have to compute the mean and standard deviation of each column. Then for each element of the column, you subtract the mean and divide by the standard deviation.
"""

# Given Data
x = [[ 3,  60,  100, -100],
     [ 2,  20,  600, -600],
     [-5,  50,  900, -900]]

# Convert to PyTorch Tensor and set to float
X = torch.tensor(x)
X= X.float()

# Print shape and data type for verification
print(X.shape)
print(X.dtype)

# Compute and display the mean and standard deviation of each column for reference
X.mean(axis = 0)
X.std(axis = 0)

X.std(axis = 0)

"""- Your task starts here
- Your normalize_matrix function should take a PyTorch tensor x as input.
- It should return a tensor where the columns are normalized.
- After implementing your function, use the code provided to verify if the mean for each column in Z is close to zero and the standard deviation is 1.
"""

def normalize_matrix(X):
  # Calculate the mean along each column (think carefully , you will take mean along axis = 0 or 1)
  mean = X.mean(axis = 0)

  # Calculate the standard deviation along each column
  std = X.std(axis = 0)

  # Normalize each element in the columns by subtracting the mean and dividing by the standard deviation
  y = (X - mean) / std

  return y  # Return the normalized matrix

Z = normalize_matrix(X)
Z

Z.mean(axis = 0)

Z.std(axis = 0)

"""##  <font color = 'indianred'>**Q2 -Calculate Gradients 1.5 Point**

Compute Gradient using  PyTorch Autograd - 2 Points
## $f(x,y) = \frac{x + \exp(y)}{\log(x) + (x-y)^3}$
Compute dx and dy at x=3 and y=4
"""

def fxy(x, y):
  # Calculate the numerator: Add x to the exponential of y
  num = x + torch.exp(y)

  # Calculate the denominator: Sum of the logarithm of x and cube of the difference between x and y
  den = torch.log(x) + (x - y)**3

  result = num / den

  # Perform element-wise division of the numerator by the denominator
  return result

# Create a single-element tensor 'x' containing the value 3.0
# make sure to set 'requires_grad=True' as you want to compute gradients with respect to this tensor during backpropagation
x = torch.tensor(3.0, requires_grad=True)

# Create a single-element tensor 'y' containing the value 4.0
# Similar to 'x', we want to compute gradients for 'y' during backpropagation, hence make sure to set 'requires_grad=True'
y = torch.tensor(4.0, requires_grad=True)

# Call the function 'fxy' with the tensors 'x' and 'y' as arguments
# The result 'f' will also be a tensor and will contain derivative information because 'x' and 'y' have 'requires_grad=True'
f = fxy(x, y)
f

# Perform backpropagation to compute the gradients of 'f' with respect to 'x' and 'y'
# Hint use backward() function on f

# CODE HERE
f.backward()

# Display the computed gradients of 'f' with respect to 'x' and 'y'
# These gradients are stored as attributes of x and y after the backward operation
# Print the gradients for x and y
dx = x.grad.item()
dy = y.grad.item()

print(f"x.grad at x=3, y=4: {dx}")
print(f"y.grad at x=3, y=4: {dy}")

"""## <font color = 'indianred'>**Q6. Numerical Precision - 2.5 Points**

Given scalars `x` and `y`, implement the following `log_exp` function such that it returns
$$-\log\left(\frac{e^x}{e^x+e^y}\right)$$.
"""

#Question
def log_exp(x, y):
    ## add your solution here and remove pass
    # CODE HERE

    numerator = torch.exp(x)

    denominator = torch.exp(x) + torch.exp(y)

    ratio = numerator / denominator

    result = -torch.log(ratio)

    return result

"""Test your codes with normal inputs:"""

# Create tensors x and y with initial values 2.0 and 3.0, respectively
x, y = torch.tensor([2.0]), torch.tensor([3.0])

# Evaluate the function log_exp() for the given x and y, and store the output in z
z = log_exp(x, y)

# Display the computed value of z
z

"""Now implement a function to compute $\partial z/\partial x$ and $\partial z/\partial y$ with `autograd`"""

def grad(forward_func, x, y):
  # Enable gradient tracking for x and y, set reauires_grad appropraitely
  x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
  y = torch.tensor(y, dtype=torch.float32, requires_grad=True)

  # Evaluate the forward function to get the output 'z'
  z = forward_func(x, y)

  # Perform the backward pass to compute gradients
  # Hint use backward() function on z
  z.backward()

  # Print the gradients for x and y
  print('x.grad =', x.grad.item())
  print('y.grad =', y.grad.item())

  # Reset the gradients for x and y to zero for the next iteration
  x.grad.zero_()
  y.grad.zero_()

"""Test your codes, it should print the results nicely."""

grad(log_exp, x, y)

"""But now let's try some "hard" inputs"""

x, y = torch.tensor([50.0]), torch.tensor([100.0])

# you may see nan/inf values as output, this is not an error
grad(log_exp, x, y)

# you may see nan/inf values as output, this is not an error
torch.exp(torch.tensor([100.0]))

"""Does your code return correct results? If not, try to understand the reason. (Hint, evaluate `exp(100)`). Now develop a new function `stable_log_exp` that is identical to `log_exp` in math, but returns a more numerical stable result.
<br> Hint: (1) $\log\left(\frac{x}{y}\right) = log ({x}) -log({y})$
<br> Hint: (2) See logsum Trick - https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
"""

def stable_log_exp(x, y):

    max_val = torch.max(x, y)
    result = max_val + torch.log(torch.exp(x - max_val) + torch.exp(y - max_val))
    return -result

log_exp(x, y)

stable_log_exp(x, y)

grad(stable_log_exp, x, y)

"""# <font color = 'indianred'>**Task 2 - Linear Regression using Batch Gradient Descent with PyTorch- 5 Points**

# <font color = 'indianred'>**Regression using Pytroch**</font>

Imagine that you're trying to figure out relationship between two variables x and y . You have some idea but you aren't quite sure yet whether the dependence is linear or quadratic.

Your goal is to use least mean squares regression to identify the coefficients for the following three models. The three models are:

1. Quadratic model where $\mathrm{y} = b + w_1 \cdot \mathrm{x} + w_2 \cdot \mathrm{x}^2$.
1. Linear model where $\mathrm{y} = b + w_1 \cdot \mathrm{x}$.
1. Linear model with no bias  where $\mathrm{y} = w_1 \cdot \mathrm{x}$.

- You will use <font color = 'indianred'>**Batch gradient descent to estimate the model co-efficients.Batch gradient descent uses complete training data at each iteration.**</font>
- You will implement only training loop (no splitting of data in to training/validation).
- The training loop will have only one ```for loop```. We need to iterate over whole data in each epoch. We do not need to create batches.
- You may have to try different values of number of epochs/ learning rate to get good results.
- You should use  Pytorch's nn.module and functions.

## <font color = 'indianred'> **Data**
"""

x = torch.tensor([1.5420291, 1.8935232, 2.1603365, 2.5381863, 2.893443, \
                    3.838855, 3.925425, 4.2233696, 4.235571, 4.273397, \
                    4.9332876, 6.4704757, 6.517571, 6.87826, 7.0009003, \
                    7.035741, 7.278681, 7.7561755, 9.121138, 9.728281])
y = torch.tensor([63.802246, 80.036026, 91.4903, 108.28776, 122.781975, \
                    161.36314, 166.50816, 176.16772, 180.29395, 179.09758, \
                    206.21027, 272.71857, 272.24033, 289.54745, 293.8488, \
                    295.2281, 306.62274, 327.93243, 383.16296, 408.65967])

# Reshape the y tensor to have shape (n, 1), where n is the number of samples.
# This is done to match the expected input shape for PyTorch's loss functions.
y = y.view(-1, 1)

# Reshape the x tensor to have shape (n, 1), similar to y, for consistency and to work with matrix operations.
x = x.view(-1, 1)

# Compute the square of each element in x.
# This may be used for polynomial features in regression models.
x2 = x * x

# Concatenate the original x tensor and its squared values (x2) along dimension 1 (columns).
# This creates a new tensor with two features: the original x and x2 (its square) . This can be useful for polynomial regression.
x_combined = torch.cat((x, x2), 1)

print(x_combined.shape, x.shape)

"""##<font color = 'indianred'>**Loss Function**"""

# Initialize Mean Squared Error (MSE) loss function with mean reduction
# 'reduction="mean"' averages the squared differences between predicted and target values
loss_function = nn.MSELoss(reduction='mean')

"""## <font color = 'indianred'> **Train Function**"""

def train(epochs, x, y, loss_function, log_interval, model, optimizer):
    """
    Train a PyTorch model using gradient descent.

    Parameters:
    epochs (int): The number of training epochs.
    x (torch.Tensor): The input features.
    y (torch.Tensor): The ground truth labels.
    loss_function (torch.nn.Module): The loss function to be minimized.
    log_interval (int): The interval at which training information is logged.
    model (torch.nn.Module): The PyTorch model to be trained.
    optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.

    Side Effects:
    - Modifies the input model's internal parameters during training.
    - Outputs training log information at specified intervals.
    """


    for epoch in range(epochs):

        # Step 1: Forward pass - Compute predictions based on the input features
        y_hat = model(x)

        # Step 2: Compute Loss
        loss = loss_function(y_hat, y)

        # Step 3: Zero Gradients - Clear previous gradient information to prevent accumulation
        optimizer.zero_grad()

        # Step 4: Calculate Gradients - Backpropagate the error to compute gradients for each parameter
        loss.backward()

        # Step 5: Update Model Parameters - Adjust weights based on computed gradients
        optimizer.step()

        # Log training information at specified intervals
        if epoch % log_interval == 0:
            print(f'epoch: {epoch + 1} --> loss {loss.item()}')

"""## <font color = 'indianred'> **Part 1**

-  <font color = 'indianred'>**For Part 1, use x_combined (we need to use both $x$ and $x^2 $) as input to the model, this means that you have two inputs.**</font>
- Use `nn.Linear` function to specify the model, <font color = 'indianred'>**think carefully what values the three arguments ```(n_ins, n_outs, bias)``` will take**.</font>.
- In PyTorch, the `nn.Linear` layer initializes its weights using Kaiming (He) initialization by default, which is well-suited for ReLU activation functions. The bias terms are initialized to zero.
-  In this assignment you will  use `nn.init` functions like `nn.init.normal_` and `nn.init.zeros_`, to explicitly override these default initializations to use your specified methods.


**Run the cell below twice**

**In the first attempt**
- Use LEARNING_RATE = 0.05
What do you observe?

Write your observations HERE:

**In the second attempt**
- Now use a LEARNING_RATE  = 0.0005,
What do you observe?

Write your observations HERE:

"""

# model 1
LEARNING_RATE = 0.0005
EPOCHS = 100000
LOG_INTERVAL= 10000

# Use PyTorch's nn.Linear to create the model for your task.
# Based on your understanding of the problem at hand, decide how you will initialize the nn.Linear layer.
# Take into consideration the number of input features, the number of output features, and whether or not to include a bias term.
class LinearRegressionPart1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_layer = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        return self.linear_layer(x)

model_part1 = LinearRegressionPart1(input_dim=2, output_dim=1)

# Initialize the weights of the model using a normal distribution with mean = 0 and std = 0.01
# Hint: To initialize the model's weights, you can use the nn.init.normal_() function.
# You will need to provide the 'model.weight' tensor and specify values for the 'mean' and 'std' arguments.
nn.init.normal_(model_part1.linear_layer.weight, mean=0, std=0.01) # Kaiming (He) initialization

# Initialize the model's bias terms to zero
# Hint: To set the model's bias terms to zero, consider using the nn.init.zeros_() function.
# You'll need to supply 'model.bias' as an argument.
nn.init.zeros_(model_part1.linear_layer.bias)

# Create an SGD (Stochastic Gradient Descent) optimizer using the model's parameters and a predefined learning rate
optimizer = torch.optim.SGD(model_part1.parameters(), lr=LEARNING_RATE)

# Start the training process for the model with specified parameters and settings
train(EPOCHS, x_combined, y, loss_function, LOG_INTERVAL, model_part1, optimizer)

print(f'Weights: {model_part1.linear_layer.weight.data}, \nBias: {model_part1.linear_layer.bias.data}')

"""## <font color = 'indianred'> **Part 2**

-  <font color = 'indianred'>**For Part 2, use $x$ as input to the model, this means that you have only one input.**</font>
- Use `nn.Linear` to specify the model, <font color = 'indianred'>**think carefully what values the three arguments ```(n_ins, n_outs, bias)``` will take**.</font>.

"""

# model 2
LEARNING_RATE = 0.01
EPOCHS = 1000
LOG_INTERVAL= 10

# Use PyTorch's nn.Linear to create the model for your task.
# Based on your understanding of the problem at hand, decide how you will initialize the nn.Linear layer.
# Take into consideration the number of input features, the number of output features, and whether or not to include a bias term.
class LinearRegressionPart2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_layer(x)


model_part2 = LinearRegressionPart2(input_dim=1, output_dim=1)

# Initialize the weights of the model using a normal distribution with mean = 0 and std = 0.01
# Hint: To initialize the model's weights, you can use the torch.nn.init.normal_() function.
# You will need to provide the 'model.weight' tensor and specify values for the 'mean' and 'std' arguments.
nn.init.normal_(model_part2.linear_layer.weight, mean=0, std=0.01)


# Initialize the model's bias terms to zero
# Hint: To set the model's bias terms to zero, consider using the nn.init.zeros_() function.
# You'll need to supply 'model.bias' as an argument.
nn.init.zeros_(model_part2.linear_layer.bias)

# Create an SGD (Stochastic Gradient Descent) optimizer using the model's parameters and a predefined learning rate
optimizer = torch.optim.SGD(model_part2.parameters(), lr=LEARNING_RATE)

# Start the training process for the model with specified parameters and settings
# Note that we are passing x as an input for this part
train(EPOCHS, x, y, loss_function, LOG_INTERVAL, model_part2, optimizer)

print(f' Weights {model_part2.linear_layer.weight.data}, \nBias: {model_part2.linear_layer.bias.data}')

"""## <font color = 'indianred'> **Part 3**
-  <font color = 'indianred'>**Part 3 is similar to part 2, the only difference is that model has no bias term now.**</font>
- **You will see that we are now running the model for only ten epochs and will get similar results**
"""

# model 3
LEARNING_RATE = 0.01
EPOCHS = 10
LOG_INTERVAL= 1

# Use PyTorch's nn.Linear to create the model for your task.
# Based on your understanding of the problem at hand, decide how you will initialize the nn.Linear layer.
# Take into consideration the number of input features, the number of output features, and whether or not to include a bias term.
class LinearRegressionPart3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_layer = nn.Linear(input_dim, output_dim, bias=False)  # Set bias to False

    def forward(self, x):
        return self.linear_layer(x)

model_part3 = LinearRegressionPart3(input_dim=1, output_dim=1)

# Initialize the weights of the model using a normal distribution with mean = 0 and std = 0.01
# Hint: To initialize the model's weights, you can use the nn.init.normal_() function.
# You will need to provide the 'model.weight' tensor and specify values for the 'mean' and 'std' arguments.
nn.init.normal_(model_part3.linear_layer.weight, mean=0, std=0.01)

# We do not need to initilaize the bias term as there is no bias term in this model

# Create an SGD (Stochastic Gradient Descent) optimizer using the model's parameters and a predefined learning rate
optimizer = torch.optim.SGD(model_part3.parameters(), lr=LEARNING_RATE)


# Start the training process for the model with specified parameters and settings
# Note that we are passing x as an input for this part
train(EPOCHS, x, y, loss_function, LOG_INTERVAL, model_part3, optimizer)

print(f' Weights {model_part3.linear_layer.weight.data}')

"""# <font color = 'indianred'>**Task 3 - MultiClass Classification using Mini Batch Gradient Descent with PyTorch- 5 Points**

- You will implement only training loop (no splitting of data in to training/validation).
- We will use minibatch Gradient Descent - Hence we will have two for loops in his case.
- You should use  Pytorch's nn.module and functions.

## <font color = 'indianred'>**Data**
"""

# Import the make_classification function from the sklearn.datasets module
# This function is used to generate a synthetic dataset for classification tasks.
from sklearn.datasets import make_classification

# Import the StandardScaler class from the sklearn.preprocessing module
# StandardScaler is used to standardize the features by removing the mean and scaling to unit variance.
from sklearn.preprocessing import StandardScaler

# Import the main PyTorch library, which provides the essential building blocks for constructing neural networks.
import torch

# Import the 'optim' module from PyTorch for various optimization algorithms like SGD, Adam, etc.
import torch.optim as optim

# Import the 'nn' module from PyTorch, which contains pre-defined layers, loss functions, etc., for neural networks.
import torch.nn as nn

# Import the 'functional' module from PyTorch; incorrect import here, it should be 'import torch.nn.functional as F'
# This module contains functional forms of layers, loss functions, and other operations.
import torch.functional as F  # Should be 'import torch.nn.functional as F'

# Import DataLoader and Dataset classes from PyTorch's utility library.
# DataLoader helps with batching, shuffling, and loading data in parallel.
# Dataset provides an abstract interface for easier data manipulation.
from torch.utils.data import DataLoader, Dataset

# Generate a synthetic dataset for classification using make_classification function.
# Parameters:
# - n_samples=1000: The total number of samples in the generated dataset.
# - n_features=5: The total number of features for each sample.
# - n_classes=3: The number of classes for the classification task.
# - n_informative=4: The number of informative features, i.e., features that are actually useful for classification.
# - n_redundant=1: The number of redundant features, i.e., features that can be linearly derived from informative features.
# - random_state=0: The seed for the random number generator to ensure reproducibility.

X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_informative=4, n_redundant=1, random_state=0)

"""
In this example, you're using `make_classification` to <font color = 'indianred'>**generate a dataset with 1,000 samples, 5 features per sample, and 3 classes for the classification problem**.</font> Of the 5 features, 4 are informative (useful for classification), and 1 is redundant (can be derived from the informative features). The `random_state` parameter ensures that the data generation is reproducible."""

# Initialize the StandardScaler object from the sklearn.preprocessing module.
# This will be used to standardize the features of the dataset.
preprocessor = StandardScaler()

# Fit the StandardScaler on the dataset (X) and then transform it.
# The fit_transform() method computes the mean and standard deviation of each feature,
# and then standardizes the features by subtracting the mean and dividing by the standard deviation.
X = preprocessor.fit_transform(X)

print(X.shape, y.shape)

X[0:5]

print(y[0:10])

"""## <font color = 'indianred'>**Dataset and Data Loaders**"""

# Convert the numpy arrays X and y to PyTorch Tensors.
# For X, we create a floating-point tensor since most PyTorch models expect float inputs for features.
# This is a  multiclass classification problem.

# ================================
# IMPORTANT: # Consider what cost function you will use and whether it expects the label tensor (y)  to be float or long type.
# ================================

x_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Define a custom PyTorch Dataset class for handling our data
class MyDataset(Dataset):
    # Constructor: Initialize the dataset with features and labels
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    # Method to return the length of the dataset
    def __len__(self):
        return self.labels.shape[0]

    # Method to get a data point by index
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

# Create an instance of the custom MyDataset class, passing in the feature and label tensors.
# This will allow the data to be used with PyTorch's DataLoader for efficient batch processing.
train_dataset = MyDataset(x_tensor, y_tensor)

# Access the first element (feature-label pair) from the train_dataset using indexing.
# The __getitem__ method of MyDataset class will be called to return this element.
train_dataset[0]

# Create Data loader from Dataset
# Use a batch size of 16
# Use shuffle = True
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

"""## <font color = 'indianred'>**Model**"""

# Student Task: Define your neural network model for multi-class classification.
# Think through what layers you should add. Note: Your task is to create a model that uses Softmax for
# classification but doesn't include any hidden layers.
# You can use nn.Linear or nn.Sequential for this task

num_features = 5
num_classes = 3

# Define your neural network model for multi-class classification
model = nn.Sequential(
    nn.Linear(in_features=num_features, out_features=num_classes),
    nn.Softmax(dim=1)
)

"""## <font color = 'indianred'>**Loss Function**"""

# Student Task: Specify the loss function for your model.
# Consider the architecture of your model, especially the last layer, when choosing the loss function.
# Reminder: The last layer in the previous step should guide your choice for an appropriate loss function for multi-class classification.

loss_function = nn.CrossEntropyLoss()

"""## <font color = 'indianred'>**Initialization**

Create a function to initilaize weights.
- Initialize weights using normal distribution with mean = 0 and std = 0.05
- Initilaize the bias term with zeros
"""

# Function to initialize the weights and biases of a neural network layer.
# This function specifically targets layers of type nn.Linear.
def init_weights(layer):
  # Check if the layer is of the type nn.Linear.
  if type(layer) == nn.Linear:
    # Initialize the weights with a normal distribution, centered at 0 with a standard deviation of 0.05.
    torch.nn.init.normal_(layer.weight, mean=0, std=0.05)
    # Initialize the bias terms to zero.
    torch.nn.init.zeros_(layer.bias)

"""## <font color = 'indianred'>**Training Loop**

**Model Training** involves five steps:

- Step 0: Randomly initialize parameters / weights
- Step 1: Compute model's predictions - forward pass
- Step 2: Compute loss
- Step 3: Compute the gradients
- Step 4: Update the parameters
- Step 5: Repeat steps 1 - 4

Model training is repeating this process over and over, for many **epochs**.

We will specify number of ***epochs*** and during each epoch we will iterate over the complete dataset and will keep on updating the parameters.

***Learning rate*** and ***epochs*** are known as hyperparameters. We have to adjust the values of these two based on validation dataset.

We will now create functions for step 1 to 4.
"""

# Function to train a neural network model.
# Arguments include the number of epochs, loss function, learning rate, model architecture, and optimizer.

def train(epochs, loss_function, learning_rate, model, optimizer):

  # Loop through each epoch
  for epoch in range(epochs):

    # Initialize variables to hold aggregated training loss and correct prediction count for each epoch
    running_train_loss = 0
    running_train_correct = 0

    # Loop through each batch in the training dataset using train_loader
    for x, y in train_loader:

      # Move input and target tensors to the device (GPU or CPU)
      x = x.to(device)
      targets = y.to(device)

      # Step 1: Forward Pass: Compute model's predictions
      output = model(x)

      # Step 2: Compute loss
      loss = loss_function(output, targets)

      # Step 3: Backward pass - Compute the gradients
      # Zero out gradients from the previous iteration
      optimizer.zero_grad()

      # Backward pass: Compute gradients based on the loss
      loss.backward()

      # Step 4: Update the parameters
      optimizer.step()

      # Accumulate the loss for the batch
      running_train_loss += loss.item()

      # Evaluate model's performance without backpropagation for efficiency
      # `with torch.no_grad()` temporarily disables autograd, improving speed and avoiding side effects during evaluation.
      with torch.no_grad():
          _, y_pred = torch.max(output, 1)  # Find the class index with the maximum predicted probability
          correct = torch.sum(y_pred == targets)  # Compute the number of correct predictions in the batch
          running_train_correct += correct  # Update the cumulative count of correct predictions for the current epoch


    # Compute average training loss and accuracy for the epoch
    train_loss = running_train_loss / len(train_loader)
    train_acc = running_train_correct / len(train_loader.dataset)

    # Display training loss and accuracy metrics for the current epoch
    print(f'Epoch : {epoch + 1} / {epochs}')
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc * 100:.4f}%')

# Fix the random seed to ensure reproducibility across runs
torch.manual_seed(100)

# Define the total number of epochs for which the model will be trained
epochs = 5

# Detect if a GPU is available and use it; otherwise, use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)  # Output the device being used

# Define the learning rate for optimization; consider its impact on model performance
learning_rate = 1

# Student Task: Configure the optimizer for model training.
# Here, we're using Stochastic Gradient Descent (SGD). Think through what parameters are needed.
# Reminder: Utilize the learning rate defined above when setting up your optimizer.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Relocate the model to the appropriate compute device (GPU or CPU)
model.to(device)

# Apply custom weight initialization; this can affect the model's learning trajectory
# The `apply` function recursively applies a function to each submodule in a PyTorch model.
# In the given context, it's used to apply the `init_weights` function to initialize the weights of all layers in the model.
# The benefit is that it provides a convenient way to systematically apply custom weight initialization across complex models,
# potentially improving model convergence and performance.
model.apply(init_weights)

# Kick off the training process using the specified settings
train(epochs, loss_function, learning_rate, model, optimizer)

# Output the learned parameters (weights and biases) of the model after training
for name, param in model.named_parameters():
  # Print the name and the values of each parameter
  print(name, param.data)

"""# <font color = 'indianred'>**Task 4 - MultiLabel Classification using Mini Batch Gradient Descent with PyTorch- 5 Points**

- You will implement only training loop (no splitting of data in to training/validation).
- We will use minibatch Gradient Descent - Hence we will have two for loops in his case.
- You should use  Pytorch's nn.module and functions.

## <font color = 'indianred'>**Data**
"""

# Import the function to generate a synthetic multilabel classification dataset
from sklearn.datasets import make_multilabel_classification

# Import the StandardScaler for feature normalization
from sklearn.preprocessing import StandardScaler

# Import PyTorch library for tensor computation and neural network modules
import torch

# Import PyTorch's optimization algorithms package
import torch.optim as optim

# Import PyTorch's neural network module for defining layers and models
import torch.nn as nn

# Import PyTorch's functional API for stateless operations
import torch.functional as F

# Import DataLoader, TensorDataset, and Dataset for data loading and manipulation
from torch.utils.data import DataLoader, TensorDataset, Dataset

X, y = make_multilabel_classification(n_samples=1000, n_features=5, n_classes=3, n_labels=2, random_state=0)

# Initialize the StandardScaler for feature normalization
preprocessor = StandardScaler()

# Fit the preprocessor to the data and transform the features for zero mean and unit variance
X = preprocessor.fit_transform(X)

# Print the shape of the feature matrix X and the label matrix y
# Students: Pay attention to these shapes as they will guide you in defining your neural network model
print(X.shape, y.shape)

X[0:5]

print(y[0:10])

"""## <font color = 'indianred'>**Dataset and Data Loaders**"""

# Student Task: Create Tensors from the numpy arrays.
# Earlier, we focused on multiclass classification; now, we are dealing with multilabel classification.

# ================================
# IMPORTANT: # Consider what cost function you will use for multilabel classification and whether it expects the label tensor (y) to be float or long type.
# ================================

x_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Define a custom PyTorch Dataset class for handling our data
class MyDataset(Dataset):
    # Constructor: Initialize the dataset with features and labels
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    # Method to return the length of the dataset
    def __len__(self):
        return self.labels.shape[0]

    # Method to get a data point by index
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

# Initialize an instance of the custom MyDataset class
# This will be our training dataset, holding our features and labels as PyTorch tensors
train_dataset = MyDataset(x_tensor, y_tensor)

# Access the first element (feature-label pair) from the train_dataset using indexing.
# The __getitem__ method of MyDataset class will be called to return this element.
# This is useful for debugging and understanding the data structure
train_dataset[0]

# Create Data lOader from Dataset
# Use a batch size of 16
# Use shuffle = True
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

"""## <font color = 'indianred'>**Model**"""

model = nn.Sequential(
    nn.Linear(in_features=num_features, out_features=num_classes),
    nn.Sigmoid())

"""## <font color = 'indianred'>**Loss Function**"""

# Student Task: Specify the loss function for your model.
# Consider the architecture of your model, especially the last layer, when choosing the loss function.
# This is a multilabel problem, so make sure your choice reflects that.

loss_function = nn.BCELoss()

"""## <font color = 'indianred'>**Initialization**

Create a function to initilaize weights.
- Initialize weights using normal distribution with mean = 0 and std = 0.05
- Initilaize the bias term with zeros
"""

# Function to initialize the weights and biases of the model's layers
# This is provided to you and is not a student task
def init_weights(layer):
  # Check if the layer is a Linear layer
  if type(layer) == nn.Linear:
    # Initialize the weights with a normal distribution, mean=0, std=0.05
    torch.nn.init.normal_(layer.weight, mean = 0, std = 0.05)
    # Initialize the bias terms to zero
    torch.nn.init.zeros_(layer.bias)

"""## <font color = 'indianred'>**Training Loop**

**Model Training** involves five steps:

- Step 0: Randomly initialize parameters / weights
- Step 1: Compute model's predictions - forward pass
- Step 2: Compute loss
- Step 3: Compute the gradients
- Step 4: Update the parameters
- Step 5: Repeat steps 1 - 4

Model training is repeating this process over and over, for many **epochs**.

We will specify number of ***epochs*** and during each epoch we will iterate over the complete dataset and will keep on updating the parameters.

***Learning rate*** and ***epochs*** are known as hyperparameters. We have to adjust the values of these two based on validation dataset.

We will now create functions for step 1 to 4.
"""

# Install the torchmetrics package, a PyTorch library for various machine learning metrics,
# to facilitate model evaluation during and after training.
!pip install torchmetrics

# Import HammingDistance from torchmetrics
# HammingDistance is useful for evaluating multi-label classification problems.
from torchmetrics import HammingDistance

"""<font color = 'indianred'>**Hamming Distance**</font> is often used in multi-label classification problems to quantify the dissimilarity between the predicted and true labels. It does this by measuring the number of label positions where predicted and true labels differ for each sample. It is a useful metric because it offers a granular level of understanding of the discrepancies between the predicted and actual labels, taking into account each label in a multi-label setting.

<font color = 'indianred'>**Unlike accuracy, which is all-or-nothing, Hamming Distance can give partial credit by considering the labels that were correctly classified** </font>, thereby providing a more granular insight into the model's performance.

Let us understand this with an example:
"""

target = torch.tensor([[0, 1], [1, 1]])
preds = torch.tensor([[0, 1], [0, 1]])
hamming_distance = HammingDistance(task="multilabel", num_labels=2)
hamming_distance(preds, target)

"""In the given example, the Hamming Distance is calculated for multi-label classification with two labels (0 and 1).

1. The target tensor has shape (2, 2): `[[0, 1], [1, 1]]`
2. The prediction tensor also has shape (2, 2): `[[0, 1], [0, 1]]`

Let's examine the individual sample pairs to understand the distance:

- For the first sample pair (target = `[0, 1]`, prediction = `[0, 1]`), the Hamming Distance is 0 because the prediction is accurate.
- For the second sample pair (target = `[1, 1]`, prediction = `[0, 1]`), the Hamming Distance is 1 for the first label (predicted 0, true label 1).

To calculate the overall Hamming Distance, we can take the number of label mismatches and divide by the total number of labels:

- Total Mismatches = 1 (from the second sample pair)
- Total Number of Labels = 2 samples * 2 labels per sample = 4

Therefore, the overall Hamming Distance is \(1 / 4 = 0.25\), which matches the output `tensor(0.2500)`.

Hamming Distance is a good metric for multi-label classification as it can capture the difference between sets of labels per sample, thereby providing a more granular measure of the model's performance.
"""

def train(epochs, loss_function, learning_rate, model, optimizer, train_loader, device):

    train_hamming_distance = HammingDistance(task="multilabel", num_labels=3).to(device)

    for epoch in range(epochs):
        # Initialize train_loss at the start of the epoch
        running_train_loss = 0.0

        # Iterate on batches from the dataset using train_loader
        for x, y in train_loader:
            # Move inputs and outputs to GPUs
            x =  x.to(device)
            targets =  y.to(device)

            # Step 1: Forward Pass: Compute model's predictions
            output = model(x)

            # Step 2: Compute loss
            loss =  loss_function(output, targets)

            # Step 3: Backward pass - Compute the gradients
            # Zero out gradients from the previous iteration
            optimizer.zero_grad()

            # Backward pass: Compute gradients based on the loss
            loss.backward()

            # Step 4: Update the parameters
            optimizer.step()

            # Update running loss
            running_train_loss += loss.item()

            with torch.no_grad():
                # Correct prediction using thresholding
                y_pred = (output > 0.5).float()

                # Update Hamming Distance metric
                train_hamming_distance.update(y_pred, targets)

        # Compute mean train loss for the epoch
        train_loss = running_train_loss / len(train_loader)

        # Compute Hamming Distance for the epoch
        epoch_hamming_distance = train_hamming_distance.compute()

        # Print the train loss and Hamming Distance for the epoch
        print(f'Epoch: {epoch + 1} / {epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Hamming Distance: {epoch_hamming_distance:.4f}')

        # Reset metric states for the next epoch
        train_hamming_distance.reset()

# Set a manual seed for reproducibility across runs
torch.manual_seed(100)

# Define hyperparameters: learning rate and the number of epochs
learning_rate = 1
epochs = 20

# Determine the computing device (GPU if available, otherwise CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Student Task: Configure the optimizer for model training.
# Here, we're using Stochastic Gradient Descent (SGD). Think through what parameters are needed.
# Reminder: Utilize the learning rate defined above when setting up your optimizer.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Transfer the model to the selected device (CPU or GPU)
model.to(device)

# Apply custom weight initialization function to the model layers
# Note: Weight initialization can significantly affect training dynamics
model.apply(init_weights)

# Call the training function to start the training process
# Note: All elements like epochs, loss function, learning rate, etc., are passed as arguments
train(epochs, loss_function, learning_rate, model, optimizer, train_loader, device)

# Loop through the model's parameters to display them
# This is helpful for debugging and understanding how well the model has learned
for name, param in model.named_parameters():
    # 'name' will contain the name of the parameter (e.g., 'layer1.weight')
    # 'param.data' will contain the parameter values
    print(name, param.data)

