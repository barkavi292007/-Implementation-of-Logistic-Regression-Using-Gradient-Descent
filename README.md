# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm:

1. Start the program and import the required libraries.
2. Initialize the input dataset, target labels, weights, bias, learning rate, and number of epochs.
3. Apply the sigmoid function to calculate predicted probabilities.
4. Compute the gradients and update the weights and bias iteratively using gradient descent.
5. Predict the final class labels using a threshold of 0.5, display the results, and plot the predicted probabilities.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt

# Dataset: Study hours
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)

# Target: Pass (1) or Fail (0)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 1000
n = len(X)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
for i in range(epochs):
    z = w * X.flatten() + b
    y_pred = sigmoid(z)

    dw = np.sum((y_pred - y) * X.flatten()) / n
    db = np.sum(y_pred - y) / n

    w = w - learning_rate * dw
    b = b - learning_rate * db

# Predict probabilities and classes
probabilities = sigmoid(w * X.flatten() + b)
predictions = (probabilities >= 0.5).astype(int)

print("Weight:", round(w, 2))
print("Bias:", round(b, 2))
print("Predicted Probabilities:", np.round(probabilities, 2))
print("Predicted Classes:", predictions)

# Plot results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, probabilities, color="red", label="Predicted Probability")
plt.axhline(0.5, color="green", linestyle="--", label="Threshold")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Using Gradient Descent")
plt.legend()
plt.show()
```

## Output:
<img width="961" height="731" alt="image" src="https://github.com/user-attachments/assets/9890e683-6ded-4134-96b5-57ca9a60cd54" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

