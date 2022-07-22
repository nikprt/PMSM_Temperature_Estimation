#Import the necessary libraries and packages:
import os
import config

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import  load_model
from keras.callbacks import History


# Import the testdata from disc:
x_test = np.load(r"\Testdata/x_test.npy")
y_test = np.load(r"\Testdata/y_test.npy")

# Load the trained model from disc:
model = load_model('ffnn_model.hp')

# Make predictions on test profile (6):
p6_predictions = model.predict(x_test)

#p6_predictions = pd.DataFrame(y_scaler.inverse_transform(p6_predictions))

print("Predicted values: ", p6_predictions)
print("Ground truth values (y_test): ", y_test)

#Compute and print the evaluation metrics MSE and MAE:
mse = mse(p6_predictions, y_test)
mae = mae(p6_predictions, y_test)

print("MSE = ", mse)
print("MAE = ", mae)

#Compute and print the error between predicted and ground truth:
error = p6_predictions - y_test

m = np.linspace(0, 40388, 40388)

#Plot predicted and ground truth data:
plt.figure()
plt.plot(m, p6_predictions[:, 0], color='tomato', label='Model Predictions')
plt.plot(m, y_test, color='cornflowerblue', label='Ground Truth PM Temperature')
plt.xlabel('Test samples')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title('Feedforward Neural Network (EWMA) Predictions')
plt.grid()
plt.show()

#Plot the error curve of the model predictions:
plt.figure()
plt.plot(m, error, color='darkred', label='Prediction Error')
plt.xlabel('Test samples', figweight='bold')
plt.ylabel('Temperature (°C)', figweight='bold')
plt.title('Model Prediction Error', figsize=14, figweight='bold')
plt.legend()
plt.grid()
plt.show()
