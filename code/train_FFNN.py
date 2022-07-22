import os
import config

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential, save_model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD

from keras.callbacks import History

from tempfile import TemporaryFile

# import the methods for data preprocessing:
from preprocessing_methods import create_features, transform_ewma, split_train_test

# Read the data into a Pandas DataFrame:
df = pd.read_csv('data/measures_v2.csv')
df = pd.DataFrame(df, columns=['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d', 'i_q',
                                'pm', 'stator_yoke', 'stator_tooth', 'stator_winding', 'profile_id'])


# Drop the column with the torque values (not needed according to the EDA results):
df.drop('torque', axis=1, inplace=True)

# Separate the columns of PM and Profile ID:
cols_to_add = pd.DataFrame(df, columns=['pm', 'profile_id'])

#-------------------------------------------------------------------------------
'''
FEATURE ENGINEERING:
--------------------
Create the additional features for absolute current, absolute voltage and effective power:
'''
# Add additional features to the dataframe by calling "creature_features" method:
df = create_features(df)

print(df.tail(20))

#-------------------------------------------------------------------------------
'''
EXPONENTIALLY WEIGHTED MOVING AVERAGES:
---------------------------------------
Compute the moving averages of the data with a defined lookback span.
This transformed dataset (holding moving averages of the timeseries data) will
help the neural network learn the time-dependent features in the data.
'''

# Call method for transofrming the data with moving averages:
df_new = transform_ewma(df, 1320, cols_to_add, ['pm_ewma_1320',
                                                         'profile_id_ewma_1320',
                                                         'stator_yoke_ewma_1320',
                                                         'stator_tooth_ewma_1320'])

# Display the new column names of the transformed dataframe:
print("Dataframe after ewma transformation method:\n", df_new.tail(20))

# Define target feature (PM temperatures) and the Profile ID for data handling:
target_features = ['pm']
PROFILE_ID_COL = 'profile_id'

# Create names for x,y columns:
x_cols = [x for x in df_new.columns.tolist() if x not in target_features + [PROFILE_ID_COL]]
y_cols = target_features

# Call 'split_train_test' method for performing a train test split
# with profile_6 as test data:
x_train, y_train, x_test, y_test = split_train_test(df, df_new, [6], x_cols, target_features)

print("x_train: \n", x_train.shape)

# Method for saving x_test, y_test to disc:
def save_testset(x_data, y_data):

    np.save(config.TESTDATA_PATH_X, x_data)
    np.save(config.TESTDATA_PATH_Y, y_data)

# Save the test data for reading it with the evaluation script:
save_testset(x_test, y_test)

'''
Model Initialization: Defining the model architecture and training the model
-------------------------------------------------------------------------
'''
# Method for creating the model and describing teh architecture:
def create_model():

    # Define the model:
    model = Sequential()
    model.add(Dense(11, input_dim=11, kernel_initializer='normal',
                    activation='relu', activity_regularizer=regularizers.l2(1.7e-3)))    # Input Layer
    #model.add(Dropout(0.30))
    model.add(Dense(14, activation='relu', activity_regularizer=regularizers.l2(1.7e-2))) # Hidden Layer 1
    #model.add(Dropout(0.20))
    model.add(Dense(14, activation='relu', activity_regularizer=regularizers.l2(1.7e-2))) # Hidden Layer 2
    #model.add(Dropout(0.20))
    model.add(Dense(14, activation='relu', activity_regularizer=regularizers.l2(1.7e-3))) # Hidden Layer 3
    model.add(Dense(14, activation='relu', activity_regularizer=regularizers.l2(1.7e-7))) # Hidden Layer 4
    model.add(Dense(1, activation='linear'))                                              # Output Layer

    # Set the optimization algorithm:
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False, name='SGD')
    # COmpile the model:
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae'])
    summary = model.summary()

    return model

# Method for training the neural network:
def train_model(model, history):
    history = model.fit(x_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, \
              verbose=2, validation_split=0.2, callbacks=[history])

# Method for plotting the history and saving it to disc:
def plot_and_save_history(history, plot_path):

    # Define the figure:
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training Loss History', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()

    # Save the plot to disc and show the result:
    plt.savefig(plot_path, dpi=300)
    plt.show()

# Define the history we get from keras.callbacks:
history = History()

print("==========================\n")
print("Start training the DNN...\n")
print("==========================\n")

# Define the model:
model = create_model()

# Train the neural network:
train_model(model, history)

# Save the model:
save_model(model, 'ffnn_model.hp')

# Plot the training history:
plot_and_save_history(history, config.PLOT_LOSS_PATH)
