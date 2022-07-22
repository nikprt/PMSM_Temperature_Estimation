# PMSM_Temperature_Estimation
Using Machine Learning/Deep Learning to estimate the permanent magnet temperature inside an electric motor.

The data hat have been used for training and evaluating of the model consists of 2 individual datasets:

- **measures.csv** --> raw measurement data from the test bench
- **pmsm_temperature_data.csv** --> normalized data 

## config.py
The script holds the paths for storing the test data, saving the model to disc, save the training's history plot.
Furthermore, some training parameters (epochs, batch size) are defined here.

## profiles_analysis.py
The data of the measurement profiles is analyzed here. With the help of several methods, insights of data distributions as well as linear feature relationships are gained here.

## preprocessing_methods.py 
All self-written methods used for pre-processing the data before training are defined here. 

## train_FFNN.py
Main script for training the neural network. A feedworkward neural network takes in the feature data with exponentially weighted moving averages and is trained with the stochastic gradient descent optimizer.
After model training, the model is saved to disc and the training history is visualized.

## evaluate_FFNN.py
Evaluation script for the trained and saved model. The permanent magnet temperature data of profile 6 is plottet with the model's predictions on profile 6.
Furthermore, the Mean Squared Error (MSE) and Mean Absolute Error (MAE) between the ground truth data and the predictions is computed.
