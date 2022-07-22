import os

# Define the base output path:
BASE_OUTPUT = r"\Output"

TESTDATA = r"\Testdata"

TESTDATA_PATH_X = os.path.sep.join([TESTDATA, "x_test.npy"])
TESTDATA_PATH_Y = os.path.sep.join([TESTDATA, "y_test.npy"])

# Define the model paths:
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "ffnn_model.tf"])

# Path for the loss:
PLOT_LOSS_PATH = os.path.sep.join([BASE_OUTPUT, "loss_curve.png"])

# Training epochs parameter
EPOCHS = 20
BATCH_SIZE = 128
