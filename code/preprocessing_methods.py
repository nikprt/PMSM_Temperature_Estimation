import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


# Method for creating additional features (absolute current, absolute voltage, effective power)
def create_features(data):

    # Take the features of the current and voltage values:
    i_q = np.array(data['i_q'].astype(float))
    i_d = np.array(data['i_d'].astype(float))
    u_q = np.array(data['u_q'].astype(float))
    u_d = np.array(data['u_d'].astype(float))

    # Iterate over entries and compute the abosulte current (i_abs):
    for row in i_q, i_d:
        i_abs = np.sqrt((i_q**2)+(i_d**2))
    # Iterate over entries and compute the abosulte voltage (u_abs):
    for row in u_q, u_d:
        u_abs = np.sqrt((u_q**2)+(u_d**2))
    # Iterate over entries and compute the effective power (p_eff):
    for row in i_d, i_q, u_d, u_q:
        p_eff = (i_d*u_d) + (i_q*u_q)

    # Stack new features together as NumPy matrix:
    new_features = np.column_stack((i_abs, u_abs, p_eff))
    # Convert np-matrix to pd dataframe:
    new_feature_df = pd.DataFrame(new_features, columns = ['i_abs', 'u_abs', 'p_eff'])
    # Put original df and new features together:
    df_new = pd.concat([data, new_feature_df], axis = 1)

    return df_new


# Method for computing exp. weighted moving averages of the data for each feature
def transform_ewma(data, lookback_span, cols_to_add, cols_to_drop):

    '''
    data = input dataframe
    span = size of lookback rate to use for the moving averages
    cols_to_drop = columns of features which won't be used for model training
    cols_to_add = 'pm' and 'profile_id' columns that have been seperated from data
    '''
    # Bootstrap the timeseries data with zero-values for the span of 1320:
    zeros = pd.DataFrame(np.zeros((lookback_span, len(data.columns))), columns=data.columns)
    data_new = pd.concat([zeros, data], axis=0, ignore_index=True)

    # Compute the exp. weighted moving averages:
    ewm_mean = [data_new.ewm(span=lookback_span).mean()\
                .rename(columns=lambda c: c + '_ewma_' + str(lookback_span))]
    concat_l = [pd.concat(ewm_mean, axis=1).astype(np.float32)]

    # Define the new, transformed dataset:
    data_new = pd.concat(concat_l, axis=1).iloc[lookback_span:, :]\
    .reset_index(drop=True)

    # Drop the columns, we won't use as input features for the neural network:
    data_new.drop(cols_to_drop, axis=1, inplace=True)

    # Put Dataframe with moving averages and the original 'pm' and 'profile_id' values together:
    df_new = pd.concat([data_new, cols_to_add], axis=1)

    return df_new

# Method for using profile 6 as testset data and assign the rest to trainset:
def split_train_test(data, ewma_data, test_profile, x_cols, target_features):

    test_set_profiles = test_profile

    # Define the train- and testdata from the dataset with moving averages:
    trainset = ewma_data.loc[~ewma_data.profile_id.isin(test_set_profiles), :].reset_index(drop=True)
    testset = ewma_data.loc[ewma_data.profile_id.isin(test_set_profiles), :].reset_index(drop=True)

    # Define the train- and test data for the target feature:
    train_pm = data.loc[~data.profile_id.isin(test_set_profiles), :].reset_index(drop=True)
    test_pm = data.loc[data.profile_id.isin(test_set_profiles), :].reset_index(drop=True)

    # Split into X matrix (input features) and y vector (target feature):
    x_train = trainset.loc[:, x_cols]
    y_train = train_pm.loc[:, target_features]
    x_test = testset.loc[:, x_cols]
    y_test = test_pm.loc[:, target_features]

    # Normalize the features with mean normalization:
    scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_cols)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_cols)
    y_train = pd.DataFrame(y_scaler.fit_transform(y_train))

    return x_train, y_train, x_test, y_test
