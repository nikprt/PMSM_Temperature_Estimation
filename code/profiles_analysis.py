'''
Loading the measurement data of the test bench PMSM in the recorded profiles.
Each profile is marked with a profile ID.

Script contains the following work:
-----------------------------------
- loading the dataset
- analyzing the distributions/sizes of the individual profiles
- analyzing the distributions of the temperature features
- feature engineering
- correlation analysis
'''
# Import libraries for the EDA:
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from preprocessing_methods import create_features

# Load the dataset with raw measurements:
df_raw = pd.read_csv('data/measures_v2.csv')

# Load the dataset with scaled values:
df_scaled = pd.read_csv('data/pmsm_temperature_data.csv')

# Method for counting and visualizing the amount of data points per profile:
def count_data(data):

    plt.figure(figsize=(11,5))

    # Count the data in each profile and plot it as barplot:
    counts = data['profile_id'].value_counts().sort_values().plot(kind = 'bar',
                                                                  edgecolor='black',
                                                                  linewidth=1,
                                                                  color = 'cornflowerblue',
                                                                  label = 'n samples')

    plt.xlabel('Profile ID', fontweight='bold')
    plt.ylabel('Data Samples', fontweight='bold')
    plt.title('Data Samples per Measurement Profile', fontsize=14, fontweight='bold')
    plt.savefig("/Users/niklaspickert/Desktop/Github Projects/PMSM Temperature Prediction/images/data_profiles.png", dpi=300)
    plt.show()

count_profiles = count_data(df_raw)

# Get the five largest profiles [20,6,65,18,66] for the further analysis:
profile20 = df_raw.loc[df_raw['profile_id'] == 20]
profile6 = df_raw.loc[df_raw['profile_id'] == 6]
profile65 = df_raw.loc[df_raw['profile_id'] == 65]
profile18 = df_raw.loc[df_raw['profile_id'] == 18]
profile66 = df_raw.loc[df_raw['profile_id'] == 66]

# Method for computing the standard deviation of the temperature features:
def get_temp_stds(data):
    std_ambient = np.array(data['ambient'].std())
    std_coolant = np.array(data['coolant'].std())
    std_pm = np.array(data['pm'].std())
    stds = np.hstack([std_ambient, std_coolant, std_pm])

    return stds

# Get the standard deviations for each of the selected profiles:
profile20_stds = get_temp_stds(profile20)
profile6_stds = get_temp_stds(profile6)
profile65_stds = get_temp_stds(profile65)
profile18_stds = get_temp_stds(profile18)
profile66_stds = get_temp_stds(profile66)

# Method for plotting the distribution of the temperature features:
def plot_temp_distributions(profile1, profile2, profile3, profile4, profile5):

    # Dictionary for the x ticks labels:
    data_dict = {'Profile 20': profile1['pm'].values, 'Profile 6': profile2['pm'].values,
                 'Profile 65': profile3['pm'].values, 'Profile 18': profile4['pm'].values,
                 'Profile 66': profile5['pm'].values}

    colors = ['darksalmon', 'darksalmon', 'darksalmon', 'darksalmon', 'darksalmon']

    fig, ax = plt.subplots(figsize=(6,4))
    boxplot = ax.boxplot(data_dict.values(), vert=True, patch_artist=True, labels=data_dict.keys())
    ax.set_xticklabels(data_dict.keys(), fontweight='bold')
    ax.yaxis.grid()
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('Magnet Temperature Distributions', fontsize=13, fontweight='bold')

    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.savefig("/Users/niklaspickert/Desktop/Github Projects/PMSM Temperature Prediction/images/pm_temps.png", dpi=300)
    plt.show()

pm_distributions = plot_temp_distributions(profile20,
                                           profile6,
                                           profile65,
                                           profile18,
                                           profile66)

'''
FEATURE ENGINEERING:
--------------------
Taking the current and voltage features for the motor axes q and d and
features representing the absolute values for both axes - "i_abs", "u_abs"

Taking the current and voltage features and computing the effective power gives
a third additional feature - "p_eff"

'''
# Create additional features and add them to the dataframe:
df_raw = create_features(df_raw)

# Drop the columns 'stator_tooth', 'stator_yoke', 'profile_id':
df_raw.drop(['stator_tooth','stator_yoke', 'profile_id'], axis=1, inplace=True)

# Print columns to make sure, the created features have been added:
print("Column names of dataframe including new features:\n", df_raw.columns)

# Method for plotting the correlation matrix including the engineered features:
def plot_correlations(data):
    corr = data.corr()
    plt.figure(figsize = (8,5))
    ax = sns.heatmap(corr, xticklabels=True, yticklabels=True,
                     annot=True, fmt=".2f", linewidth=0.5, cmap='coolwarm')
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/Users/niklaspickert/Desktop/Github Projects/PMSM Temperature Prediction/images/corr_matrix.png", dpi=300)
    plt.show()

plot_correlations(df_raw)
