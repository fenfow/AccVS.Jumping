import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import power_transform
from scipy import stats

# this file reuses a lot of the code from parts 4 and 5 to process the CSV file

# allows you to set the window size, i used this in testing
window_size = 503

# Function to calculate exponential moving average (more suitable for data with large spikes than sma)
def calc_ema(data):
    ema = data.ewm(span=50, adjust=False).mean()
    return ema

# reused part 4 and 5 code, but adapted to read a .csv file instead of an hdf5 dataset
def processFile(filename):
    # Read the data from the CSV file
    df = pd.read_csv(filename)

    # Ensuring the data is sorted in the time column
    df = df.sort_values(by='Time (s)')

    # Calculating ema using calc_ema function
    df['EMA_x'] = calc_ema(df['Acceleration x (m/s^2)'])
    df['EMA_y'] = calc_ema(df['Acceleration y (m/s^2)'])
    df['EMA_z'] = calc_ema(df['Acceleration z (m/s^2)'])

    # Split the dataframe into segments of 5 seconds and calculate features
    features_list = []
    for start in range(0, len(df), window_size):
        end = min(start + window_size, len(df))
        segment = df.iloc[start:end]
        # feature calculations
        segment_features = {
            # x features
            'xmax': segment['EMA_x'].max(),
            'xmin': segment['EMA_x'].min(),
            'xrange': segment['EMA_x'].max() - segment['EMA_x'].min(),
            'xmean': segment['EMA_x'].mean(),
            'xmedian': segment['EMA_x'].median(),
            'xvariance': segment['EMA_x'].var(),
            'xskewness': segment['EMA_x'].skew(),
            'xkurtosis': segment['EMA_x'].kurt(),
            # y features
            'ymax': segment['EMA_y'].max(),
            'ymin': segment['EMA_y'].min(),
            'yrange': segment['EMA_y'].max() - segment['EMA_y'].min(),
            'ymean': segment['EMA_y'].mean(),
            'ymedian': segment['EMA_y'].median(),
            'yvariance': segment['EMA_y'].var(),
            'yskewness': segment['EMA_y'].skew(),
            'ykurtosis': segment['EMA_y'].kurt(),
            # z features
            'zmax': segment['EMA_z'].max(),
            'zmin': segment['EMA_z'].min(),
            'zrange': segment['EMA_z'].max() - segment['EMA_z'].min(),
            'zmean': segment['EMA_z'].mean(),
            'zmedian': segment['EMA_z'].median(),
            'zvariance': segment['EMA_z'].var(),
            'zskewness': segment['EMA_z'].skew(),
            'zkurtosis': segment['EMA_z'].kurt(),
        }
        features_list.append(segment_features)

    # Converting final list to dataframe
    features_df = pd.DataFrame(features_list)

    # features minus the metadata ones
    numerical_features = ['xmax', 'xmin', 'xmean', 'xrange', 'xmedian', 'xvariance', 'xskewness', 'xkurtosis',
                          'ymax', 'ymin', 'ymean', 'yrange', 'ymedian', 'yvariance', 'yskewness', 'ykurtosis',
                          'zmax', 'zmin', 'zmean', 'zrange', 'zmedian', 'zvariance', 'zskewness', 'zkurtosis']

    # Detect outliers using the Z-score method
    z_scores = np.abs(stats.zscore(features_df[numerical_features]))
    outliers = (z_scores > 3).any(axis=1)
    print(f"Found {outliers.sum()} outliers")

    # Remove outliers
    cleanfeatures = features_df[~outliers].reset_index(drop=True)

    print("Mean and std before normalizing:")
    for feature in numerical_features:
        print(f"{feature} mean: {cleanfeatures[feature].mean()}")
        print(f"{feature} std: {cleanfeatures[feature].std()}")

    # Applying yeo-johnson method to reduce skewness in more skewed features
    skewedftrs = {'xmax', 'xmin', 'xmean', 'xrange', 'xmedian', 'xvariance', 'xkurtosis', 'yvariance', 'yskewness',
                  'ykurtosis', 'zmax', 'zmin', 'zrange', 'zmedian', 'zvariance', 'zkurtosis'}

    for ftrs in skewedftrs:
        cleanfeatures[ftrs] = power_transform(cleanfeatures[[ftrs]], method='yeo-johnson')

    # Normalization process
    scaler = preprocessing.StandardScaler()

    cleanfeatures[numerical_features] = scaler.fit_transform(cleanfeatures[numerical_features])

    print("\n")
    print((28 * '*') + ' AFTER NORMALIZING ' + (28 * '*'))

    print("\nMean and std after normalizing:")
    for feature in numerical_features:
        mean = cleanfeatures[feature].mean()
        std = cleanfeatures[feature].std()
        print(f"{feature} mean: {mean:.6f}")
        print(f"{feature} std: {std:.6f}")

    print("\n")
    print((10 * '*') + ' SKEWNESS ' + (10 * '*'))
    print("\n")

    for feature in numerical_features:
        # Checking for skewness
        skewness = cleanfeatures[feature].skew()
        print(f"Skewness of {feature}: {skewness:.2f}")
        if abs(skewness) > 0.5:
            print(f"{feature} might be imbalanced.")

    # returns the dataframe to be used in classification
    return cleanfeatures