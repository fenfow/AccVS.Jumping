import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import power_transform
from scipy import stats

# this file corresponds to part 4 and part 5 pre processing and feature extraction
features_list = []
def calc_ema(data):
    ema = data.ewm(span=50, adjust=False).mean()
    return ema

def extract_features(segment, activity):
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
        # segment metadata
        'start_time': segment['Time (s)'].iloc[0].min(),
        'activity': activity
    }
    return segment_features

numerical_features = ['xmax', 'xmin', 'xmean', 'xrange', 'xmedian', 'xvariance', 'xskewness', 'xkurtosis',
                      'ymax', 'ymin', 'ymean', 'yrange', 'ymedian', 'yvariance', 'yskewness', 'ykurtosis',
                      'zmax', 'zmin', 'zmean', 'zrange', 'zmedian', 'zvariance', 'zskewness', 'zkurtosis']

# fucntion to remove outliers using the z score method
def remove_outliers(features_df):
    # using the Z-score method
    z_scores = np.abs(stats.zscore(features_df[numerical_features]))
    outliers = (z_scores > 3).any(axis=1)
    print(f"Found {outliers.sum()} outliers")

    # return df without outliers
    return features_df[~outliers].reset_index(drop=True)


def processData(df):
    df = df.sort_values(by='Time (s)')

    # Calculating ema using calc_ema function
    df['EMA_x'] = calc_ema(df['Acceleration x (m/s^2)'])
    df['EMA_y'] = calc_ema(df['Acceleration y (m/s^2)'])
    df['EMA_z'] = calc_ema(df['Acceleration z (m/s^2)'])

    windows = len(df) // 503

    for start in range(0, len(df), 503):
        end = min(start + 503, len(df))
        segment = df.iloc[start:end]
        # feature calculations
        segment_features = extract_features(segment, 0)
        features_list.append(segment_features)

    features_df = pd.DataFrame(features_list)
    cleanfeatures = remove_outliers(features_df)

    skewedftrs = {'xrange', 'xmedian', 'xvariance', 'xkurtosis', 'ymin', 'ymean', 'ymedian', 'yvariance', 'ykurtosis',
                  'zmax', 'zmean', 'zrange', 'zmedian', 'zvariance', 'zkurtosis'}

    for ftrs in skewedftrs:
        data_to_transform = cleanfeatures[ftrs].to_numpy()
        transformed_data = power_transform(data_to_transform.reshape(-1, 1))
        cleanfeatures[ftrs] = transformed_data.reshape(-1)

    # normalization
    scaler = preprocessing.StandardScaler()

    cleanfeatures[numerical_features] = scaler.fit_transform(cleanfeatures[numerical_features])

    print(cleanfeatures.shape)
    return cleanfeatures