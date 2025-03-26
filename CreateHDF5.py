import h5py
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# this file corresponds to part 2 - data storage

# arrays which temporary store the spliced data
walking_train = []
jumping_train = []
walking_test = []
jumping_test = []
combined_test = []
combined_train = []

with h5py.File('./dataset.h5', 'w') as hdf:
    # hard coded members
    members = ['Robert', 'Andrea', 'Sascha']

    for member in members:

        # list all csv files in the data folder
        csv_files = [f for f in os.listdir('./Data/' + member) if f.endswith('.csv')]

        for csv_file in csv_files:
            df = pd.read_csv('./Data/' + member + '/' + csv_file)

            # my "clever" way of classifying what activity the data is,
            # since each file is named action_... we can retrieve that part and use it to tag the data
            activity = csv_file.split('_')[0].lower()

            # column names
            dtype = np.dtype([('Time(s)', df.dtypes['Time (s)']),
                              ('Acceleration x (m/s^2)', df.dtypes['Acceleration x (m/s^2)']),
                              ('Acceleration y (m/s^2)', df.dtypes['Acceleration y (m/s^2)']),
                              ('Acceleration z (m/s^2)', df.dtypes['Acceleration z (m/s^2)']),
                              ('Absolute acceleration (m/s^2)', df.dtypes['Absolute acceleration (m/s^2)'])])

            # segment the data into 5 second windows
            # the sample rate of the accelerometer is 100.7 data points per second,
            # which gives a window size of about 503 data points
            windows = [df[i:i + 503] for i in range(0, len(df), 503)]

            # shuffle the data
            np.random.shuffle(windows)

            # split the data 90% training, 10% testing
            train, test = train_test_split(windows, test_size=0.1)

            # add the data to the list
            if (activity == 'jumping'):
                jumping_train += train
                jumping_test += test
            else:
                walking_train += train
                walking_test += test

            combined_test += test
            combined_train += train

            # store the members data into the hdf5 file without making it into a dataset
            store = pd.HDFStore('./dataset.h5')
            store[member + '/' + csv_file[:-4]] = pd.concat(windows)

            store.close()

    combined_walking_test_array = np.concatenate([df.to_records(index=False) for df in walking_test])
    combined_jumping_test_array = np.concatenate([df.to_records(index=False) for df in jumping_test])

    combined_walking_train_array = np.concatenate([df.to_records(index=False) for df in walking_train])
    combined_jumping_train_array = np.concatenate([df.to_records(index=False) for df in jumping_train])

    combined_test_array = np.concatenate([df.to_records(index=False) for df in combined_test])
    combined_train_array = np.concatenate([df.to_records(index=False) for df in combined_train])

    # write combined data
    dataset_group = hdf.create_group('dataset')
    # subgroups within dataset
    train_group = hdf.create_group('dataset/train')
    test_group = hdf.create_group('dataset/test')
    # actual datasets in the file
    train_group.create_dataset('walking_train', data=combined_walking_train_array)
    train_group.create_dataset('jumping_train',data=combined_jumping_train_array)
    train_group.create_dataset('combined_train',data=combined_train_array)

    test_group.create_dataset('walking_test', data=combined_walking_test_array)
    test_group.create_dataset('jumping_test',data=combined_jumping_test_array)
    test_group.create_dataset('combined_test',data=combined_test_array)