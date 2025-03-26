import pandas as pd
import matplotlib.pyplot as plt

# Author: Andrea Pereira
# this file corresponds with part 3 and allows you to visualise the collected data

# Open the HDF5 file
store = pd.HDFStore('dataset.h5')

# List of folders
folders = [
    '/Sascha/Jumping_Back_Pocket', '/Sascha/Jumping_Front_Pocket',
    '/Sascha/Walking_Back_Pocket', '/Sascha/Walking_Front_Pocket',
    '/Robert/Jumping_Front_Right_Pocket', '/Robert/Jumping_Jacket_Pocket',
    '/Robert/Walking_Front_Right_Pocket', '/Robert/Walking_Jacket_Pocket',
    '/Andrea/Jumping_Right_Hand', '/Andrea/Jumping_Right_Pocket',
    '/Andrea/Walking_Right_Hand', '/Andrea/Walking_Right_Pocket'
]

# Loop through each folder
for folder in folders:
    # Read the data from the folder
    df = store[folder]

    # Plot acceleration vs. time graph
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (s)'], df['Absolute acceleration (m/s^2)'], color='red')
    plt.title('Acceleration vs. Time ({})'.format(folder))
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute Acceleration (m/s^2)')
    plt.grid(True)
    plt.show()

    # Plot histogram of absolute acceleration
    plt.figure(figsize=(10, 6))
    plt.hist(df['Absolute acceleration (m/s^2)'], bins=20, color='skyblue', density=True)
    plt.title('Histogram and Density Plot of Absolute Acceleration ({})'.format(folder))
    plt.xlabel('Absolute Acceleration (m/s^2)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

# Close the HDF5 file
store.close()
