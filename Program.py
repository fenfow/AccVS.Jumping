import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# this file corresponds with part 7 - desktop application with model deployment

# classes that the program uses
import ReadCSV
import PlotDataOut

# load trained pipeline model
clf = joblib.load('./classifier.sav')

# selected input and output file names
selected_input_file = ""
selected_output_file = "../labelled_data.csv"

def predict(inputFile, outputFile):
    df = ReadCSV.processFile(inputFile)

    # scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df)

    # predict activity
    predictions = clf.predict(features_scaled)

    # add predictions to output
    df['activity'] = np.where(predictions == 1, 'Walking', 'Jumping')

    # write the output to a csv
    df.to_csv(outputFile, index=True, index_label="window")

    PlotDataOut.plotData(outputFile)

def runModel():
    global selected_input_file, selected_output_file
    if selected_input_file:
        if selected_output_file:
            predict(selected_input_file, selected_output_file)
        else:
            # default output
            predict(selected_input_file, "labelled_data.csv")
    else:
        print("Input not selected")

def openFile():
    global selected_input_file
    temp_file = filedialog.askopenfilename(initialdir="/", title="Select .csv file",
                                            filetypes=(("CSV files", "*.csv"), ("All Files", "*.*")))
    if temp_file:
        selected_input_file = temp_file
        input_file_label.config(text="Input: " + selected_input_file)

def saveAs():
    global selected_output_file
    selected_output_file = filedialog.asksaveasfilename(initialdir="/", title="Save labelled data as",
                                                         defaultextension=".csv",
                                                         filetypes=(("CSV files", "*.csv"), ("All Files", "*.*")))
    if not selected_output_file:
        selected_output_file = "../labelled_data.csv"
    output_file_label.config(text="Output: " + selected_output_file)

# UI elements
root = tk.Tk()
root.title("Activity Classification")

input_file_label = tk.Label(root, text="Input: None")
input_file_label.pack()

button1 = tk.Button(root, text="Load .csv file", command=openFile)
button1.pack()

output_file_label = tk.Label(root, text="Output: " + selected_output_file)
output_file_label.pack()

button2 = tk.Button(root, text="Save as", command=saveAs)
button2.pack()

button3 = tk.Button(root, text="Run Model", command=runModel)
button3.pack()

root.mainloop()
