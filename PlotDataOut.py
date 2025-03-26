import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, Frame
import pandas as pd

# Author: Andrea Pereira
# This file corresponds to part 3 - visualization
def plotData(filename):
    data = pd.read_csv(filename)

    # separate by classifier
    walking_data = data[data["activity"] == "Walking"]
    jumping_data = data[data["activity"] == "Jumping"]

    fig, ax = plt.subplots(figsize=(10, 4))

    # walking is red
    ax.scatter(walking_data["window"], walking_data["activity"], c='red', label='Walking')

    # jumping is blue
    ax.scatter(jumping_data["window"], jumping_data["activity"], c='blue', label='Jumping')

    ax.set_xlabel("Window")
    ax.set_ylabel("Activity")
    ax.set_title("Activity vs Window")
    ax.legend()
    ax.grid(True)

    # tkinter plot
    # taken from https://www.geeksforgeeks.org/how-to-embed-matplotlib-charts-in-tkinter-gui/
    root = Tk()
    root.title("Activity vs Window Plot")

    plot_frame = Frame(root, width=500, height=400)
    plot_frame.pack()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    toolbar.pack()

    root.mainloop()