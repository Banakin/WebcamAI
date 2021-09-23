import tkinter as tk
from PIL import ImageTk, Image
import pandas as pd
import cv2
import os

# Main function
def main():
    # Create window
    window = tk.Tk()
    window.title("WebcamAI")

    # Initialize tkinter variables
    global imageData, datasetPath, annotationsFile, outputPath, imageIndex
    imageData = tk.StringVar(window, value="1")
    datasetPath = tk.StringVar(window, value="./data/CustomDataset/raw")
    annotationsFile = tk.StringVar(window, value="images.csv")
    outputPath = tk.StringVar(window, value="./model/model.pth")
    imageIndex = tk.IntVar(window, value=0)

    # Set up the UI components
    uiSetup(window)

    # Update image index when annotationsFile or datasetPath are updated
    imageIndexUpdate()
    annotationsFile.trace('w', imageIndexUpdate)
    datasetPath.trace('w', imageIndexUpdate)

    # tkinter loop to keep the window running
    window.mainloop();

# Set up UI components
def uiSetup(window):
    # Camera View
    viewPort = tk.Label(window, text="Viewport Goes Here")
    viewPort.grid(column=0, row=0, columnspan=3);

    # Record data section
    tk.Button(window, text ="Record").grid(column=0, row=1);

    tk.Label(window, text="Image Data").grid(column=0, row=2)
    tk.Entry(window, textvariable=imageData).grid(column=0, row=3)

    tk.Label(window, text="Path to Dataset").grid(column=0, row=4)
    tk.Entry(window, textvariable=datasetPath).grid(column=0, row=5)

    tk.Label(window, text="Annotations File (CSV)").grid(column=0, row=6)
    tk.Entry(window, textvariable=annotationsFile).grid(column=0, row=7)

    tk.Label(window, textvariable=imageIndex).grid(column=0, row=8)

    # Ai training section
    tk.Button(window, text="Train AI").grid(column=1, row=1)

    tk.Label(window, text="Model output path").grid(column=1, row=2)
    tk.Entry(window, textvariable=outputPath).grid(column=1, row=3)

    # Currently seeing section
    tk.Label(window, text="Currently Seeing:").grid(column=2, row=1)
    tk.Label(window, text="WIP, come back soon").grid(column=2, row=2)

# Updated the imageIndex variable on path change
def imageIndexUpdate(a=None, b=None, c=None):
    # print("bruh")
    imageIndex.set(len(pd.read_csv(os.path.join(datasetPath.get(), annotationsFile.get()), header=None)))

# Make sure the user is running the script intentionally
if __name__ == "__main__":
    main()