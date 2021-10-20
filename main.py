import tkinter as tk
from PIL import ImageTk, Image
import pandas as pd
import cv2
import os
from lib.image import getImage, saveImage

# Main function
def main():
    global imageData, datasetPath, annotationsFile, outputPath, imageIndex, cv2capture

    # Create window
    window = tk.Tk()
    window.title("WebcamAI")

    # Initialize tkinter variables
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

    # Start the video stream
    cv2capture = cv2.VideoCapture(0)
    webcamDisplay()

    # tkinter loop to keep the window running
    window.mainloop();

# Set up UI components
def uiSetup(window):
    global viewPort
    # Camera View
    viewPort = tk.Label(window, text="Viewport Goes Here")
    viewPort.grid(column=0, row=0, columnspan=3);

    # Record data section
    tk.Button(window, text ="Record", command=capImage).grid(column=0, row=1);

    tk.Label(window, text="Image Data").grid(column=0, row=2)
    tk.Entry(window, textvariable=imageData).grid(column=0, row=3)

    tk.Label(window, text="Path to Dataset").grid(column=0, row=4)
    tk.Entry(window, textvariable=datasetPath).grid(column=0, row=5)

    tk.Label(window, text="Annotations File (CSV)").grid(column=0, row=6)
    tk.Entry(window, textvariable=annotationsFile).grid(column=0, row=7)

    tk.Label(window, text="Next Image Will Save At Index:").grid(column=0, row=8)
    tk.Label(window, textvariable=imageIndex).grid(column=0, row=9)

    # Ai training section
    tk.Button(window, text="Train").grid(column=1, row=1)

    tk.Label(window, text="Model output path").grid(column=1, row=2)
    tk.Entry(window, textvariable=outputPath).grid(column=1, row=3)

    # Currently seeing section
    tk.Label(window, text="Currently Seeing:").grid(column=2, row=1)
    tk.Label(window, text="WIP, come back soon").grid(column=2, row=2)

# Webcam Display Loop
def webcamDisplay():
    global currentImg
    # Get the current image (Global so I can use it elsewhere)
    currentImg = getImage(cv2capture);

    # Make the image work with TkInter
    imgtk = ImageTk.PhotoImage(image=currentImg)

    # Add feed to window
    viewPort.imgtk = imgtk
    viewPort.configure(image=imgtk)
    viewPort.after(1, webcamDisplay)

# Save an image when button clicked
def capImage():
    saveImage(currentImg, annotationsFile.get(), imageIndex.get(), imageData.get(), datasetPath.get())
    imageIndex.set(imageIndex.get()+1)

# Updated the imageIndex variable on path change
def imageIndexUpdate(a=None, b=None, c=None):
    try:
        imageIndex.set(len(pd.read_csv(os.path.join(datasetPath.get(), annotationsFile.get()), header=None)))
    except:
        imageIndex.set(0)

# Make sure the user is running the script intentionally
if __name__ == "__main__":
    main()