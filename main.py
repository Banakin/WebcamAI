import tkinter as tk
from PIL import ImageTk, Image
import pandas as pd
import cv2
import os
from lib.image import getImage, saveImage
from lib.trainModel import trainAndSave

# Main function
def main():
    global imageData, datasetPath, annotationsFile, outputPath, imageIndex, cv2capture, testingDatasetPath, testingImageIndex, batchSize, testingBatchSize

    # Create window
    window = tk.Tk()
    window.title("WebcamAI")

    # Initialize tkinter variables
    imageData = tk.StringVar(window, value="1")
    datasetPath = tk.StringVar(window, value="./data/CustomDataset/raw")
    testingDatasetPath = tk.StringVar(window, value="./data/CustomDataset/testing")
    annotationsFile = tk.StringVar(window, value="images.csv")
    outputPath = tk.StringVar(window, value="./model/model.pth")
    imageIndex = tk.IntVar(window, value=0)
    testingImageIndex = tk.IntVar(window, value=0)
    batchSize = tk.IntVar(window, value=64)
    testingBatchSize = tk.IntVar(window, value=1)

    # Set up the UI components
    uiSetup(window)

    # Update image index when annotationsFile or datasetPath are updated
    imageIndexUpdate()
    annotationsFile.trace('w', imageIndexUpdate)
    datasetPath.trace('w', imageIndexUpdate)

    # Update testing image index when annotationsFile or testingDatasetPath are updated
    testingImageIndexUpdate()
    annotationsFile.trace('w', imageIndexUpdate)
    testingDatasetPath.trace('w', imageIndexUpdate)

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
    tk.Button(window, text ="Record Testing", command=capTestingImage).grid(column=0, row=2);

    tk.Label(window, text="Image Data").grid(column=0, row=3)
    tk.Entry(window, textvariable=imageData).grid(column=0, row=4)

    tk.Label(window, text="Path to Dataset").grid(column=0, row=5)
    tk.Entry(window, textvariable=datasetPath).grid(column=0, row=6)

    tk.Label(window, text="Path to Testing Dataset").grid(column=0, row=7)
    tk.Entry(window, textvariable=testingDatasetPath).grid(column=0, row=8)

    tk.Label(window, text="Annotations File Name (CSV)").grid(column=0, row=9)
    tk.Entry(window, textvariable=annotationsFile).grid(column=0, row=10)

    tk.Label(window, text="Next Image Will Save At Index:").grid(column=0, row=11)
    tk.Label(window, textvariable=imageIndex).grid(column=0, row=12)

    tk.Label(window, text="Next Testing Image Will Save At Index:").grid(column=0, row=13)
    tk.Label(window, textvariable=testingImageIndex).grid(column=0, row=14)

    # Ai training section
    tk.Button(window, text="Train", command=trainModel).grid(column=1, row=1)

    tk.Label(window, text="Batch Size:").grid(column=1, row=2)
    tk.Entry(window, textvariable=batchSize).grid(column=1, row=3)

    tk.Label(window, text="Testing Batch Size:").grid(column=1, row=4)
    tk.Entry(window, textvariable=testingBatchSize).grid(column=1, row=5)

    tk.Label(window, text="Model output path").grid(column=1, row=6)
    tk.Entry(window, textvariable=outputPath).grid(column=1, row=7)

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

# Save an image to the testing dataset
def capTestingImage():
    saveImage(currentImg, annotationsFile.get(), testingImageIndex.get(), imageData.get(), testingDatasetPath.get())
    testingImageIndex.set(testingImageIndex.get()+1)

# Train and save the model
def trainModel():
    trainAndSave(datasetPath.get(), annotationsFile.get(), batchSize.get(), testingDatasetPath.get(), testingBatchSize.get(), outputPath.get())

# Update the imageIndex variable on path change
def imageIndexUpdate(a=None, b=None, c=None):
    try:
        imageIndex.set(len(pd.read_csv(os.path.join(datasetPath.get(), annotationsFile.get()), header=None)))
    except:
        imageIndex.set(0)

# Update the imageIndex variable on path change
def testingImageIndexUpdate(a=None, b=None, c=None):
    try:
        testingImageIndex.set(len(pd.read_csv(os.path.join(testingDatasetPath.get(), annotationsFile.get()), header=None)))
    except:
        testingImageIndex.set(0)

# Make sure the user is running the script intentionally
if __name__ == "__main__":
    main()