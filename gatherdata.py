from tkinter.constants import LEFT
import numpy as np
import tkinter
from PIL import ImageTk, Image
import cv2
import os

window = tkinter.Tk()
# Title the window
window.title("Data Capture");
# Create a frame
app = tkinter.Frame(window)
app.grid()
# Create a label in the window
lmain = tkinter.Label(app)
lmain.grid()

imageIndex = tkinter.IntVar(app, value=0);
imageData = tkinter.StringVar(app, value="1");

annotationsFile = open(os.path.join("./data/CustomDataset/raw/images.csv"), "a+")

# Create a button for saving the image to the first data set
def saveImage():
    print("Saving image");
    img = video_stream();
    name = "image"+str(imageIndex.get())+".png"

    # Save image
    path = os.path.join("./data/CustomDataset/raw/", name)
    img.save(path, "PNG")
    print("Image saved at "+path)

    # Save data associated with the image
    data = name+", "+imageData.get()+"\n"
    annotationsFile.write(data)
    print(data)
    annotationsFile.read() # i have to read the file or else it doesn't save for some reason

    # Update the index
    imageIndex.set(imageIndex.get()+1)


saveBtn = tkinter.Button(app, text ="Save image", command = saveImage)
saveBtn.grid();

tkinter.Label(app, text="Image Index").grid();
i1 = tkinter.Entry(app, textvariable=imageIndex)
i1.grid()

tkinter.Label(app, text="Image Data").grid();
i2 = tkinter.Entry(app, textvariable=imageData)
i2.grid()

# Capture from camera
cv2capture = cv2.VideoCapture(0)

# function for video streaming
def video_stream():
    _, frame = cv2capture.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Center crop the image as a square (Only works on horizontal cameras)
    height = int(cv2capture.get(cv2.CAP_PROP_FRAME_HEIGHT));
    width = int(cv2capture.get(cv2.CAP_PROP_FRAME_WIDTH));

    left = int((width - height)/2);
    right = left + height;

    cropped_image = cv2image[0:height, left:right]

    # Scale the image down to 96 x 96
    resized_image = cv2.resize(cropped_image, (128, 128));

    # Make image work in tkinter
    return Image.fromarray(resized_image)

def display_video():
    img = video_stream();
    imgtk = ImageTk.PhotoImage(image=img)

    # Add feed to window
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, display_video)

def main():
    display_video()
    window.mainloop()
    print("Closing annotations file")
    annotationsFile.close()

if __name__ == "__main__":
    main();