from tkinter.constants import LEFT
import numpy as np
import tkinter
from PIL import ImageTk, Image
import cv2


window = tkinter.Tk()
window.title("Data Capture");
# Create a frame
app = tkinter.Frame(window, bg="white")
app.grid()
# Create a label in the window
lmain = tkinter.Label(app)
lmain.grid()
# Create a button for saving the image in the window
def saveImage():
   print("Saving image");
   img = ImageTk.getimage(lmain);
   img.save(os.path.join("./data/images/with_me", "1.png"), "PNG");

saveImgButton = tkinter.Button(window, text ="Save image", command = saveImage)
saveImgButton.grid();

# Capture from camera
cv2capture = cv2.VideoCapture(0)

while(cv2capture.);
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
    resized_image = cv2.resize(cropped_image, (96, 96));

    # Make image work in tkinter
    img = Image.fromarray(resized_image)
    imgtk = ImageTk.PhotoImage(image=img)

    # Add feed to window
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream)


video_stream()
window.mainloop()

# def main():
#     print("hello");

# if __name__ == "__main__":
#     main();