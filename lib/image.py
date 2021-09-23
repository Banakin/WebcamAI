import cv2
from PIL import Image

# Capture the current image from the cv2 video stream
def getImage(videoStream):
    _, frame = videoStream.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Center crop the image as a square (Only works on horizontal cameras)
    height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT));
    width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH));

    left = int((width - height)/2);
    right = left + height;

    cropped_image = cv2image[0:height, left:right]

    # Scale the image down to 96 x 96
    resized_image = cv2.resize(cropped_image, (128, 128));

    # Make image work in tkinter
    return Image.fromarray(resized_image)

# Save the current image from the cv2 video stream
def saveImage(image, annotationsFile, index, data, path):
    print("Saving image");
    name = "image"+str(index)+".png"

    # Save image
    path = os.path.join(datasetPath.get(), name)
    img.save(path, "PNG")
    print("Image saved at "+path)

    # Save data associated with the image
    annotationsFile = open(os.path.join(datasetPath.get(), "images.csv"), "a+")
    data = name+", "+imageData.get()+"\n"
    annotationsFile.write(data)
    print(data)
    annotationsFile.read() # i have to read the file or else it doesn't save for some reason
    annotationsFile.close() # close the file

    # Update the index
    imageIndex.set(imageIndex.get()+1)
