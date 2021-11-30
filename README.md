# WebcamAI
Basic machine learning software for myself, made in my AI class

Inspired by the Google's [Teachable Machine](https://teachablemachine.withgoogle.com/)

# Using the software
## Webcam Section
Just a preview of your webcam

## Dataset Section
### The Record Check Box
- Saves an image every frame to the **training** dataset along with the current value in `Image Data`

### The Record Testing Check Box
- Saves an image every frame to the **testing** dataset along with the current value in `Image Data`

### Image Data
- The data/classification associated with the testing/training image (such as 0 for no object and 1 for an object)
- Right now only supports one data point, items separated by a comma after the first data point will (probably) be ignored

### Path to Dataset
- The path as to where the **training** data will be saved. Data will be saved as png files and then there will be a csv file associating each image with its classification.

### Path to Testing Dataset
- The path as to where the **testing** data will be saved. Data will be saved as png files and then there will be a csv file associating each image with its classification.

### Annotations File Name (CSV)
- The name of the annotations file for the training and testing datasets
- Must be a CSV file

### Next Image Will Save At Index
- The index of where the next **training** image will save

### Next Testing Image Will Save At Index
- The index of where the next **testing** image will save

## Training Section
### Training Button

### Epochs

### Batch Size

### Testing Batch Size

### Model Path

## Seeing Section
### Image Labels

### Look at Camera

### Currently Seeing