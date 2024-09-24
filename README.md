**Image-Processing**

**Automatic Number Plate Recognition System**

Automatic License/Number Plate Recognition is an image processing technique used to identify a car based on its number plate &  is the process of detecting the position of a number plate and then using the Optical Character Recognition(OCR) technique to identify the text on the plate.

## Steps Involved
-Read input image and apply filters(grayscale and blur).
-Perform edge detection.
-Find contours and apply mask to separate out actual number plate.
-Extract text from images using OCR.
-Render Result.

## Dependencies
-OpenCV(pip install opencv-python)==4.5.3 [Python 3.8]
-EasyOCR(pip install easyocr)==1.4.1
-imutils(pip install imutils)==0.5.4
-PyTorch
