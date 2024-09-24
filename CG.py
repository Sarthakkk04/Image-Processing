# IMPORTING LIB
import cv2 
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils
import random

# IMG PROCESSING
image_path = "Test1.jpg"  # Ensure this path is correct
img = cv2.imread(image_path)  # Read image

if img is None:
    raise ValueError("Image not found. Please check the file path.")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to gray
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
plt.imshow(bfilter, cmap='gray')  # Show processed image
plt.title('Processed Image')
plt.show()

# EDGE DETECTION
edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
plt.imshow(edged, cmap='gray')  # Show edge-detected image
plt.title('Edge Detected Image')
plt.show()

# SEPARATE ACTUAL POINTS
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours 
contours = imutils.grab_contours(keypoints)  # Grab contours 
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Sort contours

# Loop over our contours to find the best possible approximate contour of 10 contours
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

if location is None:
    raise ValueError("No suitable contour found.")

print("Location: ", location)

mask = np.zeros(gray.shape, np.uint8)  # Create blank image with same dimensions as the original image
new_image = cv2.drawContours(mask, [location], 0, 255, -1)  # Draw contours on the mask image
new_image = cv2.bitwise_and(img, img, mask=mask)  # Take bitwise AND between the original image and mask image

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))  # Show the final image
plt.title('Image with Contours')
plt.show()

(x, y) = np.where(mask == 255)  # Find the coordinates of the four corners of the document
(x1, y1) = (np.min(x), np.min(y))  # Find the top left corner
(x2, y2) = (np.max(x), np.max(y))  # Find the bottom right corner
cropped_image = gray[x1:x2+1, y1:y2+1]  # Crop the image using the coordinates

plt.imshow(cropped_image, cmap='gray')  # Show the cropped image
plt.title('Cropped Image')
plt.show()

# OCR
reader = easyocr.Reader(['en'])  # Create an easyocr reader object with English as the language
result = reader.readtext(cropped_image)  # Read text from the cropped image

if len(result) > 0:
    text = result[0][-2]  # Extract the text from the result
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font style
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)  # Put the text on the image
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)  # Draw a rectangle around the text

    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))  # Show the final image with text
    plt.title('Final Image with OCR Text')
    plt.show()
else:
    print("No text found in the image.")