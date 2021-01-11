import numpy as np
import cv2
import imutils
import pytesseract 

# Read the image file
image = cv2.imread('input/plate.jpg')

# Resize the image - change width to 500
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)

gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)

edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4 - Canny Edges", edged)

# Find contours based on Edges
cnts,_  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)

#for cnt in contours:
x, y, w, h = cv2.boundingRect(NumberPlateCnt)

# Drawing a rectangle on copied image
rect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Cropping the text block for giving input to OCR
cropped = image[y:y + h, x:x + w]

# Apply OCR on the cropped image
text = pytesseract.image_to_string(cropped)

print(text)

cv2.waitKey(0)