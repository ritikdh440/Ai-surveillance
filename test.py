import cv2               #opencv
import numpy as numpy

#load yolo

net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layernames = net.get_layernames()
output_layers = [layernames[i[0]-1] for i in get.UnconnectedOutlayers()]

#load the image
image = cv2.imread("test_image.jpg")

cv2.imshow("Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)
