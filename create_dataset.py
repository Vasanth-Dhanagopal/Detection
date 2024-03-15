# Import opencv for computer vision stuff
import cv2
# Import matplotlib so we can visualize an image
from matplotlib import pyplot as plt

# Connect to capture device
cap = cv2.VideoCapture(0)
# Get a frame from the capture device
ret, frame = cap.read()
plt.imshow(frame)
