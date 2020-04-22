"""
EmojiEmotion - Facial Landmarks
CS1430 - Computer Vision
Brown University
"""

"""
Overview
========
This program uses OpenCV to capture images from the camera, Fourier transform
them and show the Fourier transformed image alongside the original on the screen.

$ ./liveFFT2.py

Required: A Python 3.x installation (tested on 3.5.3),
with: 
    - OpenCV (for camera reading)
    - numpy, matplotlib, scipy, argparse
"""

__author__ = "Lucas Schroeder, Christopher Luke, Antohony Peng"
__contact__ = "lucas_schroeder@brown.edu"
__date__ = "2020/04/21"

# https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672 
# Look at this tutorial for inspiration

import cv2 #opencv-based functions
import time
import math
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray

# if (you have only 1 webcam){ set device = 0} else{ chose your favorite webcam setting device = 1, 2 ,3 ... }
cap = cv2.VideoCapture(0)
while True:
    # Getting our image by webcam and converting it into a gray image scale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # show the gray image
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    width = image.shape[1]
    height = image.shape[0]
    #cv2.resizeWindow('image', 200, 200)

    cv2.imshow("image", image)
    #cv2.imshow("Output")
    
    # If you are using a 64-bit machine, you have to modify 
    # cv2.waitKey(0) line as follows : k = cv2.waitKey(0) & 0xFF
    k = cv2.waitKey(5) & 0xFF
    
    # This checks if the "esc" key was pressed to exit the livestream
    if k == 27:
        cv2.destroyAllWindows()
        break
    # The next if statement determines if the window was closed using
    # the "x" button in the top left corner of the window. 
    # cv2.getWindowProperty() returns -1 as soon as the window is closed.
    # For explanation, see the documentation for the enumeration of 
    # cv::WindowPropertyFlags: getting the flag with index 0 is the fullscreen 
    # property, but actually it doesn't matter which flag to use, they all become
    #  -1 as soon as the window is closed.
    if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:        
        break 
cv2.destroyAllWindows()
cap.release()
