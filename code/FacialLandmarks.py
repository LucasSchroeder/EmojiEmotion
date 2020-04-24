"""
EmojiEmotion - Facial Landmarks
CS1430 - Computer Vision
Brown University
"""

"""
Overview
========
This program uses OpenCV to capture images from the camera

Required: A Python 3.x installation (tested on 3.5.3),
with: 
    - OpenCV (for camera reading)
    - numpy, matplotlib, scipy, dlib, imutils
"""

__author__ = "Lucas Schroeder, Christopher Luke, Antohony Peng"
__contact__ = "lucas_schroeder@brown.edu"
__date__ = "2020/04/21"

# https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672 
# Look at this tutorial
# You must installing the dependencies by running the following line in your terminal:
#               pip install numpy opencv-python dlib imutils
# If this doesn't work, please look at this website and follow the directions
# https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/ 

# We can train our program to detect different feature points using Dlib. Read more here:
#  https://medium.com/datadriveninvestor/training-alternative-dlib-shape-predictor-models-using-python-d1d8f8bd9f5c 

from imutils import face_utils
import dlib
import cv2 #opencv-based functions
import time
import math
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

run_cam = True

# create an overlay image. You can use any image
overlaidImage = np.ones((100,100,3),dtype='uint8')*255

# Set initial value of weights
alpha = 0.4

# if (you have only 1 webcam){ set device = 0} else{ chose your favorite webcam setting device = 1, 2 ,3 ... }
cap = cv2.VideoCapture(0)
while run_cam:
    # Get the image from the webcam and convert it into a gray image scale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmarks.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        #gray = cv2.flip(image,1)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # Draw on our image, all the finded cordinate points (x,y) 
        # for (x, y) in shape:
        #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.circle(image, (shape[28,0], shape[34,1]), 2, (0, 255, 0), -1)
        # print(shape.shape)
        point1 = shape[28,:]
        point2= shape[(len(shape[:,0]))-1]
        print(point1)
        print(point2)
        #gray = cv2.flip(image,1)
        # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
        added_image = cv2.addWeighted(image[point1[1]-50:point1[1]+50,point1[0]-50:point1[0]+50,:],alpha,overlaidImage[0:100,0:100,:],1-alpha,0)
        # Change the region with the result
        image[point1[1]-50:point1[1]+50,point1[0]-50:point1[0]+50] = added_image

    # show the gray image
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    width = image.shape[1]
    height = image.shape[0]
    #cv2.resizeWindow('image', 200, 200)

    cv2.imshow("image", image)
    #cv2.imshow("Output")
    
    # If you are using a 64-bit machine, you have to modify 
    # cv2.waitKey(0) line as follows : k = cv2.waitKey(0) & 0xFF
    k = cv2.waitKey(1) 
    # press a to increase alpha by 0.1
    if k == ord('a'):
        alpha +=0.1
        if alpha >=1.0:
            alpha = 1.0
    # press d to decrease alpha by 0.1
    elif k== ord('d'):
        alpha -= 0.1
        if alpha <=0.0:
            alpha = 0.0
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
    #if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1: 
    #    print("break")       
    #    break 
cv2.destroyAllWindows()
cap.release()
