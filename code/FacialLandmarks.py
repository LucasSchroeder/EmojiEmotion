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
#from tensorflow.keras.models import load_model

from keras.preprocessing.image import img_to_array
import imutils
from tensorflow.keras.models import model_from_json

# https://github.com/Paulymorphous/Facial-Emotion-Recognition
###
# This method will load the weights that our CNN has produced
def load_model(path):
    json_file = open(path + 'chris_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    model = model_from_json(loaded_model_json)
    model.load_weights(path + "chris_weights_fer63_86.h5")
    print("Loaded model fromasadad disk")
    return model

# We will use this method to predict the emotion of a subject. It takes in 
# the whole webcam image and then resizes it according to the bounding 
# box around the face. Before using the model to predict the emotion, we 
# resize it to a 48x48 image, just like the image samples from the FER2013
# dataset. 
def predict_emotion(gray, x, y, w, h):
    face = np.expand_dims(np.expand_dims(cv2.resize(gray[y:y+w, x:x+h]/255, (48, 48)),-1), 0)
    #face = np.expand_dims(np.expand_dims(np.resize(cv2.imread('fear.png',0), (48, 48)),-1), 0)
    prediction = model.predict([face])

    return(int(np.argmax(prediction)), round(max(prediction[0])*100, 2))


# Once we have a working model for our emotion recognition, we can use these lines to 
# load the model, which will call load_model, defined above
###############################################################################################
path = "Models/"
path1 = "Models/chris_weights_fer63_86.h5"
path2 = '_mini_XCEPTION.102-0.66.hdf5'
model = load_model(path)
###############################################################################################

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
emoji_path_dict = {0: "EmojiImages/AngryEmoji.png", 1: "EmojiImages/AngryEmoji.png", 2: "EmojiImages/ScaredEmoji.png", 3: "EmojiImages/SmilingEmoji.png", 4: "EmojiImages/SadEmoji.png", 5: "EmojiImages/SurprisedEmoji.png", 6: "EmojiImages/NeutralEmoji.png"}

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
#cap = cv2.VideoCapture(cv2.CAP_DSHOW)

# This is a boolean that can be set to True or False to turn the emoji feature on and off. The program
# reacts to the 'y' and 'n' keys to turn the emoji on (y) and off (n). 
drawEmoji = True
previousEmotion = 0
while run_cam:
    
    # Get the image from the webcam and convert it into a gray image scale
    _, image = cap.read()
    if drawEmoji == True:
        #image = imutils.resize(image,width=300)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.flip(image,1)
        # image = cv2.flip(image,1)

        # Get faces into webcam's image
        rects = detector(gray, 0)
        

        # For each detected face, find the landmarks.
        for (i, rect) in enumerate(rects):
            # compute the bounding box of the face and draw it on the image
            # https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/
            (cornerX, cornerY, squareWidth, squareHeight) = face_utils.rect_to_bb(rect)
            cornerX = cornerX +10
            cornerY = cornerY -10
            squareWidth = squareWidth - 20
            squareHeight = squareHeight +10
            cv2.rectangle(image,(cornerX,cornerY), (cornerX+squareWidth,cornerY+squareHeight),(255,255,0),1)
            
            emotion_id, confidence = predict_emotion(gray, cornerX, cornerY, squareWidth, squareHeight)
            print("Confidence of prediction: ", confidence)
            if confidence < 20:
                emotion_id = previousEmotion
                print("previous emotion is used")
            previousEmotion = emotion_id

            emotion = emotion_dict[emotion_id]
            cv2.putText(image, emotion,(cornerX+20,cornerY-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), lineType=cv2.LINE_AA)
            Emoji_File_Path= emoji_path_dict[emotion_id]
            


            # Predict the shape of the face and transfom it to numpy array
            #gray = cv2.flip(image,1)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
        
            # Draw on our image, all the cordinate points (x,y) 
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
            point1 = shape[29,:]
            point2= shape[(len(shape[:,0]))-1]
            
            # Get the standard deviation between points to calculate the ratio relating to how far you are from the camera
            ratio = np.std(shape[:,0])
            disp = int(2.8*ratio)
            image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Check boundaries to prevent segmentation fault
            if (0 < point1[0]-disp < image_width and 0 < point1[0]+disp < image_width
            and 0 < point1[1]-disp < image_height and 0 < point1[1]+disp < image_height):
                emoji = cv2.imread(Emoji_File_Path, -1)
                emoji = cv2.resize(emoji, (2*disp,2*disp))

                x_offset = point1[0] - disp
                y_offset = point1[1] - disp
                y1, y2 = y_offset, y_offset + emoji.shape[0]
                x1, x2 = x_offset, x_offset + emoji.shape[1]

                alpha_s = emoji[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                # cv2.rectangle(image,(x1,y1), (x2,y2),(255,255,255),1)

                for c in range(0, 3):
                    image[y1:y2, x1:x2, c] = (alpha_s * emoji[:, :, c] +
                                            alpha_l * image[y1:y2, x1:x2, c])

    # show the gray image
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    width = image.shape[1]
    height = image.shape[0]
    
    cv2.imshow("image", image)
    
    # If you are using a 64-bit machine, you have to modif
    # cv2.waitKey(0) line as follows : k = cv2.waitKey(0) & 0xFF
    k = cv2.waitKey(1) 
    # press y to turn on emoji
    if k == ord('y'):
        drawEmoji = True
    # press n to turn off emoji
    elif k== ord('n'):
        drawEmoji = False
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
