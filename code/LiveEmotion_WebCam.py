"""
EmojiEmotion - Live Emotion WebCam
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
    - numpy, matplotlib, scipy, dlib, 
"""

__author__ = "Lucas Schroeder, Christopher Luke, Antohony Peng"
__contact__ = "lucas_schroeder@brown.edu"
__date__ = "2020/04/21"

import sys 
import argparse
from imutils import face_utils
import dlib
import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--OnlineModel',
        default='false',
        help='Set this flag to "true" is you want to compare our model to one found online')

    return parser.parse_args()

# This method will load the weights that our CNN has produced
def load_model_from_path(path):
    json_file = open(path + 'chris_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    model = model_from_json(loaded_model_json)
    model.load_weights(path + "fer_dataaug.h5")
    return model

# We use this method to predict the emotion of a subject. It takes in 
# the whole webcam image and then resizes it according to the bounding 
# box around the face. 
# 
# Before using the model to predict the emotion, we resize it to a either 
# a 48x48 image (like the image samples from the FER2013 dataset) if we 
# are using our model, or a 64x64 image to work with the online emotion.
def predict_emotion(gray, x, y, w, h):
    face = np.expand_dims(np.expand_dims(cv2.resize(gray[y:y+w, x:x+h]/255, (input_imageShape, input_imageShape)),-1), 0)
    prediction = model.predict([face])

    return(int(np.argmax(prediction)), round(max(prediction[0])*100, 2))

ARGS = parse_args()
# we can use these lines to load the model
if ARGS.OnlineModel == 'false':
    path = "Models/"
    input_imageShape = 48
    model = load_model_from_path(path)
    print('Your are using our trained model.')
else:
    
    path = 'Models/OnlineModel/_mini_XCEPTION.102-0.66.hdf5'
    input_imageShape = 64
    model = load_model(path, compile=False)
    print('You are using an online model')

# emotion dictionary for the classification and the file paths
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
emoji_path_dict = {0: "EmojiImages/AngryEmoji.png", 1: "EmojiImages/DisgustEmoji.png", 2: "EmojiImages/ScaredEmoji.png", 3: "EmojiImages/SmilingEmoji.png", 4: "EmojiImages/SadEmoji.png", 5: "EmojiImages/SurprisedEmoji.png", 6: "EmojiImages/NeutralEmoji.png"}

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# create an overlay image. You can use any image
overlaidImage = np.ones((100,100,3),dtype='uint8')*255

cap = cv2.VideoCapture(0)

# This is a boolean that can be set to True or False to turn the emoji feature on and off. The program
# reacts to the 'y' and 'n' keys to turn the emoji on (y) and off (n). 
drawEmoji = True
previousEmotion = 0
while True:
    
    # Get the image from the webcam and convert it into a gray image scale
    _, image = cap.read()
    if drawEmoji == True:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(gray, 0)
        
        # For each detected face, find the landmarks.
        for (i, rect) in enumerate(rects):
            # compute the bounding box of the face and draw it on the image
            (cornerX, cornerY, squareWidth, squareHeight) = face_utils.rect_to_bb(rect)
            cornerX = cornerX +10
            cornerY = cornerY -10
            squareWidth = squareWidth - 20
            squareHeight = squareHeight +10
            
            # predict the emotion by passing the computed bounding box into the prediction function
            emotion_id, confidence = predict_emotion(gray, cornerX, cornerY, squareWidth, squareHeight)
            
            # This is a threshold that makes sure the emotion does not continuously change
            # if the confidence of the predicted emotion is low
            if confidence < 20:
                emotion_id = previousEmotion
            previousEmotion = emotion_id

            emotion = emotion_dict[emotion_id]
            cv2.putText(image, emotion,(cornerX+20,cornerY-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), lineType=cv2.LINE_AA)
            # get the file path to the correct emoji image
            Emoji_File_Path= emoji_path_dict[emotion_id]

            # Predict the landmarks of the face and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            point1 = shape[29,:]
            
            # Get the standard deviation between points to calculate the ratio 
            # relating to how far you are from the camera
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

                # Finally, put the emoji on the image in the correct location
                for c in range(0, 3):
                    image[y1:y2, x1:x2, c] = (alpha_s * emoji[:, :, c] +
                                            alpha_l * image[y1:y2, x1:x2, c])

    # show the image
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    width = image.shape[1]
    height = image.shape[0]
    
    cv2.imshow("image", image)
    
    # If you are using a 64-bit machine, you have to modify
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
cv2.destroyAllWindows()
cap.release()
