"""
EmojiEmotion - CNN
CS1430 - Computer Vision
Brown University
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense


df=pd.read_csv('fer2013.csv')

num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width, height = 48, 48

x_training, y_training, x_testing, y_testing = [], [], [], []

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    if 'Training' in row['Usage']:
        x_training.append(np.array(val,'float32'))
        y_training.append(row['emotion'])

    elif 'PublicTest' in row['Usage']:
        x_testing.append(np.array(val,'float32'))
        y_testing.append(row['emotion'])

x_training = np.array(x_training,'float32')
y_training = np.array(y_training,'float32')
x_testing = np.array(x_testing,'float32')
y_testing = np.array(y_testing,'float32')

#preprocess

x_training = np.divide(np.subtract(x_training - np.mean(x_training)), np.std(x_training))
y_training = np.divide(np.subtract(y_training - np.mean(y_training)), np.std(y_training))

x_testing = np.divide(np.subtract(x_testing - np.mean(x_testing)), np.std(x_testing))
y_testing = np.divide(np.subtract(y_testing - np.mean(y_testing)), np.std(y_testing))

model = Sequential([
    # Block 1
    Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"),
    Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2"),
    MaxPool2D(2, name="block1_pool"),
    # Block 2
    Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
    Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2"),
    MaxPool2D(2, name="block2_pool"),
    # Block 3
    Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"),
    Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2"),
    Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3"),
    MaxPool2D(2, name="block3_pool"),
    # Block 4
    Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1"),
    Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2"),
    Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3"),
    MaxPool2D(2, name="block4_pool"),
    # Block 5
    Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1"),
    Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2"),
    Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3"),
    MaxPool2D(2, name="block5_pool"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_labels, activation= "softmax")
])

model.fit(x_training, y_training, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_testing, y_testing), shuffle=True)

fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")