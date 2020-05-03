"""
EmojiEmotion - CNN
CS1430 - Computer Vision
Brown University
"""

import numpy as np
import pandas as pd
# import tensorflow as tf
# #import hyperparameters as hp
# from tensorflow.keras.layers import \
#         Conv2D, MaxPool2D, Dropout, Flatten, Dense
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.losses import categorical_crossentropy
#from tensorflow.keras.utils import to_categorical


from keras import optimizers
from keras.utils import to_categorical
from keras.layers import *
from keras.models import Sequential
#from keras.layers import MaxPooling2D # Maxpooling function
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
#from tensorboard.utils import ImageLabelingLogger, ConfusionMatrixLogger

#install pandas, np_utils 


df=pd.read_csv('fer2013.csv')
df.dropna()

num_features = 64
num_labels = 7
batch_size = 32
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

x_training = np.array(x_training,'float32')/255
print(x_training[0])
y_training = np.array(y_training,'float32')
print(y_training[0])
x_testing = np.array(x_testing,'float32')/255
y_testing = np.array(y_testing,'float32')

y_training=to_categorical(y_training, num_classes=num_labels)
y_testing=to_categorical(y_testing, num_classes=num_labels)

#preprocess

x_training = np.divide(np.subtract(x_training, np.mean(x_training)), np.std(x_training))
#y_training = np.divide(np.subtract(y_training, np.mean(y_training)), np.std(y_training))

x_testing = np.divide(np.subtract(x_testing, np.mean(x_testing)), np.std(x_testing))
#y_testing = np.divide(np.subtract(y_testing, np.mean(y_testing)), np.std(y_testing))

x_training = x_training.reshape(x_training.shape[0], 48, 48, 1)
x_testing = x_testing.reshape(x_testing.shape[0], 48, 48, 1)

# model = Sequential([
#     # Block 1
#     Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1", input_shape=(x_training.shape[1:])),
#     Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2"),
#     MaxPool2D(2, name="block1_pool"),
#     tf.keras.layers.Dropout(0.2),
#     # Block 2
#     Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
#     Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2"),
#     MaxPool2D(2, name="block2_pool"),
#     tf.keras.layers.Dropout(0.2),
#     # Block 3, 
#     Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"),
#     Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2"),
#     Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3"),
#     MaxPool2D(2, name="block3_pool"),
#     tf.keras.layers.Dropout(0.2),
#     # Block 4
#     Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1"),
#     Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2"),
#     Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3"),
#     MaxPool2D(2, name="block4_pool"),
#     tf.keras.layers.Dropout(0.2),
#     # Block 5
#     Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1"),
#     Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2"),
#     Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3"),
#     MaxPool2D(2, name="block5_pool"),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation= "relu"),
#     tf.keras.layers.Dense(64, activation= "relu"),
#     tf.keras.layers.Dense(num_labels, activation= "softmax")
# ])

model = Sequential()
    
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.01)))
model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))
    
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
    
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
    
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
    
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))



learning_rate = 0.001
#model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr= learning_rate), metrics=['accuracy'])

adam = optimizers.Adam(lr = learning_rate)
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
"""callback_list = [
     tf.keras.callbacks.ModelCheckpoint(
            filepath="weights.e{epoch:02d}-" + \
                    "acc{val_sparse_categorical_accuracy:.4f}.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(x_training)
    ] """

print(model.summary())
model.fit(x_training, y_training, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_testing, y_testing), shuffle=True)

fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")