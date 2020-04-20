"""
EmojiEmotion - CNN
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense

class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        # TODO: Build your own convolutional neural network, using Dropout at
        #       least once. The input image will be passed through each Keras
        #       layer in self.architecture sequentially. Refer to the imports
        #       to see what Keras layers you can use to build your network.
        #       Feel free to import other layers, but the layers already
        #       imported are enough for this assignment.
        #
        #       Remember: Your network must have under 15 million parameters!
        #       You will see a model summary when you run the program that
        #       displays the total number of parameters of your network.
        #
        #       Remember: Because this is a 15-scene classification task,
        #       the output dimension of the network must be 15. That is,
        #       passing a tensor of shape [batch_size, img_size, img_size, 1]
        #       into the network will produce an output of shape
        #       [batch_size, 15].
        #
        #       Note: Keras layers such as Conv2D and Dense give you the
        #             option of defining an activation function for the layer.
        #             For example, if you wanted ReLU activation on a Conv2D
        #             layer, you'd simply pass the string 'relu' to the
        #             activation parameter when instantiating the layer.
        #             While the choice of what activation functions you use
        #             is up to you, the final layer must use the softmax
        #             activation function so that the output of your network
        #             is a probability distribution.
        #
        #       Note: Flatten is a very useful layer. You shouldn't have to
        #             explicitly reshape any tensors anywhere in your network.
        #
        # ====================================================================
        #https://www.tensorflow.org/tutorials/images/cnn 

        self.architecture = []
        self.architecture.append(Conv2D(32,kernel_size=(7,7),activation='relu',strides=(1, 1),padding='valid'))
        self.architecture.append(Conv2D(32,kernel_size=(7,7),activation='relu',strides=(1, 1),padding='valid'))
        self.architecture.append(MaxPool2D(pool_size=(2, 2),strides=(2,2)))
        self.architecture.append(Conv2D(64,kernel_size=(7,7),activation='relu',strides=(1, 1),padding='valid'))
        self.architecture.append(Conv2D(64,kernel_size=(7,7),activation='relu',strides=(1, 1),padding='valid'))
        self.architecture.append(MaxPool2D(pool_size=(2, 2),strides=(2,2)))
        self.architecture.append(Dropout(0.2)) # https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
        self.architecture.append(Flatten())
        self.architecture.append(Dense(15, activation='softmax'))

        # https://towardsdatascience.com/the-quest-of-higher-accuracy-for-cnn-models-42df5d731faf
        # ====================================================================

    def call(self, img):
        """ Passes input image through the network. """

        for layer in self.architecture:
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
