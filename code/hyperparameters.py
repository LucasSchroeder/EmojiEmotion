"""
EmojiEmotion - CNN
CS1430 - Computer Vision
Brown University
"""

num_features = 64

# Resize image size for task 1. Task 2 must have an image size of 224,
# so that is hard coded in run.py in the main function.
width, height = 48, 48

# The number of image scene classes. Don't change this.
num_labels = 7

# Sample size for calculating the mean and standard deviation of the
# training data. This many images will be randomly seleted to be read
# into memory temporarily.
#preprocess_sample_size = 400

# Training parameters

# num_epochs is the number of epochs. If you experiment with more
# complex networks you might need to increase this. Likewise if you add
# regularization that slows training.
epochs = 30

# batch_size defines the number of training examples per batch:
batch_size = 32

# learning_rate is a critical parameter that can dramatically affect
# whether training succeeds or fails. For most of the experiments in this
# project the default learning rate is safe.
learning_rate = 0.001

# Momentum on the gradient (if you use a momentum-based optimizer)
#momentum = 0.01
