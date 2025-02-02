"""
EmojiEmotion - CNN
CS1430 - Computer Vision
Brown University
"""

import io
import os
import sklearn.metrics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import hyperparameters as hp

def plot_to_image(figure):
    """ Converts a pyplot figure to an image tensor. """

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image

class ImageLabelingLogger(tf.keras.callbacks.Callback):
    """ Keras callback for logging a plot of test images and their
    predicted labels for viewing in Tensorboard. """

    def __init__(self, datasets):
        super(ImageLabelingLogger, self).__init__()

        self.datasets = datasets
        self.task = datasets.task

        print("Done setting up image labeling logger.")

    def on_epoch_end(self, epoch, logs=None):
        self.log_image_labels(epoch, logs)

    def log_image_labels(self, epoch_num, logs):
        """ Writes a plot of test images and their predicted labels
        to disk. """

        fig = plt.figure(figsize=(9, 9))
        count = 0
        for batch in self.datasets.train_data:
            for i, image in enumerate(batch[0]):
                plt.subplot(5, 5, count+1)

                correct_class_idx = batch[1][i]
                probabilities = self.model(np.array([image])).numpy()[0]
                predict_class_idx = np.argmax(probabilities)

                if self.task == '1':
                    image = np.clip(image, 0., 1.)
                    plt.imshow(image, cmap='gray')
                else:
                    # Undo VGG preprocessing
                    mean = [103.939, 116.779, 123.68]
                    image[..., 0] += mean[0]
                    image[..., 1] += mean[1]
                    image[..., 2] += mean[2]
                    image = image[:, :, ::-1]
                    image = image / 255.
                    image = np.clip(image, 0., 1.)

                    plt.imshow(image)

                is_correct = correct_class_idx == predict_class_idx

                title_color = 'g' if is_correct else 'r'

                plt.title(self.datasets.idx_to_class[predict_class_idx], color=title_color)
                plt.axis('off')

                count += 1
                if count == 25:
                    break

            if count == 25:
                break

        figure_img = plot_to_image(fig)

        file_writer_il = tf.summary.create_file_writer('logs/image_labels')

        with file_writer_il.as_default():
            tf.summary.image("Image Label Predictions", figure_img, step=epoch_num)


class ConfusionMatrixLogger(tf.keras.callbacks.Callback):
    """ Keras callback for logging a confusion matrix for viewing
    in Tensorboard. """

    def __init__(self, datasets):
        super(ConfusionMatrixLogger, self).__init__()

        self.datasets = datasets

    def on_epoch_end(self, epoch, logs=None):
        self.log_confusion_matrix(epoch, logs)

    def log_confusion_matrix(self, epoch, logs):
        """ Writes a confusion matrix plot to disk. """

        test_pred = []
        test_true = []
        count = 0
        for i in self.datasets.test_data:
            test_pred.append(self.model.predict(i[0]))
            test_true.append(i[1])
            count += 1
            if count >= 1500 / hp.batch_size:
                break

        test_pred = np.array(test_pred)
        test_pred = np.argmax(test_pred, axis=-1).flatten()
        test_true = np.array(test_true).flatten()

        # Source: https://www.tensorflow.org/tensorboard/image_summaries
        cm = sklearn.metrics.confusion_matrix(test_true, test_pred)
        figure = self.plot_confusion_matrix(cm, class_names=self.datasets.classes)
        cm_image = plot_to_image(figure)

        file_writer_cm = tf.summary.create_file_writer('logs/confusion_matrix')

        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix (on validation set)", cm_image, step=epoch)

    def plot_confusion_matrix(self, cm, class_names):
        """ Plots a confusion matrix returned by
        sklearn.metrics.confusion_matrix(). """

        # Source: https://www.tensorflow.org/tensorboard/image_summaries
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        threshold = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return figure
