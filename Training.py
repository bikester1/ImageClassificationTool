from Protocols import *
from ImagePreprocessing import *
from Data import *

from keras.optimizers import *
from keras.activations import *
from keras.initializers import *
from keras.losses import *
from tensorflow import *
from tensorflow.keras import *
from numpy import *
from matplotlib import pyplot as plt

import datetime
import numpy as np
import tensorflow as tf
import keras.optimizers


class NNModel(Observable):
    
    def single_prediction(self, img: ImageData) -> dict[str, float]:
        pass

    def fit_model(self, epochs: int = 1):
        pass
    # TODO: Implement Base type
    pass


class ImageClassifierV01(NNModel):
    
    def __init__(self, output_tags: list[str], image_set: list[(ImageData, list[str])], percent_validation=20):
        self.output_tags = output_tags
        
        # Training Sets
        self.training_set = image_set
        self.percent_validation = percent_validation
        self.imgs_train = np.array([])
        self.imgs_validation = np.array([])
        self.tags_train = np.array([])
        self.tags_validation = np.array([])
        
        self.update_training_sets()
        self.model_fit_history = []
        
        self.model = Sequential([
            layers.Input(shape=(512, 512, 1)),
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.RandomFlip('horizontal'),
            layers.experimental.preprocessing.RandomRotation(0.5),
            layers.experimental.preprocessing.RandomZoom((0.3, -0.3), (0.3, -0.3)),
            layers.MaxPool2D(2),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=leaky_relu,
            ),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=leaky_relu,
            ),
            layers.MaxPool2D(2),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=leaky_relu,
            ),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=leaky_relu,
            ),
            layers.MaxPool2D(2),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=leaky_relu,
            ),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=leaky_relu,
            ),
            layers.MaxPool2D(16),
            layers.Flatten(),
            layers.Dense(16, activation=leaky_relu, kernel_regularizer="L2",),
            layers.Dense(len(self.output_tags), activation=sigmoid,),
        ])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.BinaryCrossentropy(),
            metrics='binary_crossentropy',
        )
        self.model.summary()
        self.print_weights()
    
    def one_hot_from_tag_set(self, tags: set[str]) -> np.array:
        print([int(tag in tags) for tag in self.output_tags])
        return np.array([int(tag in tags) for tag in self.output_tags])
    
    @updates("imgs_train", "tags_train", "imgs_validation", "tags_validation")
    def update_training_sets(self):
        np_array_imgs = [img_dat.np_array/255 for img_dat, tag in self.training_set]
        np_array_tags = [self.one_hot_from_tag_set(set(tag)) for _, tag in self.training_set]
        
        test_size = (len(self.training_set) * self.percent_validation) // 100
        test_size -= 1
        self.imgs_train = np.array(np_array_imgs[:-test_size])
        self.tags_train = np.array(np_array_tags[:-test_size])
    
        self.imgs_validation = np.array(np_array_imgs[-test_size:])
        self.tags_validation = np.array(np_array_tags[-test_size:])
        
        #np.savez("\\NP\\img_train.npz",
        #         imgs_train=self.imgs_train,
        #         tags_train=self.tags_train,
        #         imgs_validation=self.imgs_validation,
        #         tags_validation=self.tags_validation
        #         )
    
    @updates("model")
    def fit_model(self, epochs: int = 1):
        if not len(self.imgs_train):
            return
        
        if not len(self.imgs_validation):
            validation_data = None
        else:
            validation_data = (self.imgs_validation, self.tags_validation)
        
        history = self.model.fit(
            self.imgs_train,
            self.tags_train,
            batch_size=100,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[self.tensorboard_callback],
        )
        self.model_fit_history.append(history)
        self.print_weights()

    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
  
        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure
    
    def print_weights(self):
        for layer in self.model.weights:
            average = np.average(layer)
            varaince = np.var(layer)
            print(f"Average: {average}, Variance: {varaince}")
    
    def single_prediction(self, img: ImageData):
        x_pred = np.array([img.np_array/255])
        predictions = self.model.predict(x_pred)
        out = {}
        for tag, pred in zip(self.output_tags, predictions[0]):
            print(f"{tag}: {pred}")
            out[tag] = pred
        
        return out
