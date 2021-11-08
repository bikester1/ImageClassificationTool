"""This module is for holding different neural networks and their logic."""
import datetime
from abc import abstractmethod

import cachetools
import keras
import numpy as np
import tensorflow as tf
from keras.activations import gelu
from tensorflow.keras.activations import relu, sigmoid
from tensorflow import optimizers, losses
from tensorflow.keras import Sequential, layers
from tensorflow.keras.regularizers import L2, L1
from tensorflow.keras.initializers import VarianceScaling
from tensorflow import random
from tensorflow import math as tf_math


from data import ImageData, DataSet
from protocols import Observable, updates


class NNModel(Observable):
    """NNModel is an abstract interface for all neural
    network models used for image classification.
    It specifies a load and save function so that models
    can be easily saved to disk and loaded on demand.
    """

    # pylint: disable=super-init-not-called
    def __init__(self):
        super().__default_init_implementation__()

    @abstractmethod
    def single_prediction(self, img: ImageData) -> dict[str, float]:
        """Runs a single image through the neural network
        and outputs a dictionary of predicted outputs.
        Output tags depend on initialized tags for the nn.

        :param img: image to be analyzed
        :return: a dictionary with the tag name as key and predicted likelihood as a float.
        """

    @abstractmethod
    def fit_model(self, epochs: int = 1):
        """Takes an integer number of epochs to train this NN for.
        Default is 1 epoch.

        :param epochs: Number of epochs to train for.
        :return: None.
        """

    @abstractmethod
    def save_model(self, file_path: str):
        """Saves entire model to file_path. Note: this is not the same as saving weights.

        :param file_path: Path at which to save model.
        :return: None.
        """

    @abstractmethod
    def load_model(self, file_path: str):
        """Loads entire model to file_path. Note: this is not the same as loading weights.

        :param file_path: Path at which to load model.
        :return: None.
        """


#pylint: disable=too-many-instance-attributes
class ModelBaseClass(NNModel):
    """Base implementation of the model interface.
    Most Models will fall in line with this implementation and
    should be purely for configuration
    """
    #pylint: disable=too-many-arguments
    def __init__(self, output_tags: list[str], training_set: DataSet,
                 percent_validation=20, model: Sequential = Sequential(),
                 min_training_set_size=5000):
        super().__init__()
        self.output_tags = output_tags

        # Training Sets
        self._training_set = training_set
        self._percent_validation = percent_validation
        self._imgs_train = np.array([])
        self._imgs_validation = np.array([])
        self._tags_train = np.array([])
        self._tags_validation = np.array([])
        self._min_training_set_size = min_training_set_size
        self.model = model

        self.update_training_sets()
        self._model_fit_history = []

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )

    def multi_hot_from_tag_set(self, tags: set[str]) -> np.array:
        """Creates a multi-hot encoded array from a set of tags.

        :param tags: Set of tags as strings to create a multi-hot encoded array.
        :return: multi-hot array.
        """
        print([int(tag in tags) for tag in self.output_tags])
        return np.array([int(tag in tags) for tag in self.output_tags])

    @staticmethod
    def mix(array_a: np.array, array_b: np.array, pct_a: float) -> np.array:
        """Mixes two arrays representing images by randomly taking a percentage of on images
        pixels over the other."""
        ret_arr = np.zeros(shape=np.shape(array_a))
        rand = np.random.random(size=np.shape(array_a)) - pct_a
        rand = np.ceil(rand)
        for item_a, item_b, out, item_rand in np.nditer([array_a, array_b, ret_arr, rand],
                                      flags=["external_loop", "refs_ok"],
                                      op_flags=[
                                          ["readonly"],
                                          ["readonly"],
                                          ["writeonly"],
                                          ["readonly"],
                                      ]):
            out[...] = item_a * item_rand + item_b * (1 - item_rand)
        return ret_arr

    @staticmethod
    def interp(array_a: list, array_b: list, pct_a: float) -> np.array:
        """Interpolates between two 1 dimensional lists. Mostly used to combine expected model
        outputs."""
        ret_arr = []
        for ind, _ in enumerate(array_a):
            ret_arr.append(array_a[ind] * pct_a + array_b[ind] * (1 - pct_a))
        return ret_arr

    @updates("imgs_train", "tags_train", "imgs_validation", "tags_validation")
    def update_training_sets(self):
        """Updates training arrays after the training set has been updated or replaced.

        :return: None.
        """
        # np_array_imgs = random.shuffle([
        #     img_dat.np_array / 255
        #     for img_dat, tag in self._training_set
        # ], 15)
        # np_array_tags = random.shuffle([
        #     self.multi_hot_from_tag_set(set(tag))
        #     for _, tag in self._training_set
        # ], 15)
        np_array_imgs = self._training_set.images_as_np_array
        np_array_tags = [
            self.multi_hot_from_tag_set(set(tag))
            for tag in self._training_set.labels
        ]
        training_set_imgs = []
        training_set_tags = []
        # while len(training_set_imgs) < self._min_training_set_size:
        #     print(len(training_set_imgs))
        #     a = random.randint(0, len(np_array_tags)-1)
        #     b = random.randint(0, len(np_array_tags)-1)
        #     pct = random.random()
        #     training_set_imgs.append(self.mix(np_array_imgs[a], np_array_imgs[b], pct))
        #     training_set_tags.append(self.interp(np_array_tags[a], np_array_tags[b], pct))
        training_set_tags = np_array_tags
        training_set_imgs = np_array_imgs

        test_size = (len(training_set_imgs) * self._percent_validation) // 100
        test_size -= 1
        #test_size = len(training_set_imgs) - 2
        self._imgs_train = np.array(training_set_imgs[:-test_size])
        self._tags_train = np.array(training_set_tags[:-test_size], dtype=np.float32)
        print(self._tags_train.shape)
        print(self._imgs_train.shape)

        self._imgs_validation = np.array(training_set_imgs[-test_size:])
        self._tags_validation = np.array(training_set_tags[-test_size:], dtype=np.float32)

    def print_weights(self):
        """Prints all the weights of the model to the terminal. Used primarily
        for debugging purposes.

        :return: None.
        """
        for layer in self.model.weights:
            average = np.average(layer)
            variance = np.var(layer)
            print(f"Average: {average}, variance: {variance}")

    @updates("model")
    def fit_model(self, epochs: int = 1):
        if len(self._imgs_train) == 0:
            return

        if len(self._imgs_validation) == 0:
            validation_data = None
        else:
            validation_data = (self._imgs_validation, self._tags_validation)

        history = self.model.fit(
            self._imgs_train,
            self._tags_train,
            batch_size=16,
            epochs=epochs,
            validation_data=validation_data,
            shuffle=True,
        )

    def single_prediction_array(self, array: np.array):
        x_pred = np.array([array])
        predictions = self.model.predict(x_pred)
        out = {}
        for tag, pred in zip(self.output_tags, predictions[0]):
            #print(f"{tag}: {pred}")
            out[tag] = pred

        return out

    def single_prediction(self, img: ImageData):
        x_pred = np.array([img.np_array/255])
        predictions = self.model.predict(x_pred)
        out = {}
        for tag, pred in zip(self.output_tags, predictions[0]):
            print(f"{tag}: {pred}")
            out[tag] = pred

        return out

    def save_model(self, file_path: str):
        self.model.save(file_path)

    @updates("model")
    def load_model(self, file_path: str):
        self.model = tf.keras.models.load_model(file_path)

    def intermediate_model_from_layer_index(self, layer: int):
        out_layer = self.model.layers[layer].output
        return keras.Model(inputs=self.model.input, outputs=out_layer)

    def output_from_layer(self, img: ImageData, layer: int):
        """Takes a model input and then returns the output at the specified layer"""
        print(self.model.layers)
        model = self.intermediate_model_from_layer_index(layer)
        print((img.np_array/255).shape)
        return model.predict(np.array([img.np_array/255]))


class ImageClassifierV01(ModelBaseClass):
    """Model version 1 for image classification."""

    @staticmethod
    def custom_regularizer(x):
        std_dev = (tf_math.reduce_std(x) - 1)
        mean = tf_math.reduce_mean(x) - 0.25
        loss = ((mean * mean * 2) + (std_dev * std_dev))/100
        return loss

    def __init__(self, output_tags: list[str], training_set: DataSet,
                 percent_validation=20):
        super().__init__(output_tags, training_set, percent_validation)

        self.model = Sequential([
            layers.Input(shape=(256, 256, 1)),
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.RandomFlip(),
            layers.experimental.preprocessing.RandomRotation(0.25),
            #layers.experimental.preprocessing.RandomContrast(0.5),
            layers.Conv2D(
                32,
                3,
                strides=1,
                padding="same",
                activation=gelu,
                kernel_initializer=VarianceScaling(),
                bias_initializer=VarianceScaling(),
                activity_regularizer=self.custom_regularizer,
            ),
            layers.MaxPool2D(2),
            layers.Conv2D(
                32,
                3,
                strides=1,
                padding="same",
                activation=gelu,
                kernel_initializer=VarianceScaling(),
                bias_initializer=VarianceScaling(),
                activity_regularizer=self.custom_regularizer,
            ),
            layers.MaxPool2D(2),
            layers.Conv2D(
                32,
                5,
                strides=3,
                padding="same",
                activation=gelu,
                kernel_initializer=VarianceScaling(),
                bias_initializer=VarianceScaling(),
                activity_regularizer=self.custom_regularizer,
            ),
            layers.MaxPool2D(2),
            layers.Conv2D(
                32,
                5,
                strides=3,
                padding="same",
                activation=gelu,
                kernel_initializer=VarianceScaling(),
                bias_initializer=VarianceScaling(),
                activity_regularizer=self.custom_regularizer,
            ),
            layers.Flatten(),
            #layers.Dense(32, gelu),
            layers.Dense(
                len(self.output_tags),
                activation=sigmoid,
            ),
        ])

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, amsgrad=True, epsilon=0.01),
            loss=losses.BinaryCrossentropy(),
            metrics='binary_crossentropy',
        )
        self.model.summary()
        self.print_weights()
