"""This module is for holding different neural networks and their logic."""
import datetime
import random
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import relu, sigmoid
from tensorflow import optimizers, losses
from tensorflow.keras import Sequential, layers

from data import ImageData
from protocols import Observable, updates


class NNModel(Observable):
    """NNModel is an abstract interface for all neural
    network models used for image classification.
    It specifies a load and save function so that models
    can be easily saved to disk and loaded on demand.
    """

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


class ModelBaseClass(NNModel):
    """Base implementation of the model interface.
    Most Models will fall in line with this implementation and
    should be purely for configuration
    """
    def __init__(self, output_tags: list[str], image_set: list[(ImageData, list[str])],
                 percent_validation=20, model: Sequential = Sequential(),
                 min_training_set_size=5000):
        super().__init__()
        self.output_tags = output_tags

        # Training Sets
        self._training_set = image_set
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
    def mix(a: np.array, b: np.array, pct_a: float) -> np.array:
        ret_arr = np.zeros(shape=np.shape(a))
        rand = np.random.random(size=np.shape(a)) - pct_a
        rand = np.ceil(rand)
        for x, y, out, r in np.nditer([a, b, ret_arr, rand],
                                      flags=["external_loop", "refs_ok"],
                                      op_flags=[
                                          ["readonly"],
                                          ["readonly"],
                                          ["writeonly"],
                                          ["readonly"],
                                      ]):
            out[...] = x * r + y * (1 - r)
        return ret_arr

    @staticmethod
    def interp(a: list, b: list, pct_a: float) -> np.array:
        ret_arr = []
        for x in range(len(a)):
            ret_arr.append(a[x] * pct_a + b[x] * (1 - pct_a))
        return ret_arr

    @updates("imgs_train", "tags_train", "imgs_validation", "tags_validation")
    def update_training_sets(self):
        """Updates training arrays after the training set has been updated or replaced.

        :return: None.
        """
        np_array_imgs = [img_dat.np_array / 255 for img_dat, tag in self._training_set]
        np_array_tags = [self.multi_hot_from_tag_set(set(tag)) for _, tag in self._training_set]
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
        self._imgs_train = np.array(training_set_imgs[:-test_size])
        self._tags_train = np.array(training_set_tags[:-test_size], dtype=np.float32)

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
            batch_size=100,
            epochs=epochs,
            validation_data=validation_data,
        )
        self._model_fit_history.append(history)
        self.print_weights()

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


class ImageClassifierV01(ModelBaseClass):
    """Model version 1 for image classification."""
    def __init__(self, output_tags: list[str], image_set: list[(ImageData, list[str])],
                 percent_validation=20):
        super().__init__(output_tags, image_set, percent_validation)

        self.model = Sequential([
            layers.Input(shape=(256, 256, 1)),
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.RandomFlip('horizontal'),
            layers.experimental.preprocessing.RandomRotation(0.5),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=relu,
            ),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=relu,
            ),
            layers.MaxPool2D(2),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=relu,
            ),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=relu,
            ),
            layers.MaxPool2D(2),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=relu,
            ),
            layers.Conv2D(
                10,
                3,
                strides=1,
                padding="same",
                activation=relu,
            ),
            layers.MaxPool2D(16),
            layers.Flatten(),
            layers.Dense(16, activation=relu,),
            layers.Dense(len(self.output_tags), activation=sigmoid,),
        ])

        self.model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.BinaryCrossentropy(),
            metrics='binary_crossentropy',
        )
        self.model.summary()
        self.print_weights()
