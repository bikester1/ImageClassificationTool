"""
This module contains controllers which create and manage GUI/Business logic objects.
"""
import warnings
from pathlib import Path
from typing import TypeVar, Union

from PyQt6.QtWidgets import QMainWindow, QTextEdit, QPushButton, QFileDialog, QMenuBar, \
    QGridLayout, QWidget

from protocols import Observable, updates
from tagging import ImageTagging
from data import ImageData, DogDataSet, PersonalPicturesDataSet
from gui import ImageWidget, ModelWidget, ModelFilterVisualization
from training import NNModel, ImageClassifierV01, ModelBaseClass

_classT = TypeVar("_classT")


class ClassNotAvailable(Exception):
    """Deprecated"""


class Controller(Observable):
    """Base class to define implementation and abstraction for a controller.
    Controllers create GUI widgets and business logic objects and coordinates
    data between the two.
    """
    # TODO: Implement Base Class

    def __init__(self):
        super().__default_init_implementation__()

    @property
    def data_connections(self) -> list[type]:
        """Deprecated"""
        warnings.warn("test", DeprecationWarning)
        return []

    def missing_data(self, data: list[type]) -> list[type]:
        """ Returns true if data is available """
        missing_data = [data_type for data_type in data if data_type not in self.data_connections]
        return missing_data


class TrainingController(Controller):
    """Controller for training and visualizing model.
    Creates and manages a NN Model to be trained on images.
    """
    def __init__(self):
        super().__init__()
        # Business Logic Objects
        self.image_tagger = ImageTagging()

        training_set = [
            (ImageData(fil), self.image_tagger.tagged_images[fil.name]["Tags"])
            for fil in self.image_tagger.all_images
            if fil.name in self.image_tagger.tagged_images
        ]
        training_set = PersonalPicturesDataSet()
        training_set.load_data()
        dog_set = DogDataSet()
        dog_set.load_data(load_arrays=False)
        dog_set.shuffle_array()
        dog_set = dog_set[:10000]
        dog_set._load_arrays()
        labels = {label for img in dog_set.labels for label in img}
        print(f"{len(training_set)} images for training")
        self.model: ModelBaseClass = ImageClassifierV01(list(labels), dog_set)
        self.model.fit_model(1)
        self.model.attach(lambda _, x=self: x.dispatch_update("model"), "model")
        self.current_image = ImageData(self.image_tagger.get_next_image())

        # GUI Objects
        self.main_window_widget = QMainWindow()

        self.input_text_box = QTextEdit()

        self.img_preview = ImageWidget()
        self.img_preview.set_image(self.current_image)
        self.attach(lambda _, x=self: self.img_preview.set_image(x.current_image), "current_image")

        layer = 4
        self.layer_preview = ModelFilterVisualization(5)
        self.attach(lambda _, x=self: self.layer_preview.update_images(
            x.model.output_from_layer(x.current_image, layer)), "model")
        self.attach(lambda _, x=self: self.layer_preview.update_images(
            x.model.output_from_layer(x.current_image, layer)), "current_image")
        self.layer_preview.update_images(self.model.output_from_layer(self.current_image, layer))

        self.model_preview = ModelWidget(self.model, self.current_image)
        self.attach(lambda _, x=self: self.model_preview.model_updated(x.model), "model")
        self.attach(lambda _, x=self:
                    self.model_preview.current_image_updated(x.current_image), "current_image")

        self.next_img_button = QPushButton()
        self.next_img_button.setText("Next Image")
        self.next_img_button.pressed.connect(self.next_button_pressed)

        self.save_tags_button = QPushButton()
        self.save_tags_button.setText("Set Tags")
        self.save_tags_button.pressed.connect(self.save_tags_button_pressed)

        # Dialogs
        self.file_save_dialog = QFileDialog()
        self.file_save_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        self.file_save_dialog.setFileMode(QFileDialog.FileMode.Directory)
        self.file_save_dialog.fileSelected.connect(self.save_nn)
        self.file_load_dialog = QFileDialog()
        self.file_load_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        self.file_load_dialog.setFileMode(QFileDialog.FileMode.Directory)
        self.file_load_dialog.fileSelected.connect(self.load_nn)

        self.menu_bar = QMenuBar()
        self.layout_menu_bar()

        self.layout = QGridLayout()
        self._layout_widgets()

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.layout)

        self.main_window_widget.setCentralWidget(self.main_widget)
        self.main_window_widget.show()

    def _layout_widgets(self):
        """Class method for specifying the layout of widgets.

        :return: None.
        """
        self.layout.addWidget(self.img_preview, 3, 1)
        self.layout.addWidget(self.model_preview, 3, 3)
        self.layout.addWidget(self.input_text_box, 5, 1, 1, 3)
        self.layout.addWidget(self.next_img_button, 7, 1)
        self.layout.addWidget(self.save_tags_button, 7, 3)
        self.layout.addWidget(self.layer_preview, 9, 1, 1, 3)

    def layout_menu_bar(self):
        """Class method for specifying the layout of menu bar.

        :return: None.
        """
        file_menu = self.menu_bar.addMenu("File")
        file_menu.addAction("Save NN", self.save_nn_button_pressed)
        file_menu.addAction("Load NN", self.load_nn_button_pressed)

        self.main_window_widget.setMenuBar(self.menu_bar)

    @updates("current_image")
    def set_current_image(self, img: Union[Path, ImageData]):
        """ Sets the current image attribute and fires an update to observers of
        current_image.

        :param img: image to be displayed
        :return: None.
        """
        if isinstance(img, Path):
            img = ImageData(img)
        self.current_image = img

    @updates("model")
    def fit_model(self):
        """Fits the model and fires an update to observers of model.

        :return: None.
        """
        self.model.fit_model(10)

    def next_button_pressed(self):
        """Called when the next button is pressed in the GUI

        :return: None.
        """
        self.set_current_image(self.image_tagger.get_next_image())
        self.fit_model()

    def save_tags_button_pressed(self):
        """Called when the next button is pressed in the GUI

        :return: None.
        """
        tokens = self.input_text_box.toPlainText().split(",")
        self.image_tagger.tag_image(self.current_image.file_path.name, tokens)
        self.set_current_image(self.image_tagger.get_next_image())

    def save_nn_button_pressed(self):
        """Called when the save nn button is pressed in the GUI.
        Only opens the dialog menu.

        :return: None.
        """
        self.file_save_dialog.show()

    def load_nn_button_pressed(self):
        """Called when the load nn button is pressed in the GUI
        Only opens the dialog menu.

        :return: None.
        """
        self.file_load_dialog.show()

    def save_nn(self):
        """Called when a folder is selected for saving nn. Actually saves to disk

        :return: None.
        """
        file_path = self.file_save_dialog.selectedFiles()[0]
        self.model.save_model(file_path)
        print(f"File Saved {file_path}")

    @updates("model")
    def load_nn(self, _: int):
        """Called when a folder is selected for loading nn. Actually loads from disk.

        :return: None.
        """
        file_path = self.file_load_dialog.selectedFiles()[0]
        self.model.load_model(file_path)
        print(f"File Loaded {file_path}")
