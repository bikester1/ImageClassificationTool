"""
This module contains all of the custom GUI widget elements.
"""
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout, QTextEdit, QFormLayout

from data import ImageData
from training import NNModel


class ImageWidget(QWidget):
    """
    This widget provides a preview window for a image set using the ImageData Class.
    """
    def __init__(self):
        super().__init__()
        self.label = QLabel()
        self.pix_map = QPixmap()
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.label)

    def set_image(self, img: ImageData):
        """
        Sets image to be previewed based on an np.array in ImageData
        :param img: image to be previewed
        :return: None.
        """
        print("Image Preview Updated")
        img = QImage(img.np_array, img.np_array.shape[0], img.np_array.shape[1],
                     QImage.Format.Format_Grayscale8)
        self.pix_map = QPixmap(img)
        self.label.setPixmap(self.pix_map)


class ModelWidget(QWidget):
    """
    This widget provides a visualization of a NNModel. It will also provide
    predictions when given an image to use.
    """
    def __init__(self, model: NNModel, current_image: ImageData,
                 registered_model_name: str = "model",
                 registered_cur_img_name: str = "current_image"):
        super().__init__()
        self.model = model
        self.current_image = current_image
        self.registered_model_name = registered_model_name
        self.registered_cur_img_name = registered_cur_img_name

        self.prediction_output = QTextEdit()
        self.prediction_output.setDisabled(True)

        self.setBaseSize(400, 400)

        self.setLayout(QFormLayout())
        self.layout().addRow("Predicted Values", self.prediction_output)

        self.current_image_updated(self.current_image)

    @property
    def update_callbacks(self) -> dict[str:callable]:
        """
        returns list of callbacks that can be used to register updates.
        :return: dictionary of callbacks by variable name
        """
        return {
            self.registered_cur_img_name: self.current_image_updated,
            self.registered_model_name: self.model_updated,
        }

    def model_updated(self, model: NNModel):
        """
        Call this method when the model this widget has gets updated.
        :param model: Always provide the current model in the event that it is changed.
        :return: None.
        """
        self.model = model
        self.update_widget()

    def current_image_updated(self, img: ImageData):
        """
        Call this method when the current_image this widget has gets updated.
        :param img: Always provide the current image in the event that it is changed.
        :return: None.
        """
        self.current_image = img
        self.update_widget()

    def update_widget(self):
        """
        Called to update the widget after variables are set.
        :return: None.
        """
        text_dict = self.model.single_prediction(self.current_image)
        out = []
        for key, value in text_dict.items():
            out.append(f"{key}: {value}\n")

        out = "".join(out)
        self.prediction_output.setText(out)
