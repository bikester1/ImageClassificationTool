from Hashing import *
from Training import *

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

import numpy as np


class ImageWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.label = QLabel()
        self.pix_map = QPixmap()
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.label)
    
    def set_image(self, img: ImageData):
        print("Image Preview Updated")
        img = QImage(img.np_array, img.np_array.shape[1], img.np_array.shape[0], QImage.Format.Format_Mono)
        self.pix_map = QPixmap(img)
        self.label.setPixmap(self.pix_map)


class ModelWidget(QWidget):
    
    def __init__(self, model: NNModel, current_image: ImageData, registered_model_name: str = "model",
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
        return {
            self.registered_cur_img_name: self.current_image_updated,
            self.registered_model_name: self.model_updated,
        }
    
    def model_updated(self, model: NNModel):
        self.model = model
        self.update_widget()
    
    def current_image_updated(self, img: ImageData):
        self.current_image = img
        self.update_widget()
    
    def update_widget(self):
        text_dict = self.model.single_prediction(self.current_image)
        out = []
        for key, value in text_dict.items():
            out.append(f"{key}: {value}\n")
        
        out = "".join(out)
        self.prediction_output.setText(out)
        pass
