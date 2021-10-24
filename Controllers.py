from GUI import *
from Training import *
from Protocols import *
from Tagging import *

from typing import TypeVar
from PyQt6.QtWidgets import *


_classT = TypeVar("_classT")


class ClassNotAvailable(Exception):
    pass


class Controller(Observable):
    # TODO: Implement Base Class
    
    @property
    def data_connections(self) -> list[type]:
        pass
        return []
    
    def missing_data(self, data: list[type]) -> list[type]:
        """ Returns true if data is available """
        missing_data = [data_type for data_type in data if data_type not in self.data_connections]
        return missing_data
    
    def create_and_register_callbacks(self, _classT: type, *args: list[callable]) -> _classT:
        instance = _classT(args)


class TrainingController(Controller):
    
    def __init__(self):
        self.image_tagger = ImageTagging()
        
        training_set = [
            (ImageData(fil), self.image_tagger.tagged_images[fil.name]["Tags"])
            for fil in self.image_tagger.all_images[:100]
            if fil.name in self.image_tagger.tagged_images
        ]
        print(f"{len(training_set)} images for training")
        self.model: NNModel = ImageClassifierV01(["duke"], training_set)
        self.model.fit_model(1)
        self.current_image = ImageData(self.image_tagger.get_next_image())
        self.main_window_widget = QMainWindow()
        
        self.input_text_box = QTextEdit()
        
        self.img_preview = ImageWidget()
        self.img_preview.set_image(self.current_image)
        self.attach(lambda _, x=self: self.img_preview.set_image(x.current_image), "current_image")
        
        self.model_preview = ModelWidget(self.model, self.current_image)
        self.attach(lambda _, x=self: self.model_preview.model_updated(x.model), "model")
        self.attach(lambda _, x=self: self.model_preview.current_image_updated(x.current_image), "current_image")
        
        self.next_img_button = QPushButton()
        self.next_img_button.setText("Next Image")
        self.next_img_button.pressed.connect(self.next_button_pressed)
        
        self.file_dialog = QFileDialog()
        self.file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        self.file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        self.file_dialog.fileSelected.connect(self.save_nn)
        self.save_nn_button = QPushButton()
        self.save_nn_button.setText("Save NN")
        self.save_nn_button.pressed.connect(self.save_nn_button_pressed)
        
        self.menu_bar = QMenuBar()
        self.layout_menu_bar()
        
        self.layout = QGridLayout()
        self.layout_widgets()
        
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.layout)
        
        self.main_window_widget.setCentralWidget(self.main_widget)
        self.main_window_widget.show()
    
    def layout_widgets(self):
        self.layout.addWidget(self.img_preview, 3, 1)
        self.layout.addWidget(self.model_preview, 3, 3)
        self.layout.addWidget(self.input_text_box, 5, 1, 1, 3)
        self.layout.addWidget(self.next_img_button, 7, 1)
        self.layout.addWidget(self.save_nn_button, 7, 3)
    
    def layout_menu_bar(self):
        file_menu = self.menu_bar.addMenu("File")
        file_menu.addAction("Save NN", self.save_nn_button_pressed)
        file_menu.addAction("Load NN", self.load_nn_button_pressed)
        
        self.main_window_widget.setMenuBar(self.menu_bar)
    
    @updates("current_image")
    def set_current_image(self, img: Union[Path, ImageData]):
        if isinstance(img, Path):
            img = ImageData(img)
        self.current_image = img
    
    @updates("model")
    def fit_model(self):
        self.model.fit_model(10)
    
    def next_button_pressed(self):
        self.set_current_image(self.image_tagger.get_next_image())
        self.fit_model()
        
    def save_nn_button_pressed(self):
        self.file_dialog.show()
    
    def load_nn_button_pressed(self):
        self.file_dialog.show()
    
    def save_nn(self):
        file_path = self.file_dialog.selectedFiles()[0]
        self.model.save_model(file_path)
        print(f"File Saved {file_path}")
        
    def save_nn(self):
        file_path = self.file_dialog.selectedFiles()[0]
        self.model.save_model(file_path)
        print(f"File Saved {file_path}")


