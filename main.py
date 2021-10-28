"""
Pylint:

pylint --rcfile=pylintrc.lintrc controllers data gui hashing image_preprocessing main meta_data protocols tagging tests training
"""
from PyQt6.QtWidgets import QApplication

from controllers import TrainingController

if __name__ == "__main__":
    app = QApplication([])
    test = TrainingController()

    app.exec()
