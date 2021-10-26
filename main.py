from PyQt6.QtWidgets import QApplication

from controllers import TrainingController

if __name__ == "__main__":
    app = QApplication([])
    test = TrainingController()

    app.exec()
