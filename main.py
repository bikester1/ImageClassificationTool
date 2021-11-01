"""Main entry point for python program"""
# pylint: disable-next=line-too-long
# pylint --rcfile=pylintrc.lintrc controllers data gui hashing image_preprocessing main meta_data protocols tagging tests training

import subprocess
from pathlib import Path
from subprocess import Popen

import psutil
from PyQt6.QtWidgets import QApplication
from psutil import Process

from controllers import TrainingController


def stop_mining():
    """Stops my running etherminers."""
    for pid in psutil.pids():
        try:
            proc: Process = Process(pid)
        except psutil.NoSuchProcess:
            continue

        if proc.name() == "ethminer.exe":
            etherminer_proc: Process = proc

    try:
        for parent in etherminer_proc.parents():
            if parent.name() == "supervisor.exe":
                sup_proc = parent

        for parent in sup_proc.parents():
            if parent.name() == "cmd.exe":
                cmd_proc = parent
    except UnboundLocalError:
        pass

    try:
        cmd_proc.terminate()
    except (UnboundLocalError, psutil.NoSuchProcess):
        pass

    try:
        sup_proc.terminate()
    except (UnboundLocalError, psutil.NoSuchProcess):
        pass

    try:
        etherminer_proc.terminate()
    except (UnboundLocalError, psutil.NoSuchProcess):
        pass


if __name__ == "__main__":
    stop_mining()
    app = QApplication([])
    test = TrainingController()

    app.exec()
    start_bat = Path(r"C:\Users\Brenden Cohen\Desktop\Personal Finances\Crypto\ETH\Start.bat")
    new_process = Popen(
        "Start.bat", cwd=r"C:\Users\Brenden Cohen\Desktop\Personal "
                         r"Finances\Crypto\ETH",
        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        )
