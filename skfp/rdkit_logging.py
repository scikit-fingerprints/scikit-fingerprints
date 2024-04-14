import logging
import os
import sys

import rdkit
from rdkit import rdBase

rdBase.LogToPythonLogger()


class CaptureLogger(logging.Handler):
    """
    Handler for disabling logging from rdkit. Use it as a context manager.
    Code taken from - https://github.com/rdkit/rdkit/discussions/5435.
    """

    def __init__(self, module=None):
        super().__init__(level=logging.DEBUG)
        self.logs = {}
        self.devnull = open(os.devnull, "w")
        rdkit.log_handler.setStream(self.devnull)
        rdkit.logger.addHandler(self)

    def __enter__(self):
        return self.logs

    def __exit__(self, *args):
        self.release()

    def handle(self, record) -> bool:
        key = record.levelname
        val = self.format(record)
        self.logs[key] = self.logs.get(key, "") + val
        return False

    def release(self) -> None:
        rdkit.log_handler.setStream(sys.stderr)
        rdkit.logger.removeHandler(self)
        self.devnull.close()
