import logging
import os
import sys
from typing import Optional

import rdkit
from joblib import Parallel
from rdkit import rdBase
from tqdm import tqdm

rdBase.LogToPythonLogger()


class ProgressParallel(Parallel):
    def __init__(self, total: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total = total

    def __call__(self, *args, **kwargs):
        with tqdm(total=self.total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


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

    def release(self):
        rdkit.log_handler.setStream(sys.stderr)
        rdkit.logger.removeHandler(self)
        self.devnull.close()
        return self.logs
