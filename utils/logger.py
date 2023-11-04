import contextlib
import joblib
import logging
import os
import sys

import rdkit
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

rdBase.LogToPythonLogger()


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class CaptureLogger(logging.Handler):
    """
    Handler for disabling logging from rdkit. Use it as a context manager.
    Code taken from - https://github.com/rdkit/rdkit/discussions/5435.
    """

    def __init__(self, module=None):
        super(CaptureLogger, self).__init__(level=logging.DEBUG)
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
