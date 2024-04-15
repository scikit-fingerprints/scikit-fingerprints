from contextlib import contextmanager

from rdkit.rdBase import BlockLogs


@contextmanager
def no_rdkit_logs():
    try:
        _rdkit_logs_blocker = BlockLogs()
        yield _rdkit_logs_blocker
    finally:
        del _rdkit_logs_blocker
