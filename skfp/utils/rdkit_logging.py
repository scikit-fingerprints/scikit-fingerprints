from contextlib import contextmanager

from rdkit.rdBase import BlockLogs, DisableLog, EnableLog


@contextmanager
def no_rdkit_logs(suppress_warnings: bool = False):
    """
    Context manager that disables RDKit logging. If ``suppress_warnings`` is True,
    also blocks all warnings and info messages.
    """
    try:
        if suppress_warnings:
            DisableLog("rdApp.*")

        _rdkit_logs_blocker = BlockLogs()
        yield _rdkit_logs_blocker
    finally:
        del _rdkit_logs_blocker
        if suppress_warnings:
            EnableLog("rdApp.*")
