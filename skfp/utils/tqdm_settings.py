"""Submodule providing the TQDM Settings data class."""

from typing import Dict, Any
from tqdm.auto import tqdm


class TQDMSettings:
    """Class to store TQDM settings."""

    def __init__(self):
        """Initialize the TQDM settings."""
        self._settings: Dict[str, Any] = {}
        # Initialize with default settings
        self.leave(True).dynamic_ncols(False).desc("")

    def disable(self) -> "TQDMSettings":
        """Disable TQDM."""
        self._settings["disable"] = True
        return self

    def is_disabled(self) -> bool:
        """Check if TQDM is disabled."""
        return self._settings.get("disable", False)

    def total(self, total: int) -> "TQDMSettings":
        """Set total number of iterations."""
        self._settings["total"] = total
        return self

    def desc(self, desc: str) -> "TQDMSettings":
        """Set description."""
        self._settings["desc"] = desc
        return self

    def leave(self, leave: bool) -> "TQDMSettings":
        """Set leave flag."""
        self._settings["leave"] = leave
        return self

    def dynamic_ncols(self, dynamic_ncols: bool) -> "TQDMSettings":
        """Set dynamic number of columns."""
        self._settings["dynamic_ncols"] = dynamic_ncols
        return self

    def into_tqdm(self) -> tqdm:
        """Return a new TQDM instance with the current settings."""
        return tqdm(**self._settings)
