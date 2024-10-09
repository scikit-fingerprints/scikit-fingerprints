"""Submodule providing the TQDM Settings data class."""

from typing import Dict, Any, Iterable
from tqdm.auto import tqdm


class TQDMSettings:
    """Class to store TQDM settings."""

    def __init__(self):
        """Initialize the TQDM settings."""
        self._settings: Dict[str, Any] = {}

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

    def position(self, position: int) -> "TQDMSettings":
        """Set position."""
        self._settings["position"] = position
        return self

    def ncols(self, ncols: int) -> "TQDMSettings":
        """Set number of columns."""
        self._settings["ncols"] = ncols
        return self

    def dynamic_ncols(self, dynamic_ncols: bool) -> "TQDMSettings":
        """Set dynamic number of columns."""
        self._settings["dynamic_ncols"] = dynamic_ncols
        return self

    def iterable(self, iterable: Iterable) -> "TQDMSettings":
        """Set iterable."""
        self._settings["iterable"] = iterable
        return self

    def into_tqdm(self) -> tqdm:
        """Return a new TQDM instance with the current settings."""
        return tqdm(**self._settings)
