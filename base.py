from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class FingerprintTransformer(ABC):
    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        pass
