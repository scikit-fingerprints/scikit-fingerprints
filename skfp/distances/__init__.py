import inspect
import sys

from .braun_blanquet import (
    braun_blanquet_binary_distance,
    braun_blanquet_binary_similarity,
    bulk_braun_blanquet_binary_distance,
    bulk_braun_blanquet_binary_similarity,
)
from .ct4 import (
    bulk_ct4_binary_distance,
    bulk_ct4_binary_similarity,
    bulk_ct4_count_distance,
    bulk_ct4_count_similarity,
    ct4_binary_distance,
    ct4_binary_similarity,
    ct4_count_distance,
    ct4_count_similarity,
)
from .dice import (
    bulk_dice_binary_distance,
    bulk_dice_binary_similarity,
    bulk_dice_count_distance,
    bulk_dice_count_similarity,
    dice_binary_distance,
    dice_binary_similarity,
    dice_count_distance,
    dice_count_similarity,
)
from .fraggle import (
    bulk_fraggle_distance,
    bulk_fraggle_similarity,
    fraggle_distance,
    fraggle_similarity,
)
from .harris_lahey import (
    bulk_harris_lahey_binary_distance,
    bulk_harris_lahey_binary_similarity,
    harris_lahey_binary_distance,
    harris_lahey_binary_similarity,
)
from .kulczynski import (
    bulk_kulczynski_binary_distance,
    bulk_kulczynski_binary_similarity,
    kulczynski_binary_distance,
    kulczynski_binary_similarity,
)
from .mcconnaughey import (
    bulk_mcconnaughey_binary_distance,
    bulk_mcconnaughey_binary_similarity,
    mcconnaughey_binary_distance,
    mcconnaughey_binary_similarity,
)
from .mcs import (
    bulk_mcs_distance,
    bulk_mcs_similarity,
    mcs_distance,
    mcs_similarity,
)
from .rand import (
    bulk_rand_binary_distance,
    bulk_rand_binary_similarity,
    rand_binary_distance,
    rand_binary_similarity,
)
from .rogot_goldberg import (
    bulk_rogot_goldberg_binary_distance,
    bulk_rogot_goldberg_binary_similarity,
    rogot_goldberg_binary_distance,
    rogot_goldberg_binary_similarity,
)
from .russell import (
    bulk_russell_binary_distance,
    bulk_russell_binary_similarity,
    russell_binary_distance,
    russell_binary_similarity,
)
from .simpson import (
    bulk_simpson_binary_distance,
    bulk_simpson_binary_similarity,
    simpson_binary_distance,
    simpson_binary_similarity,
)
from .sokal_sneath import (
    bulk_sokal_sneath_2_binary_distance,
    bulk_sokal_sneath_2_binary_similarity,
    sokal_sneath_2_binary_distance,
    sokal_sneath_2_binary_similarity,
)
from .tanimoto import (
    bulk_tanimoto_binary_distance,
    bulk_tanimoto_binary_similarity,
    bulk_tanimoto_count_distance,
    bulk_tanimoto_count_similarity,
    tanimoto_binary_distance,
    tanimoto_binary_similarity,
    tanimoto_count_distance,
    tanimoto_count_similarity,
)

# exclude Fraggle and MCS, which operate on molecules
_functions = [
    (name, func)
    for name, func in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    if "fraggle" not in name and "mcs" not in name
]


_METRICS = {
    name: func
    for name, func in _functions
    if not name.startswith("bulk_") and name.endswith("_distance")
}
_METRIC_NAMES = set(_METRICS.keys())

_SIMILARITIES = {
    name: func
    for name, func in _functions
    if not name.startswith("bulk_") and name.endswith("_similarity")
}
_SIMILARITY_NAMES = set(_SIMILARITIES.keys())

_BULK_METRICS = {
    name: func
    for name, func in _functions
    if name.startswith("bulk_") and name.endswith("_distance")
}
_BULK_METRIC_NAMES = set(_BULK_METRICS.keys())

_BULK_SIMILARITIES = {
    name: func
    for name, func in _functions
    if name.startswith("bulk_") and name.endswith("_similarity")
}
_BULK_SIMILARITY_NAMES = set(_BULK_SIMILARITIES.keys())
