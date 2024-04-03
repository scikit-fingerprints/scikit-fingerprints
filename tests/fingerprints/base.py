import re

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import (
    AtomPairFingerprint,
    MACCSFingerprint,
    PhysiochemicalPropertiesFingerprint,
)

"""
We cannot test most of FingerprintTransformer directly, as it is an abstract base
class (ABC), but its methods are used extensively by inheriting classes. Therefore,
we use inheriting fingerprints as proxies.
"""


def test_base_is_always_fitted(smiles_list):
    atom_pair_fp = AtomPairFingerprint(n_jobs=-1)
    assert atom_pair_fp.__sklearn_is_fitted__()


def test_base_transform_copy(smiles_list):
    atom_pair_fp = AtomPairFingerprint(n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list, copy=False)
    X_skfp_2 = atom_pair_fp.transform(smiles_list, copy=True)
    assert np.array_equal(X_skfp, X_skfp_2)


def test_base_verbose_progress(smiles_list, capsys):
    atom_pair_fp = AtomPairFingerprint(n_jobs=-1, verbose=10)
    _ = atom_pair_fp.transform(smiles_list)
    stderr = capsys.readouterr().err  # tqdm outputs to stderr

    # example output: 17%|█▋        | 2/12 [00:00<00:00, 458.09it/s]

    # percentage, e.g. 10%
    assert re.search(r"\d+%", stderr)

    # processed iterations, e.g. 1/10
    assert re.search(r"\d+/\d+", stderr)

    # time, e.g. 00:01
    assert re.search(r"\d\d:\d\d", stderr)

    # iterations per second, e.g. 1.23it/s
    assert re.search(r"\d+\.\d+it/s", stderr)


def test_base_invalid_params(smiles_list):
    maccs_fp = MACCSFingerprint(sparse=None)
    with pytest.raises(InvalidParameterError):
        maccs_fp.transform(smiles_list)


def test_base_hash_fingerprint_bits():
    pp_fp = PhysiochemicalPropertiesFingerprint()
    X = [1, 2, 3, 4]
    with pytest.raises(ValueError) as exc_info:
        pp_fp._hash_fingerprint_bits(X, fp_size=1, count=False, sparse=False)

    assert "Fingerprint hashing requires instances of one of" in str(exc_info)
