from collections.abc import Sequence
from numbers import Real
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import require_mols_with_conf_ids


class GETAWAYFingerprint(BaseFingerprintTransformer):
    r"""
    GETAWAY (GEometry, Topology, and Atom-Weights AssemblY) fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, where
    bits are features related to various autocorrelations and information measures
    defined on Molecular Influence Matrix (MIM).

    MIM matrix H is based on the centered atom coordinates (x,y,z) of a given
    conformer, and is therefore rotation invariant, and all features are independent
    of the conformer alignment. Diagonal elements of MIM matrix, called leverages,
    measure "influence" of each atom in determining the whole shape of the molecule.
    The influence matrix R, measuring strength of interatomic interactions, is then
    defined based on atom leverages and their spatial distances.

    GETAWAY descriptors consist of 273 features (see [3]_ [4]_ [5]_ [6]_ for precise
    definitions):

    - 7 related to general molecule shape, defined only on H and R matrices
    - 7 sets of autocorrelation descriptors, each defined on topological distances
      (shortest paths) from 0 to 8 (inclusive)

    Autocorrelation descriptors are unweighted, or weighted by: atomic mass, van der
    Waals volume, electronegativity, polarizability, ion polarity, and IState [1]_ [2]_.
    Those weights are relative to the carbon, e.g. molecular weight is: MW(atom_type) / MW(carbon).

    Typical correct values should be small, but it often results in NaN or infinity for
    some descriptors. Value clipping with ``clip_val`` parameter, feature selection, and/or
    imputation should be used.

    Parameters
    ----------
    clip_val : float or None, default=2147483647
        Value to clip results at, both positive and negative ones.The default value is
        the maximal value of 32-bit integer, but should often be set lower, depending
        on the application. ``None`` means that no clipping is applied.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int = 273
        Number of output features, size of fingerprints.

    requires_conformers : bool = True
        Value is always True, as this fingerprint is 3D based. It always requires
        molecules with conformers as inputs, with ``conf_id`` integer property set.

    References
    ----------
    .. [1] RDKit IState discussion
        https://github.com/rdkit/rdkit/discussions/6122

    .. [2] RDKit IState implementation
        https://github.com/rdkit/rdkit/blob/df2b6c8a8c775748c1dcac83af0f92af434bab81/Code/GraphMol/Descriptors/MolData3Ddescriptors.cpp#L127

    .. [3] `Rajarshi Guha
        "GETAWAY"
        <http://www.rguha.net/writing/notes/desc/node5.html>`_

    .. [4] `Roberto Todeschini and Viviana Consonni
        "Molecular Descriptors for Chemoinformatics"
        <https://onlinelibrary.wiley.com/doi/book/10.1002/9783527628766>`_

    .. [5] `Guillaume Godin
        "3D	descriptors in RDKit"
        UGM 2017
        <https://github.com/rdkit/UGM_2017/blob/master/Presentations/Godin_3D_Descriptors.pdf>`_

    .. [6] `Viviana Consonni, Roberto Todeschini, and Manuela Pavan
        "Structure/Response Correlations and Similarity/Diversity Analysis by GETAWAY Descriptors.
        1. Theory of the Novel 3D Molecular Descriptors"
        J. Chem. Inf. Comput. Sci. 2002, 42, 3, 682–692
        <https://pubs.acs.org/doi/abs/10.1021/ci015504a>`_

    Examples
    --------
    >>> from skfp.fingerprints import GETAWAYFingerprint
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = GETAWAYFingerprint()
    >>> fp
    GETAWAYFingerprint()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen = ConformerGenerator()
    >>> mols = conf_gen.transform(mols)
    >>> fp.transform(mols)  # doctest: +SKIP
    array([[ 0.   ,    nan,  1.585, ...,  1.   , -0.   ,  1.   ],
           [ 0.   ,  0.   ,  2.763, ...,  1.   ,  0.   ,  1.   ],
           [ 0.   ,  0.   ,  1.   , ...,  1.   ,  0.   , 13.076],
           [ 4.755,  1.   ,  2.502, ..., -1.   , -0.   ,  2.467]])
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "clip_val": [None, Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        clip_val: float = 2147483647,  # max int32 value
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_features_out=273,
            requires_conformers=True,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.clip_val = clip_val

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcGETAWAY

        X = require_mols_with_conf_ids(X)
        X = [CalcGETAWAY(mol, confId=mol.GetIntProp("conf_id")) for mol in X]
        if self.clip_val is not None:
            X = np.clip(X, -self.clip_val, self.clip_val)

        return csr_array(X) if self.sparse else np.array(X)
