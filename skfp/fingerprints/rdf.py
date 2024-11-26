from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import require_mols_with_conf_ids


class RDFFingerprint(BaseFingerprintTransformer):
    r"""
    RDF (Radial Distribution Function descriptors) fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, where
    features are based on the radial distribution function (RDF) of interatomic
    distances.

    RDF function can be interpreted as the probability distribution of finding
    an atom in a spherical volume of given radius ``r``, and is defined for all
    atoms i and j with distance :math:`r_{ij}` in the molecule with N atoms as:

    .. math::

        RDF(r) = \sum_{i}^{N-1} \sum_{j > i}^N w_i * w_j * e^{B (r - r_{ij})^2}

    This results in a Gaussian distribution, centered around each distance
    :math:`r_{ij}`, with width depending on the smoothing parameter B, which is set
    to 100 (similar to DRAGON software). Radii between 1 and 30 (inclusive) are used,
    corresponding to distances from 1Å to 16Å.

    7 weighting variants are used, unweighted and 6 based on atomic features:
    unweighted, atomic mass, van der Waals volume, electronegativity, polarizability,
    ion polarity, and IState [1]_ [2]_. They are relative to the carbon, e.g. molecular
    weight is: MW(atom_type) / MW(carbon).

    This results in 210 features. They are invariant to translation and rotation,
    independent of molecule size, and unique for a given conformation. See [3]_ [4]_
    [5]_ [6]_ for details.

    Parameters
    ----------
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
    n_features_out : int = 210
        Number of output features, size of fingerprints.

    requires_conformers : bool = True
        Value is always True, as this fingerprint is 3D based. It always requires
        molecules with conformers as inputs, with ``conf_id`` integer property set.

    See Also
    --------
    :class:`MORSEFingerprint` : Related fingerprint, which uses scattered electron
        intensity instead of radial distribution function.

    References
    ----------
    .. [1] RDKit IState discussion
        https://github.com/rdkit/rdkit/discussions/6122

    .. [2] RDKit IState implementation
        https://github.com/rdkit/rdkit/blob/df2b6c8a8c775748c1dcac83af0f92af434bab81/Code/GraphMol/Descriptors/MolData3Ddescriptors.cpp#L127

    .. [3] `DRAGON documentation
        "RDF descriptors"
        <http://chemgps.bmc.uu.se/help/dragonx/RDFdescriptors.html>`_

    .. [4] `Roberto Todeschini and Viviana Consonni
        "Molecular Descriptors for Chemoinformatics"
        <https://onlinelibrary.wiley.com/doi/book/10.1002/9783527628766>`_

    .. [5] `Guillaume Godin
        "3D	descriptors in RDKit"
        UGM 2017
        <https://github.com/rdkit/UGM_2017/blob/master/Presentations/Godin_3D_Descriptors.pdf>`_

    .. [6] `Hemmer, Markus C., Valentin Steinhauer, and Johann Gasteiger
        "Deriving the 3D structure of organic molecules from their infrared spectra"
        Vibrational spectroscopy 19.1 (1999): 151-164
        <https://www.sciencedirect.com/science/article/pii/S0924203199000144>`_

    Examples
    --------
    >>> from skfp.fingerprints import RDFFingerprint
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = RDFFingerprint()
    >>> fp
    RDFFingerprint()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen = ConformerGenerator()
    >>> mols = conf_gen.transform(mols)
    >>> fp.transform(mols)  # doctest: +SKIP
    array([[1.930e+00, 2.070e-01, 0.000e+00, ..., 0.000e+00, 0.000e+00, 0.000e+00],
           [1.790e+00, 9.990e-01, 4.160e-01, ..., 0.000e+00, 0.000e+00, 0.000e+00],
           [1.150e-01, 0.000e+00, 0.000e+00, ..., 0.000e+00, 0.000e+00, 0.000e+00],
           [1.427e+00, 9.920e-01, 1.443e+00, ..., 0.000e+00, 0.000e+00, 0.000e+00]])
    """

    def __init__(
        self,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_features_out=210,
            requires_conformers=True,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import CalcRDF

        X = require_mols_with_conf_ids(X)
        X = [CalcRDF(mol, confId=mol.GetIntProp("conf_id")) for mol in X]
        return csr_array(X) if self.sparse else np.array(X)
