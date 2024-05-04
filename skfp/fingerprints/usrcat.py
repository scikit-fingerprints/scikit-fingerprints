from collections.abc import Sequence
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import require_mols_with_conf_ids


class USRCATFingerprint(BaseFingerprintTransformer):
    """
    USRCAT (Ultrafast Shape Recognition with CREDO Atom Types) fingerprint.

    The implementation uses RDKit. This fingerprint extends the USR fingerprint by
    additionally considering pharmacophoric atom types [1]_. Firstly, the four
    reference points are computed, and they are used in all five feature groups. Each
    group is processed exactly the same, i.e. distance distributions to reference
    points are calculated, and features from the first three moments (mean, standard
    deviation, cubic root of skewness) are computed. This results in 12 features.

    USRCAT expands on USR by considering 4 additional subsets of atoms, based on their
    pharmacophoric types: hydrophobic, aromatic, hydrogen bond donor or acceptor atoms.
    For each atoms subset, distance distribution and moments are calculated like in
    the original USR. This results in 5 * 12 = 60 features.

    This is a 3D fingerprint, and requries molecules with ``conf_id`` integer property
    set. They can be generated with :class:`~skfp.preprocessing.ConformerGenerator`.
    Furthermore, only molecules with 3 or more atoms are allowed, to allow computation
    of all three moments.

    Parameters
    ----------
    errors : {"raise", "NaN", "ignore"}, default="raise"
        How to handle errors during fingerprint calculation. ``"raise"`` immediately
        raises any errors. ``"NaN"`` returns NaN values for molecules which resulted in
        errors. ``"ignore"`` suppresses errors and does not return anything for
        molecules with errors. This potentially results in less output vectors than
        input molecules, and should be used with caution.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when computing fingerprints.

    Attributes
    ----------
    n_features_out : int = 60
        Number of output features, size of fingerprints.

    requires_conformers : bool = True
        Value is always True, as this fingerprint is 3D based. It always requires
        molecules with conformers as inputs, with ``conf_id`` integer property set.

    See Also
    --------
    :class:`USR` : Related fingerprint, which USRCAT expands.

    References
    ----------
    .. [1] `Adrian M. Schreyer and Tom Blundell
        "USRCAT: real-time ultrafast shape recognition with pharmacophoric constraints"
        J Cheminform 4, 27 (2012)
        <https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-4-27#citeas>`_

    Examples
    --------
    >>> from skfp.fingerprints import USRCATFingerprint
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["CC=O"]
    >>> fp = USRCATFingerprint()
    >>> fp
    USRCATFingerprint()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen = ConformerGenerator()
    >>> mols = conf_gen.transform(mols)
    >>> fp.transform(mols)
    array([[ 1.33723405,  0.39526642, -0.90192794,  1.34623889,  0.73604224,
        -0.65044688,  2.00626805,  1.0036678 , -0.95792307,  1.78586959,
         0.94315243, -0.78539494,  0.64006787,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  2.31078831,  0.        ,
         0.        ,  1.10534475,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  1.75727917,  0.        ,  0.        ,  2.31078831,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         3.19576752,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "errors": [StrOptions({"raise", "NaN", "ignore"})],
    }

    def __init__(
        self,
        errors: str = "raise",
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=60,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.errors = errors

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute USRCAT fingerprints. If ``errors`` is set to ``"ignore"``, then in
        case of errors less than n_samples values may be returned.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit Mol objects, with conformers generated and
            ``conf_id`` integer property set.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.
        """
        y = np.empty(len(X))
        X, _ = self.transform_x_y(X, y, copy=copy)
        return X

    def transform_x_y(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[Union[np.ndarray, csr_array], np.ndarray]:
        """
        Compute USRCAT fingerprints. If ``errors`` is set to ``"ignore"``, then in
        case of errors less than n_samples values may be returned. The returned
        values for X and y are properly synchronized.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit Mol objects, with conformers generated and
            ``conf_id`` integer property set.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.

        copy : bool, default=False
            Copy the inputs X and y or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.
        """
        if copy:
            X = deepcopy(X)
            y = deepcopy(y)

        X = super().transform(X)

        if self.errors == "ignore":
            # errors are marked as NaN rows
            idxs_to_keep = [
                idx for idx, x in enumerate(X) if not np.any(np.isnan(x.data))
            ]
            X = X[idxs_to_keep]
            y = y[idxs_to_keep]

        return X, y

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import GetUSRCAT

        X = require_mols_with_conf_ids(X)

        if self.errors == "raise":
            fps = [GetUSRCAT(mol, confId=mol.GetIntProp("conf_id")) for mol in X]
        else:  # self.errors in {"NaN", "ignore"}
            fps = []
            for mol in X:
                try:
                    fp = GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
                except ValueError:
                    fp = np.full(self.n_features_out, np.NaN)
                fps.append(fp)

        return np.array(fps)
