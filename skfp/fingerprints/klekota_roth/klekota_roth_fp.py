from collections import deque
from collections.abc import Sequence

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseSubstructureFingerprint
from skfp.utils import ensure_mols

from .smarts_tree import PatternNode, _load_tree


class KlekotaRothFingerprint(BaseSubstructureFingerprint):
    """
    Klekota-Roth fingerprint.

    A substructure fingerprint based on [1]_, with implementation based on CDK [2]_.
    Tests for presence of 4860 predefined substructures which are predisposed for
    bioactivity.

    Parameters
    ----------
    count : bool, default=False
        Whether to return binary (bit) features, or their counts.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int = 4860
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] `Klekota, Justin, and Frederick P Roth.
        “Chemical substructures that enrich for biological activity.”
        Bioinformatics (Oxford, England) vol. 24,21 (2008): 2518-25.
        <https://pubmed.ncbi.nlm.nih.gov/18784118/>`_

    .. [2] `Chemistry Development Kit (CDK) KlekotaRothFingerprinter
        <https://cdk.github.io/cdk/latest/docs/api/org/openscience/cdk/fingerprint/KlekotaRothFingerprinter.html>`_

    Examples
    --------
    >>> from skfp.fingerprints import KlekotaRothFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = KlekotaRothFingerprint()
    >>> fp
    KlekotaRothFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], shape=(4, 4860), dtype=uint8)
    """

    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        # note that those patterns were released as public domain:
        # https://github.com/cdk/cdk/blob/main/descriptor/fingerprint/src/main/java/org/openscience/cdk/fingerprint/KlekotaRothFingerprinter.java
        self._feature_names: list[str]
        self._pattern_atoms: dict[str, Mol]
        self._root: PatternNode

        self._root, self._feature_names, self._pattern_atoms = _load_tree()
        super().__init__(
            patterns=self._feature_names,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Get fingerprint output feature names. They are raw SMARTS patterns
        used as feature definitions.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Klekota-Roth feature names.
        """
        return np.asarray(self._feature_names, dtype=object)

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute Klekota-Roth fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 4860)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        X = ensure_mols(X)

        n_bits = self.n_features_out
        bits = np.zeros((len(X), n_bits), dtype=np.uint32 if self.count else np.uint8)
        root_children = self._root.children

        if self.count:
            set_value = lambda mol, pattern: len(mol.GetSubstructMatches(pattern))
        else:
            set_value = lambda _mol, _pattern: 1

        for i, mol in enumerate(X):
            stack: deque[PatternNode] = deque(root_children)
            atom_contents = self._count_atom_patterns(mol)
            while stack:
                node = stack.pop()

                if any(
                    atom_contents[key] < val
                    for key, val in node.atom_requirements.items()
                ):
                    continue

                if not mol.HasSubstructMatch(node.pattern_mol):
                    continue

                if node.is_terminal:
                    bits[i][node.feature_bit] = set_value(mol, node.pattern_mol)

                stack.extend(node.children)

        return csr_array(bits) if self.sparse else bits

    def _count_atom_patterns(self, mol: Mol) -> dict[str, int]:
        """
        Count occurrences of atom-level patterns in a molecule.
        """
        atom_contents = dict.fromkeys(self._pattern_atoms, 0)
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atomic_num = atom.GetAtomicNum()
            hcount = atom.GetTotalNumHs()
            charge = atom.GetFormalCharge()
            aromatic = atom.GetIsAromatic()

            symbol = symbol.lower() if aromatic else symbol

            # plain element symbol
            if symbol in atom_contents:
                atom_contents[symbol] += 1

            # atomic number pattern
            key = f"[#{atomic_num}]"
            if key in atom_contents:
                atom_contents[key] += 1

            # hydrogen count pattern
            key = f"[{symbol}&H{hcount}]"
            if key in atom_contents:
                atom_contents[key] += 1

            # charge pattern
            if charge != 0:
                sign = "+" if charge > 0 else "-"
                key = f"[{symbol}&{sign}]"
                if key in atom_contents:
                    atom_contents[key] += 1

            # negation of hydrogen
            if atomic_num != 1:
                atom_contents["[!#1]"] += 1

        return atom_contents
