import json
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseSubstructureFingerprint
from skfp.utils import ensure_mols

_TREE_PATH = Path(__file__).parent / "data" / "tree.json"


class _PatternNode:
    """
    Node in the SMARTS pattern tree.

    Attributes
    ----------
    smarts : str = None
        SMARTS string defining the pattern.

    pattern_mol : Mol = None
        RDKit Mol object of the pattern.

    is_terminal : bool = False
        Whether this node corresponds to a complete pattern or just a prefix.

    feature_bit : int = None
        Index of the corresponding fingerprint bit.

    children : list[_PatternNode] = []
        Child nodes.

    atom_requirements : defaultdict[str, int]
        Minimal atom requirements needed to match at this node.
    """

    __slots__ = (
        "atom_requirements",
        "children",
        "feature_bit",
        "is_terminal",
        "pattern_mol",
        "smarts",
    )

    def __init__(self):
        self.smarts: str | None = None
        self.pattern_mol: Mol | None = None
        self.is_terminal: bool = False
        self.feature_bit: int | None = None
        self.atom_requirements: defaultdict[str, int] = defaultdict(int)
        self.children: list[_PatternNode] = []


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
        self._feature_names: list[str] = []
        self._pattern_atoms: dict[str, Mol] = {}
        self._root: _PatternNode = _PatternNode()
        self._load_tree()
        super().__init__(
            patterns=self._feature_names,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
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

    def _dict_to_node(self, d: dict) -> _PatternNode:
        """
        Recursively convert a dict representation of a pattern tree
        into a _PatternNode tree.
        """
        node = _PatternNode()
        node.smarts = d.get("smarts")
        node.pattern_mol = Chem.MolFromSmarts(node.smarts) if node.smarts else None
        node.is_terminal = d.get("is_terminal", False)
        node.feature_bit = d.get("feature_bit")
        node.atom_requirements = defaultdict(int, d.get("atom_requirements", {}))
        node.children = [
            self._dict_to_node(node_dict) for node_dict in d.get("children", [])
        ]

        if (
            node.is_terminal
            and node.smarts is not None
            and node.feature_bit is not None
        ):
            self._feature_names[int(node.feature_bit)] = node.smarts
        return node

    def _load_tree(self) -> None:
        """
        Load the pattern tree from a JSON file into internal representation.
        """
        file = _TREE_PATH
        if not file.exists():
            raise FileNotFoundError(f"Tree file not found: {file}")

        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for key in ("n_nodes", "atoms", "tree"):
            if key not in data:
                raise KeyError(f"Missing key {key} in tree file {file}")

        self._feature_names = [""] * data["n_terminal_nodes"]
        self._pattern_atoms = {key: Chem.MolFromSmarts(key) for key in data["atoms"]}
        self._root = self._dict_to_node(data["tree"])

        if any(f == "" for f in self._feature_names):
            raise ValueError("SMARTS or feature_bit missing in terminal nodes")

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
            stack: deque[_PatternNode] = deque(root_children)
            atom_contents = defaultdict(int)
            for key, atom in self._pattern_atoms.items():
                atom_contents[key] = len(mol.GetSubstructMatches(atom))
            while stack:
                node = stack.pop()

                for key, val in node.atom_requirements.items():
                    if atom_contents[key] < val:
                        break
                else:
                    if not mol.HasSubstructMatch(node.pattern_mol):
                        continue

                    if node.is_terminal:
                        bits[i][node.feature_bit] = set_value(mol, node.pattern_mol)

                    stack.extend(node.children)

        return csr_array(bits) if self.sparse else bits
