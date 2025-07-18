import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Mol

_TREE_PATH = Path(__file__).parent / "tree_data.json"


@dataclass(slots=True, frozen=True)
class PatternNode:
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

    smarts: str | None = None
    pattern_mol: Mol | None = None
    is_terminal: bool = False
    feature_bit: int | None = None
    atom_requirements: dict[str, int] = field(default_factory=dict)
    children: list["PatternNode"] = field(default_factory=list)


def _dict_to_node(d: dict, feature_names: list[str]) -> PatternNode:
    """
    Recursively convert a dict representation of a pattern tree
    into a _PatternNode tree.
    """
    node = PatternNode(
        smarts=d.get("smarts"),
        pattern_mol=Chem.MolFromSmarts(d.get("smarts") or ""),
        is_terminal=d.get("is_terminal", False),
        feature_bit=d.get("feature_bit"),
        atom_requirements=d.get("atom_requirements", {}),
        children=[
            _dict_to_node(node_dict, feature_names)
            for node_dict in d.get("children", [])
        ],
    )

    if node.is_terminal:
        # node.smarts and node.feature_bit can be None only in non-terminal nodes.
        # These values are set when generating the tree.
        feature_names[int(node.feature_bit)] = node.smarts  # type: ignore

    return node


@lru_cache(maxsize=1)
def _load_tree() -> tuple[PatternNode, list[str], dict[str, Mol]]:
    """
    Load the pattern tree from a JSON file into internal representation.
    """
    path = _TREE_PATH
    if not path.exists():
        raise FileNotFoundError("Klekota-Roth SMARTS tree file not found")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    feature_names = [""] * data["n_terminal_nodes"]
    pattern_atoms = {key: Chem.MolFromSmarts(key) for key in data["atoms"]}
    root = _dict_to_node(data["tree"], feature_names)
    return root, feature_names, pattern_atoms
