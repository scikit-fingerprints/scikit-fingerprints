from rdkit.Chem.rdMolDescriptors import (
    GetMorganFingerprint,
    GetMorganFingerprintAsBitVect,
    GetHashedMorganFingerprint,
    GetMACCSKeysFingerprint,
    GetAtomPairFingerprint,
    GetHashedAtomPairFingerprint,
    GetHashedAtomPairFingerprintAsBitVect,
    GetTopologicalTorsionFingerprint,
    GetHashedTopologicalTorsionFingerprint,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
    GetFeatureInvariants,
    GetConnectivityInvariants,
)
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

fp_descriptors = {
    "morgan_default": GetMorganFingerprint,
    "morgan_hashed": GetHashedMorganFingerprint,
    "morgan_as_bit_vect": GetMorganFingerprintAsBitVect,
    "maccs_keys": GetMACCSKeysFingerprint,
    "atom_pair_default": GetAtomPairFingerprint,
    "atom_pair_hashed": GetHashedAtomPairFingerprint,
    "atom_pair_as_bit_vect": GetHashedAtomPairFingerprintAsBitVect,
    "topological_torsion_default": GetTopologicalTorsionFingerprint,
    "topological_torsion_hashed": GetHashedTopologicalTorsionFingerprint,
    "topological_torsion_as_bit_vect": GetHashedTopologicalTorsionFingerprintAsBitVect,
    "erg": GetErGFingerprint,
}
