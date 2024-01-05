import numpy as np
import pandas as pd
import pytest
import rdkit.Chem.rdFingerprintGenerator as fpgens

# from e3fp.conformer.generate import (
#     FORCEFIELD_DEF,
#     MAX_ENERGY_DIFF_DEF,
#     POOL_MULTIPLIER_DEF,
#     RMSD_CUTOFF_DEF,
# )
# from e3fp.conformer.generator import ConformerGenerator
# from e3fp.pipeline import fprints_from_mol
from ogb.graphproppred import GraphPropPredDataset
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from scipy.sparse import csr_array, vstack

from skfp import (
    E3FP,
    ECFP,
    MHFP,
    AtomPairFingerprint,
    ERGFingerprint,
    MACCSKeysFingerprint,
    MAP4Fingerprint,
    RDKitFingerprint,
    TopologicalTorsionFingerprint,
)
from skfp.helpers.map4_mhfp_helpers import get_map4_fingerprint, get_mhfp

dataset_name = "ogbg-molhiv"
GraphPropPredDataset(name=dataset_name, root="../dataset")
dataset = pd.read_csv(
    f"../dataset/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
)
smiles_data = dataset["smiles"]


@pytest.fixture
def example_molecules():
    return smiles_data


@pytest.fixture
def rdkit_example_molecules():
    return [Chem.MolFromSmiles(x) for x in smiles_data]


def test_ECFP_bit_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    ecfp = ECFP(n_jobs=-1, sparse=False, count=False)
    X_emf = ecfp.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_ECFP_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    ecfp = ECFP(n_jobs=-1, sparse=True, count=False)
    X_emf = ecfp.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_ECFP_count_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    ecfp = ECFP(n_jobs=-1, sparse=False, count=True)
    X_emf = ecfp.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_ECFP_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    ecfp = ECFP(n_jobs=-1, sparse=True, count=True)
    X_emf = ecfp.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_FCFP_bit_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    invgen = fpgens.GetMorganFeatureAtomInvGen()
    fp_gen = fpgens.GetMorganGenerator(atomInvariantsGenerator=invgen)
    ecfp = ECFP(use_fcfp=True, n_jobs=-1, sparse=False, count=False)
    X_emf = ecfp.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_FCFP_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    invgen = fpgens.GetMorganFeatureAtomInvGen()
    fp_gen = fpgens.GetMorganGenerator(atomInvariantsGenerator=invgen)
    ecfp = ECFP(use_fcfp=True, n_jobs=-1, sparse=True, count=False)
    X_emf = ecfp.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_FCFP_count_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    invgen = fpgens.GetMorganFeatureAtomInvGen()
    fp_gen = fpgens.GetMorganGenerator(atomInvariantsGenerator=invgen)
    ecfp = ECFP(use_fcfp=True, n_jobs=-1, sparse=False, count=True)
    X_emf = ecfp.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_FCFP_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    invgen = fpgens.GetMorganFeatureAtomInvGen()
    fp_gen = fpgens.GetMorganGenerator(atomInvariantsGenerator=invgen)
    ecfp = ECFP(use_fcfp=True, n_jobs=-1, sparse=True, count=True)
    X_emf = ecfp.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_atom_pair_bit_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=False, count=False)
    X_emf = atom_pair.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_atom_pair_sparse_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=True, count=False)
    X_emf = atom_pair.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_atom_pair_cound_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=False, count=True)
    X_emf = atom_pair.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_atom_pair_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=True, count=True)
    X_emf = atom_pair.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_topological_torsion_bit_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=False, count=False
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_topological_torsion_sparse_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=True, count=False
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_topological_torsion_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=False, count=True
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_topological_torsion_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=True, count=True
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_maccs_keys_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = MACCSKeysFingerprint(n_jobs=-1, sparse=False)
    X_emf = erg.transform(X)
    X_rdkit = np.array([GetMACCSKeysFingerprint(x) for x in X_for_rdkit])

    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_maccs_keys_sparse_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = MACCSKeysFingerprint(n_jobs=-1, sparse=True)
    X_emf = erg.transform(X)
    X_rdkit = csr_array([GetMACCSKeysFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_erg_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = ERGFingerprint(n_jobs=-1, sparse=False)
    X_emf = erg.transform(X)
    X_rdkit = np.array([GetErGFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_erg_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = ERGFingerprint(n_jobs=-1, sparse=True)
    X_emf = erg.transform(X)
    X_rdkit = csr_array([GetErGFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_map4_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    map4 = MAP4Fingerprint(random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = np.stack([get_map4_fingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_map4_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    map4 = MAP4Fingerprint(sparse=True, random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = vstack([csr_array(get_map4_fingerprint(x)) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_map4_count_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    map4 = MAP4Fingerprint(count=True, random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = np.stack(
        [get_map4_fingerprint(x, count=True) for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_map4_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    map4 = MAP4Fingerprint(sparse=True, count=True, random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = vstack(
        [csr_array(get_map4_fingerprint(x, count=True)) for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_mhfp_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    mhfp = MHFP(random_state=0, n_jobs=-1)
    X_emf = mhfp.transform(X)
    X_rdkit = np.array([get_mhfp(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_mhfp_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    mhfp = MHFP(random_state=0, n_jobs=-1, sparse=True)
    X_emf = mhfp.transform(X)
    X_rdkit = csr_array([get_mhfp(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_mhfp_count_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    mhfp = MHFP(random_state=0, n_jobs=-1, count=True)
    X_emf = mhfp.transform(X)
    X_rdkit = np.array([get_mhfp(x, count=True) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_mhfp_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    mhfp = MHFP(random_state=0, n_jobs=-1, sparse=True, count=True)
    X_emf = mhfp.transform(X)
    X_rdkit = csr_array([get_mhfp(x, count=True) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


# def test_e3fp(example_molecules):
#     X = example_molecules
#
#     e3fp_fp = E3FP(
#         4096,
#         1.5,
#         is_folded=True,
#         n_jobs=-1,
#         verbose=0,
#         sparse=False,
#     )
#     X_emf = e3fp_fp.transform(X)
#
#     confgen_params = {
#         "first": 1,
#         "num_conf": 3,
#         "pool_multiplier": POOL_MULTIPLIER_DEF,
#         "rmsd_cutoff": RMSD_CUTOFF_DEF,
#         "max_energy_diff": MAX_ENERGY_DIFF_DEF,
#         "forcefield": FORCEFIELD_DEF,
#         "get_values": True,
#         "seed": 0,
#     }
#     fprint_params = {
#         "bits": 4096,
#         "radius_multiplier": 1.5,
#         "rdkit_invariants": True,
#     }
#
#     conf_gen = ConformerGenerator(**confgen_params)
#
#     def e3fp_func(smiles):
#         # creating molecule object
#         mol = Chem.MolFromSmiles(smiles)
#         mol.SetProp("_Name", smiles)
#         mol = PropertyMol(mol)
#         mol.SetProp("_SMILES", smiles)
#
#         try:
#             # getting a molecule and the fingerprint
#             mol, values = conf_gen.generate_conformers(mol)
#             fps = fprints_from_mol(mol, fprint_params=fprint_params)
#
#             # chose the fingerprint with the lowest energy
#             energies = values[2]
#             fp = fps[np.argmin(energies)].fold(1024)
#
#             return fp.to_vector()
#         except RuntimeError:
#             return np.full(shape=1024, fill_value=-1)
#
#     X_seq = [e3fp_func(smiles) for smiles in X]
#
#     X_seq = np.array([fp.toarray().squeeze() for fp in X_seq])
#     if not np.all(X_emf == X_seq):
#         raise AssertionError
#
#
# def test_e3fp_sparse(example_molecules):
#     X = example_molecules
#
#     e3fp_fp = E3FP(
#         4096,
#         1.5,
#         is_folded=True,
#         n_jobs=-1,
#         verbose=0,
#         sparse=True,
#     )
#     X_emf = e3fp_fp.transform(X)
#
#     confgen_params = {
#         "first": 1,
#         "num_conf": 3,
#         "pool_multiplier": POOL_MULTIPLIER_DEF,
#         "rmsd_cutoff": RMSD_CUTOFF_DEF,
#         "max_energy_diff": MAX_ENERGY_DIFF_DEF,
#         "forcefield": FORCEFIELD_DEF,
#         "get_values": True,
#         "seed": 0,
#     }
#     fprint_params = {
#         "bits": 4096,
#         "radius_multiplier": 1.5,
#         "rdkit_invariants": True,
#     }
#
#     conf_gen = ConformerGenerator(**confgen_params)
#
#     def e3fp_func(smiles):
#         # creating molecule object
#         mol = Chem.MolFromSmiles(smiles)
#         mol.SetProp("_Name", smiles)
#         mol = PropertyMol(mol)
#         mol.SetProp("_SMILES", smiles)
#
#         try:
#             # getting a molecule and the fingerprint
#             mol, values = conf_gen.generate_conformers(mol)
#             fps = fprints_from_mol(mol, fprint_params=fprint_params)
#
#             # chose the fingerprint with the lowest energy
#             energies = values[2]
#             fp = fps[np.argmin(energies)].fold(1024)
#
#             return fp.to_vector()
#         except RuntimeError:
#             return np.full(shape=1024, fill_value=-1)
#
#     X_seq = [e3fp_func(smiles) for smiles in X]
#
#     X_seq = vstack(X_seq)
#
#     if not np.all(X_emf.toarray() == X_seq.toarray()):
#         raise AssertionError


def test_rdk_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetRDKitFPGenerator()
    rdk = RDKitFingerprint(
        random_state=0, n_jobs=-1, sparse=False, count=False
    )
    X_emf = rdk.transform(X)
    X_seq = np.array([fp_gen.GetFingerprint(x).ToList() for x in X_for_rdkit])
    if not np.all(X_emf == X_seq):
        raise AssertionError


def test_rdk_count_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetRDKitFPGenerator()
    rdk = RDKitFingerprint(random_state=0, n_jobs=-1, sparse=False, count=True)
    X_emf = rdk.transform(X)
    X_seq = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_seq):
        raise AssertionError


def test_rdk_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetRDKitFPGenerator()
    rdk = RDKitFingerprint(random_state=0, n_jobs=-1, sparse=True, count=False)
    X_emf = rdk.transform(X)
    X_seq = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_seq.toarray()):
        raise AssertionError


def test_rdk_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetRDKitFPGenerator()
    rdk = RDKitFingerprint(random_state=0, n_jobs=-1, sparse=True, count=True)
    X_emf = rdk.transform(X)
    X_seq = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_seq.toarray()):
        raise AssertionError


def test_input_validation(example_molecules):
    X = example_molecules

    X_test = X.copy()

    rdkit_generator = fpgens.GetMorganGenerator()
    ecfp = ECFP(n_jobs=-1)
    fp_function = rdkit_generator.GetFingerprint

    X_test = np.array([Chem.MolFromSmiles(x) for x in X_test])
    X_seq = np.array([fp_function(x) for x in X_test])

    # 1) Some of the molecules are still given as SMILES
    X = [Chem.MolFromSmiles(x) if i % 2 == 0 else x for i, x in enumerate(X)]
    X_ecfp = ecfp.transform(X)
    if not np.all(X_ecfp == X_seq):
        raise AssertionError

    # 2) There exists an element in the list, which is not a molecule
    X_2 = np.copy(X)
    X_2[1] = 1

    with pytest.raises(ValueError) as exec_info:
        X_ecfp_2 = ecfp.transform(X_2)

    if not (
        str(exec_info.value)
        == "Passed value is neither rdkit.Chem.rdChem.Mol nor SMILES"
    ):
        raise AssertionError
