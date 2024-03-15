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
from rdkit import Chem

# from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP
from rdkit.Chem import AddHs, MolFromSmiles, MolToSmiles
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from scipy.sparse import csr_array, vstack

from skfp import *
from skfp import ERGFingerprint  # E3FP,
from skfp.helpers.map4_mhfp_helpers import get_map4_fingerprint

smiles_data = pd.read_csv("./hiv_mol.csv.zip", nrows=100)["smiles"]


@pytest.fixture
def smiles_molecules():
    return smiles_data


@pytest.fixture
def mol_object_molecules():
    return [Chem.MolFromSmiles(x) for x in smiles_data]


def test_ECFP_bit_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetMorganGenerator()
    ecfp = ECFP(n_jobs=-1, sparse=False, count=False)
    X_emf = ecfp.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_ECFP_sparse_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetMorganGenerator()
    ecfp = ECFP(n_jobs=-1, sparse=True, count=False)
    X_emf = ecfp.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_ECFP_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetMorganGenerator()
    ecfp = ECFP(n_jobs=-1, sparse=False, count=True)
    X_emf = ecfp.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_ECFP_sparse_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetMorganGenerator()
    ecfp = ECFP(n_jobs=-1, sparse=True, count=True)
    X_emf = ecfp.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_FCFP_bit_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    invgen = fpgens.GetMorganFeatureAtomInvGen()
    fp_gen = fpgens.GetMorganGenerator(atomInvariantsGenerator=invgen)
    ecfp = ECFP(use_fcfp=True, n_jobs=-1, sparse=False, count=False)
    X_emf = ecfp.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_FCFP_sparse_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    invgen = fpgens.GetMorganFeatureAtomInvGen()
    fp_gen = fpgens.GetMorganGenerator(atomInvariantsGenerator=invgen)
    ecfp = ECFP(use_fcfp=True, n_jobs=-1, sparse=True, count=False)
    X_emf = ecfp.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_FCFP_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    invgen = fpgens.GetMorganFeatureAtomInvGen()
    fp_gen = fpgens.GetMorganGenerator(atomInvariantsGenerator=invgen)
    ecfp = ECFP(use_fcfp=True, n_jobs=-1, sparse=False, count=True)
    X_emf = ecfp.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_FCFP_sparse_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    invgen = fpgens.GetMorganFeatureAtomInvGen()
    fp_gen = fpgens.GetMorganGenerator(atomInvariantsGenerator=invgen)
    ecfp = ECFP(use_fcfp=True, n_jobs=-1, sparse=True, count=True)
    X_emf = ecfp.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_atom_pair_bit_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=False, count=False)
    X_emf = atom_pair.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_atom_pair_sparse_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=True, count=False)
    X_emf = atom_pair.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_atom_pair_cound_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=False, count=True)
    X_emf = atom_pair.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_atom_pair_sparse_count_fingerprint(
    smiles_molecules, mol_object_molecules
):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=True, count=True)
    X_emf = atom_pair.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_topological_torsion_bit_fingerprint(
    smiles_molecules, mol_object_molecules
):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=False, count=False
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_topological_torsion_sparse_fingerprint(
    smiles_molecules, mol_object_molecules
):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=True, count=False
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_topological_torsion_count_fingerprint(
    smiles_molecules, mol_object_molecules
):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
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
    smiles_molecules, mol_object_molecules
):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
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


def test_maccs_keys_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    maccs_keys = MACCSKeysFingerprint(n_jobs=-1, sparse=False)
    X_emf = maccs_keys.transform(X)
    X_rdkit = np.array([GetMACCSKeysFingerprint(x) for x in X_for_rdkit])

    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_maccs_keys_sparse_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    maccs_keys = MACCSKeysFingerprint(n_jobs=-1, sparse=True)
    X_emf = maccs_keys.transform(X)
    X_rdkit = csr_array([GetMACCSKeysFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_erg_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    erg = ERGFingerprint(n_jobs=-1, sparse=False)
    X_emf = erg.transform(X)
    X_rdkit = np.array([GetErGFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_erg_sparse_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    erg = ERGFingerprint(n_jobs=-1, sparse=True)
    X_emf = erg.transform(X)
    X_rdkit = csr_array([GetErGFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_map4_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    map4 = MAP4Fingerprint(random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = np.stack([get_map4_fingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_map4_sparse_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    map4 = MAP4Fingerprint(sparse=True, random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = vstack([csr_array(get_map4_fingerprint(x)) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_map4_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    map4 = MAP4Fingerprint(count=True, random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = np.stack(
        [get_map4_fingerprint(x, count=True) for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_map4_sparse_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    map4 = MAP4Fingerprint(sparse=True, count=True, random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = vstack(
        [csr_array(get_map4_fingerprint(x, count=True)) for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_mhfp_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    mhfp = MHFP(random_state=0, n_jobs=-1)
    X_emf = mhfp.transform(X)
    encoder = MHFPEncoder(2048, 0)
    X_rdkit = np.array(
        MHFPEncoder.EncodeMolsBulk(
            encoder,
            X_for_rdkit,
        )
    )
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_mhfp_sparse_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    mhfp = MHFP(random_state=0, n_jobs=-1, sparse=True)
    X_emf = mhfp.transform(X)
    encoder = MHFPEncoder(2048, 0)
    X_rdkit = csr_array(
        MHFPEncoder.EncodeMolsBulk(
            encoder,
            X_for_rdkit,
        )
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_avalon_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_transformer = AvalonFingerprint(random_state=0, n_jobs=-1)
    X_emf = fp_transformer.transform(X)
    X_rdkit = np.stack([GetAvalonFP(x) for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_avalon_sparse_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_transformer = AvalonFingerprint(sparse=True, random_state=0, n_jobs=-1)
    X_emf = fp_transformer.transform(X)
    X_rdkit = vstack([csr_array(GetAvalonFP(x)) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_avalon_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_transformer = AvalonFingerprint(count=True, random_state=0, n_jobs=-1)
    X_emf = fp_transformer.transform(X)
    X_rdkit = np.stack([GetAvalonCountFP(x).ToList() for x in X_for_rdkit])
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_avalon_sparse_count_fingerprint(
    smiles_molecules, mol_object_molecules
):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_transformer = AvalonFingerprint(
        sparse=True, count=True, random_state=0, n_jobs=-1
    )
    X_emf = fp_transformer.transform(X)
    X_rdkit = vstack(
        [csr_array(GetAvalonCountFP(x).ToList()) for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_estate_sum_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    estate = EStateFingerprint(n_jobs=-1, sparse=False)
    X_emf = estate.transform(X)
    X_rdkit = np.array([FingerprintMol(x) for x in X_for_rdkit])[:, 1]
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_estate_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    estate = EStateFingerprint(n_jobs=-1, variant="binary", count=True)
    X_emf = estate.transform(X)
    X_rdkit = np.array([FingerprintMol(x) for x in X_for_rdkit])[:, 0]
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_estate_binary_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    estate = EStateFingerprint(n_jobs=-1, variant="binary")
    X_emf = estate.transform(X)
    X_rdkit = np.array([FingerprintMol(x) for x in X_for_rdkit])[:, 0] > 0
    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_estate_sparse_sum_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    estate = EStateFingerprint(n_jobs=-1, sparse=True)
    X_emf = estate.transform(X)
    X_rdkit = csr_array(
        np.array([FingerprintMol(x) for x in X_for_rdkit])[:, 1]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_estate_sparse_count_fingerprint(
    smiles_molecules, mol_object_molecules
):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    estate = EStateFingerprint(
        n_jobs=-1, sparse=True, variant="binary", count=True
    )
    X_emf = estate.transform(X)
    X_rdkit = csr_array(
        np.array([FingerprintMol(x) for x in X_for_rdkit])[:, 0]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_estate_sparse_binary_fingerprint(
    smiles_molecules, mol_object_molecules
):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    estate = EStateFingerprint(n_jobs=-1, sparse=True, variant="binary")
    X_emf = estate.transform(X)
    X_rdkit = csr_array(
        np.array([FingerprintMol(x) for x in X_for_rdkit])[:, 0] > 0
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


def test_pharmacophore_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules

    fp_transformer = PharmacophoreFingerprint(n_jobs=-1)
    X_emf = fp_transformer.transform(X)

    factory = Gobbi_Pharm2D.factory

    X_for_rdkit = [AddHs(x) for x in X_for_rdkit]
    X_rdkit = np.array(
        [Generate.Gen2DFingerprint(x, factory) for x in X_for_rdkit]
    )

    if not np.all(X_emf == X_rdkit):
        raise AssertionError


def test_pharmacophore_sparse_fingerprint(
    smiles_molecules, mol_object_molecules
):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules

    fp_transformer = PharmacophoreFingerprint(n_jobs=-1, sparse=True)
    X_emf = fp_transformer.transform(X)

    factory = Gobbi_Pharm2D.factory

    X_for_rdkit = [AddHs(x) for x in X_for_rdkit]
    X_rdkit = csr_array(
        [Generate.Gen2DFingerprint(x, factory) for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_rdkit.toarray()):
        raise AssertionError


# def test_e3fp(smiles_molecules, mol_object_molecules):
#     X = smiles_molecules
#     X_for_rdkit = mol_object_molecules
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
# def test_e3fp_sparse(smiles_molecules, mol_object_molecules):
#     X = smiles_molecules
#     X_for_rdkit = mol_object_molecules
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


def test_rdk_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetRDKitFPGenerator()
    rdk = RDKitFingerprint(
        random_state=0, n_jobs=-1, sparse=False, count=False
    )
    X_emf = rdk.transform(X)
    X_seq = np.array([fp_gen.GetFingerprint(x).ToList() for x in X_for_rdkit])
    if not np.all(X_emf == X_seq):
        raise AssertionError


def test_rdk_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetRDKitFPGenerator()
    rdk = RDKitFingerprint(random_state=0, n_jobs=-1, sparse=False, count=True)
    X_emf = rdk.transform(X)
    X_seq = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf == X_seq):
        raise AssertionError


def test_rdk_sparse_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetRDKitFPGenerator()
    rdk = RDKitFingerprint(random_state=0, n_jobs=-1, sparse=True, count=False)
    X_emf = rdk.transform(X)
    X_seq = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    if not np.all(X_emf.toarray() == X_seq.toarray()):
        raise AssertionError


def test_rdk_sparse_count_fingerprint(smiles_molecules, mol_object_molecules):
    X = smiles_molecules
    X_for_rdkit = mol_object_molecules
    fp_gen = fpgens.GetRDKitFPGenerator()
    rdk = RDKitFingerprint(random_state=0, n_jobs=-1, sparse=True, count=True)
    X_emf = rdk.transform(X)
    X_seq = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    if not np.all(X_emf.toarray() == X_seq.toarray()):
        raise AssertionError


def test_mol_to_smiles(smiles_molecules, mol_object_molecules):
    X_smiles = smiles_molecules

    transformer_to_mol = MolFromSmilesTransformer()
    X_mols = transformer_to_mol.transform(X_smiles)
    X_mols_rdkit = [MolFromSmiles(x) for x in X_smiles]

    transformer_to_smiles = MolToSmilesTransformer()
    X_new_smiles = transformer_to_smiles.transform(X_mols)
    X_new_smiles_rdkit = [MolToSmiles(x) for x in X_mols_rdkit]
    if not all(
        [x == smiles for x, smiles in zip(X_new_smiles, X_new_smiles_rdkit)]
    ):
        raise AssertionError


def test_input_validation(smiles_molecules):
    X = smiles_molecules

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
