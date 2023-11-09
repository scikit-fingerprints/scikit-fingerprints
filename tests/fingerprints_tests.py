import numpy as np
import pytest
import rdkit.Chem.rdFingerprintGenerator as fpgens
from e3fp.pipeline import fprints_from_smiles
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from scipy.sparse import csr_array

from e3fp.conformer.generate import (
    NUM_CONF_DEF,
    POOL_MULTIPLIER_DEF,
    RMSD_CUTOFF_DEF,
    MAX_ENERGY_DIFF_DEF,
    FORCEFIELD_DEF,
)
from e3fp.fingerprint.metrics import tanimoto

from featurizers.fingerprints import (
    AtomPairFingerprint,
    ERGFingerprint,
    E3FP,
    MAP4Fingerprint,
    MHFP,
    MACCSKeysFingerprint,
    MorganFingerprint,
    TopologicalTorsionFingerprint,
)
from featurizers.map4_mhfp_helper_functions import get_map4_fingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

smiles_data = [
    "Oc1ncnc2c1sc1nc3ccccc3n12",
    "CC1=CC(=C(c2cc(C)c(O)c(C(=O)O)c2)c2c(Cl)ccc(S(=O)(=O)O)c2Cl)C=C(C(=O)O)C1=O.[NaH]",
    "Cc1ccc2nsnc2c1[N+](=O)[O-]",
    "COc1cccc(NC(=O)CC(=O)N2N=C(N(CCC#N)c3ccc(Cl)cc3)CC2c2ccccc2)c1",
    "CCN(CCO)CCNc1ccc(C)c2sc3ccccc3c(=O)c12",
]


@pytest.fixture
def example_molecules():
    return smiles_data


@pytest.fixture
def rdkit_example_molecules():
    return [Chem.MolFromSmiles(x) for x in smiles_data]


def test_morgan_bit_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1, sparse=False, count=False)
    X_emf = morgan.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


# WARNING - in case of failure it will try to overload memory and result in error
def test_morgan_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1, sparse=True, count=False)
    X_emf = morgan.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_morgan_count_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1, sparse=False, count=True)
    X_emf = morgan.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    assert np.all(X_emf == X_rdkit)


def test_morgan_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1, sparse=True, count=True)
    X_emf = morgan.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_atom_pair_bit_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=False, count=False)
    X_emf = atom_pair.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


def test_atom_pair_sparse_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=True, count=False)
    X_emf = atom_pair.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


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
    assert np.all(X_emf == X_rdkit)


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
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


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
    assert np.all(X_emf == X_rdkit)


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
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


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
    assert np.all(X_emf == X_rdkit)


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
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_maccs_keys_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = MACCSKeysFingerprint(n_jobs=-1, sparse=False)
    X_emf = erg.transform(X)
    X_rdkit = np.array([GetMACCSKeysFingerprint(x) for x in X_for_rdkit])

    assert np.all(X_emf == X_rdkit)


def test_maccs_keys_sparse_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = MACCSKeysFingerprint(n_jobs=-1, sparse=True)
    X_emf = erg.transform(X)
    X_rdkit = csr_array([GetMACCSKeysFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_erg_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = ERGFingerprint(n_jobs=-1, sparse=False)
    X_emf = erg.transform(X)
    X_rdkit = np.array([GetErGFingerprint(x) for x in X_for_rdkit])

    assert np.all(X_emf == X_rdkit)


def test_erg_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = ERGFingerprint(n_jobs=-1, sparse=True)
    X_emf = erg.transform(X)
    X_rdkit = csr_array([GetErGFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_map4_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    map4 = MAP4Fingerprint(random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = np.array([get_map4_fingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)

def test_map4_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    map4 = MAP4Fingerprint(sparse=True, random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = csr_array([get_map4_fingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())

def test_map4_count_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    map4 = MAP4Fingerprint(count=True,random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = np.array([get_map4_fingerprint(x,is_counted=True) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)

def test_map4_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    map4 = MAP4Fingerprint(sparse=True,count=True, random_state=0, n_jobs=-1)
    X_emf = map4.transform(X)
    X_rdkit = csr_array([get_map4_fingerprint(x,is_counted=True) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())




def test_mhfp6_fingerprint(example_molecules):
    X = example_molecules

    # Concurrent
    map4_fp = MHFP(random_state=0, n_jobs=-1)
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_map4 = map4_fp.transform(X.copy())

    # Sequential
    map4_fp_seq = MHFP(random_state=0, n_jobs=1)
    X_seq = map4_fp_seq.transform(X)

    assert np.array_equal(X_map4, X_seq)


def test_e3fp(example_molecules):
    X = example_molecules

    X_2 = X.copy()

    confgen_params = {
        "num_conf": NUM_CONF_DEF,
        "first": 1,
        "pool_multiplier": POOL_MULTIPLIER_DEF,
        "rmsd_cutoff": RMSD_CUTOFF_DEF,
        "max_energy_diff": MAX_ENERGY_DIFF_DEF,
        "force_field": FORCEFIELD_DEF,
    }

    fprint_params = {
        "bits": 4096,
        "radius_multiplier": 1.5,
        "rdkit_invariants": True,
    }

    # Concurrent
    e3fp_fp = E3FP(
        **confgen_params,
        **fprint_params,
        is_folded=False,
        standardise=False,
        n_jobs=-1,
        verbose=0
    )
    X_e3fp = e3fp_fp.transform(X)

    confgen_params = {
        "num_conf": NUM_CONF_DEF,
        "first": 1,
        "pool_multiplier": POOL_MULTIPLIER_DEF,
        "rmsd_cutoff": RMSD_CUTOFF_DEF,
        "max_energy_diff": MAX_ENERGY_DIFF_DEF,
        "forcefield": FORCEFIELD_DEF,
        "seed": 0,
    }

    # Sequential
    X_seq = np.array(
        [
            fprints_from_smiles(
                x,
                x,
                confgen_params=confgen_params,
                fprint_params=fprint_params,
            )
            for x in X_2
        ],
        dtype=object,
    )
    X_seq = X_seq.flatten()

    if type(X_seq[0]) is list:
        new_X_seq = []
        for x_seq in X_seq:
            for fp in x_seq:
                new_X_seq.append(fp)

        X_seq = np.array(new_X_seq, dtype=object)

    for i in range(len(X_e3fp)):
        assert tanimoto(X_e3fp[i], X_seq[i]) == 1


def test_input_validation(example_molecules):
    X = example_molecules

    X_test = X.copy()

    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1)
    fp_function = rdkit_generator.GetFingerprint

    X_test = np.array([Chem.MolFromSmiles(x) for x in X_test])
    X_seq = np.array([fp_function(x) for x in X_test])

    # 1) Some of the molecules are still given as SMILES
    X = [Chem.MolFromSmiles(x) if i % 2 == 0 else x for i, x in enumerate(X)]
    X_morgan = morgan.transform(X)
    assert np.all(X_morgan == X_seq)

    # 2) There exists an element in the list, which is not a molecule
    X_2 = np.copy(X)
    X_2[1] = 1

    with pytest.raises(ValueError) as exec_info:
        X_morgan_2 = morgan.transform(X_2)

    assert (
        str(exec_info.value)
        == "Passed value is neither rdkit.Chem.rdChem.Mol nor SMILES"
    )
