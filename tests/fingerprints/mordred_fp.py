import numpy as np
from mordred import Calculator, descriptors
from mordred.error import Missing
from scipy.sparse import csr_array

from skfp.fingerprints import MordredFingerprint


def test_mordred_fingerprint(smiles_list, mols_list):
    mordred_transformer = MordredFingerprint(sparse=False, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smiles_list)

    calc = Calculator(descriptors, ignore_3D=True)
    X_seq = [calc(mol) for mol in mols_list]
    X_seq = np.array(
        [x if not isinstance(x, Missing) else np.nan for x in X_seq], dtype=np.float32
    )

    assert ((X_skfp == X_seq) | (np.isnan(X_skfp) & np.isnan(X_seq))).all()


def test_mordred_sparse_fingerprint(smiles_list, mols_list):
    mordred_transformer = MordredFingerprint(sparse=True, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smiles_list)

    calc = Calculator(descriptors, ignore_3D=True)
    X_seq = [calc(mol) for mol in mols_list]
    X_seq = csr_array(
        [x if not isinstance(x, Missing) else np.nan for x in X_seq], dtype=np.float32
    )

    assert (
        (X_skfp.data == X_seq.data) | (np.isnan(X_skfp.data) & np.isnan(X_seq.data))
    ).all()


def test_mordred_3d_fingerprint(smiles_list, mols_list):
    mordred_transformer = MordredFingerprint(ignore_3D=False, sparse=False, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smiles_list)

    calc = Calculator(descriptors, ignore_3D=False)
    X_seq = [calc(mol) for mol in mols_list]
    X_seq = np.array(
        [x if not isinstance(x, Missing) else np.nan for x in X_seq], dtype=np.float32
    )

    assert ((X_skfp == X_seq) | (np.isnan(X_skfp) & np.isnan(X_seq))).all()


def test_mordred_3d_sparse_fingerprint(smiles_list, mols_list):
    mordred_transformer = MordredFingerprint(ignore_3D=False, sparse=True, n_jobs=-1)
    X_skfp = mordred_transformer.transform(smiles_list)

    calc = Calculator(descriptors, ignore_3D=False)
    X_seq = [calc(mol) for mol in mols_list]
    X_seq = csr_array(
        [x if not isinstance(x, Missing) else np.nan for x in X_seq], dtype=np.float32
    )

    assert (
        (X_skfp.data == X_seq.data) | (np.isnan(X_skfp.data) & np.isnan(X_seq.data))
    ).all()
