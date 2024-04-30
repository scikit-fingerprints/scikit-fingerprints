import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import GetUSR, GetUSRCAT
from scipy.sparse import csr_array

from skfp.fingerprints import USRDescriptor


def test_usr_bit_fingerprint(mols_conformers_list):
    usr_fp = USRDescriptor(sparse=False, n_jobs=2)
    X_skfp = usr_fp.transform(mols_conformers_list)

    X_rdkit = []
    for mol in mols_conformers_list:
        try:
            mol_fp = GetUSR(mol)
        except ValueError:
            mol_fp = np.empty((12,))
            mol_fp.fill(np.nan)
        finally:
            X_rdkit.append(mol_fp)
    X_rdkit = np.array(X_rdkit)

    assert np.array_equal(X_skfp, X_rdkit, equal_nan=True)
    assert X_skfp.shape == (len(mols_conformers_list), 12)


def test_usr_sparse_bit_fingerprint(mols_conformers_list):
    usr_fp = USRDescriptor(sparse=True, n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_list)
    X_rdkit = []
    for mol in mols_conformers_list:
        try:
            mol_fp = GetUSR(mol)
        except ValueError:
            mol_fp = np.empty((12,))
            mol_fp.fill(np.nan)
        finally:
            X_rdkit.append(mol_fp)
    X_rdkit = csr_array(X_rdkit)

    assert np.array_equal(X_skfp.data, X_rdkit.data, equal_nan=True)
    assert X_skfp.shape == (len(mols_conformers_list), 12)
