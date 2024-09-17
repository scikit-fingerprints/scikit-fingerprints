import numpy as np
from rdkit.Chem.rdMolDescriptors import CalcRDF
from scipy.sparse import csr_array

from skfp.fingerprints import RDFFingerprint


def test_rdf_fingerprint(mols_conformers_list):
    rdf_fp = RDFFingerprint(n_jobs=-1)
    X_skfp = rdf_fp.transform(mols_conformers_list)

    X_rdkit = np.array(
        [CalcRDF(mol, confId=mol.GetIntProp("conf_id")) for mol in mols_conformers_list]
    )

    assert np.allclose(X_skfp, X_rdkit, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 210)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_rdf_sparse_fingerprint(mols_conformers_list):
    rdf_fp = RDFFingerprint(sparse=True, n_jobs=-1)
    X_skfp = rdf_fp.transform(mols_conformers_list)

    X_rdkit = csr_array(
        [CalcRDF(mol, confId=mol.GetIntProp("conf_id")) for mol in mols_conformers_list]
    )

    assert np.allclose(X_skfp.data, X_rdkit.data, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 210)
