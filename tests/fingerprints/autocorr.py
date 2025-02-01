import numpy as np
from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR2D, CalcAUTOCORR3D

from skfp.fingerprints import AutocorrFingerprint


def test_autocorr_fingerprint(smiles_list, mols_list):
    autocorr_fp = AutocorrFingerprint(use_3D=False, n_jobs=-1)
    X_skfp = autocorr_fp.transform(smiles_list)
    X_rdkit = np.array([CalcAUTOCORR2D(mol) for mol in mols_list])

    assert np.allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 192)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_autocorr_3D_fingerprint(mols_conformers_list):
    autocorr_fp = AutocorrFingerprint(use_3D=True, n_jobs=-1)
    X_skfp = autocorr_fp.transform(mols_conformers_list)

    X_rdkit = np.array(
        [
            CalcAUTOCORR3D(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_list
        ]
    )

    assert np.allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(mols_conformers_list), 80)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_autocorr_feature_names():
    autocorr_fp = AutocorrFingerprint()
    feature_names = autocorr_fp.get_feature_names_out()

    assert len(feature_names) == autocorr_fp.n_features_out
    assert len(feature_names) == len(set(feature_names))

    assert feature_names[0] == "Moreau-Broto mass distance 0"
    assert feature_names[1] == "Moreau-Broto mass distance 1"
    assert feature_names[8] == "Moreau-Broto van der Waals volume distance 0"
    assert feature_names[48] == "Centered Moreau-Broto mass distance 0"
    assert feature_names[-2] == "Geary IState distance 6"
    assert feature_names[-1] == "Geary IState distance 7"


def test_autocorr_3D_feature_names():
    autocorr_fp = AutocorrFingerprint(use_3D=True)
    feature_names = autocorr_fp.get_feature_names_out()

    assert len(feature_names) == autocorr_fp.n_features_out
    assert len(feature_names) == len(set(feature_names))

    assert feature_names[0] == "Moreau-Broto unweighted distance 0"
    assert feature_names[1] == "Moreau-Broto unweighted distance 1"
    assert feature_names[10] == "Moreau-Broto mass distance 0"
    assert feature_names[-2] == "Moreau-Broto covalent radius distance 8"
    assert feature_names[-1] == "Moreau-Broto covalent radius distance 9"
