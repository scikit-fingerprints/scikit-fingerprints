import numpy as np
import pytest
from rdkit.Chem import AddHs, MolFromSmiles

from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer


def test_conformer_generator(smallest_mols_list):
    conf_gen = ConformerGenerator(n_jobs=-1)
    mols_with_confs = conf_gen.transform(smallest_mols_list)

    assert len(mols_with_confs) == len(smallest_mols_list)
    assert all(mol.HasProp("conf_id") for mol in mols_with_confs)


def test_conformer_generator_too_few_tries():
    # this molecule has hard conformers, requiring many tries
    mol_from_smiles = MolFromSmilesTransformer()
    mols = mol_from_smiles.transform(["C1C2CC3C1C1CN3C21"])

    conf_gen = ConformerGenerator(suppress_warnings=True, max_gen_attempts=10)
    with pytest.raises(ValueError) as exc_info:
        conf_gen.transform(mols)

    assert "Could not generate conformer for" in str(exc_info)


def test_conformer_generator_with_hydrogens(smallest_mols_list):
    conf_gen = ConformerGenerator(n_jobs=-1)
    mols_with_confs = conf_gen.transform(smallest_mols_list)

    smallest_mols_with_hs = [AddHs(mol) for mol in smallest_mols_list]
    mols_with_confs_2 = conf_gen.transform(smallest_mols_with_hs)

    assert len(mols_with_confs) == len(smallest_mols_list)
    assert len(mols_with_confs_2) == len(smallest_mols_list)
    assert all(mol.HasProp("conf_id") for mol in mols_with_confs_2)
    assert all(
        mol.GetIntProp("conf_id") == mol_2.GetIntProp("conf_id")
        for mol, mol_2 in zip(mols_with_confs, mols_with_confs_2, strict=False)
    )


def test_conformer_generator_force_field_optimization(smallest_mols_list):
    conf_gen = ConformerGenerator(optimize_force_field="UFF", n_jobs=-1)
    mols_with_confs = conf_gen.transform(smallest_mols_list)

    assert len(mols_with_confs) == len(smallest_mols_list)
    assert all(mol.HasProp("conf_id") for mol in mols_with_confs)


def test_conformer_generator_multiple_conformers(smallest_mols_list):
    conf_gen = ConformerGenerator(
        num_conformers=3,
        optimize_force_field="UFF",
        multiple_confs_select="min_energy",
        n_jobs=-1,
    )
    mols_with_confs = conf_gen.transform(smallest_mols_list)

    assert len(mols_with_confs) == len(smallest_mols_list)
    assert all(mol.HasProp("conf_id") for mol in mols_with_confs)


def test_conformer_generator_error_handling(smallest_mols_list):
    y = np.zeros(len(smallest_mols_list))

    # last molecule has hard conformers, requiring many tries
    mol_from_smiles = MolFromSmilesTransformer()
    mols = mol_from_smiles.transform(["O", "CC", "[C-]#N", "CC=O", "C1C2CC3C1C1CN3C21"])

    conf_gen = ConformerGenerator(max_gen_attempts=100, errors="raise", n_jobs=-1)
    with pytest.raises(ValueError) as exc_info:
        conf_gen.transform_x_y(mols, y)
    assert "Could not generate conformer for" in str(exc_info)

    conf_gen = ConformerGenerator(max_gen_attempts=100, errors="filter", n_jobs=-1)
    mols_conf, y_conf = conf_gen.transform_x_y(mols, y)
    assert all(mol.HasProp("conf_id") for mol in mols_conf)
    assert len(mols_conf) == len(y_conf)
    assert len(mols_conf) == len(mols) - 1

    conf_gen = ConformerGenerator(max_gen_attempts=100, errors="ignore", n_jobs=-1)
    mols_conf, y_conf = conf_gen.transform_x_y(mols, y)
    assert all(mol.HasProp("conf_id") for mol in mols_conf)
    assert mols_conf[-1].GetIntProp("conf_id") == -1
    assert len(mols_conf) == len(y_conf)
    assert len(mols_conf) == len(mols)


def test_conformer_generator_copy_y():
    mols = [MolFromSmiles("O")]
    labels = np.array([1])
    conf_gen = ConformerGenerator()
    mols_out, labels_out = conf_gen.transform_x_y(mols, labels, copy=True)

    assert np.array_equal(labels_out, labels)
    assert labels_out is not labels
