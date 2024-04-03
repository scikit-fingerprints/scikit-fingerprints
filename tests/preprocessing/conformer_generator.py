import pytest
from rdkit.Chem import AddHs

from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer


def test_conformer_generator(smallest_mols_list):
    conf_gen = ConformerGenerator()
    mols_with_confs = conf_gen.transform(smallest_mols_list)

    assert len(mols_with_confs) == len(smallest_mols_list)
    assert all(hasattr(mol, "conf_id") for mol in mols_with_confs)


def test_conformer_generator_too_few_tries():
    # this molecule does not have any conformers
    # https://greglandrum.github.io/rdkit-blog/posts/2023-05-17-understanding-confgen-errors.html
    mol_from_smiles = MolFromSmilesTransformer()
    mols = mol_from_smiles.transform([r"C1C[C@]2(F)CC[C@]1(Cl)C2"])

    conf_gen = ConformerGenerator(max_conf_gen_attempts=1)
    with pytest.raises(ValueError) as exc_info:
        conf_gen.transform(mols)

    assert "Could not generate conformer for" in str(exc_info)


def test_conformer_generator_with_hydrogens(smallest_mols_list):
    conf_gen = ConformerGenerator()
    mols_with_confs = conf_gen.transform(smallest_mols_list)

    smallest_mols_with_hs = [AddHs(mol) for mol in smallest_mols_list]
    mols_with_confs_2 = conf_gen.transform(smallest_mols_with_hs)

    assert len(mols_with_confs) == len(smallest_mols_list)
    assert len(mols_with_confs_2) == len(smallest_mols_list)
    assert all(hasattr(mol, "conf_id") for mol in mols_with_confs_2)
    assert all(
        mol.conf_id == mol_2.conf_id
        for mol, mol_2 in zip(mols_with_confs, mols_with_confs_2)
    )
