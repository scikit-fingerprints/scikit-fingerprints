from skfp.preprocessing.filters.utils import get_num_charged_functional_groups


def test_get_num_charged_functional_groups(mols_list):
    # just a smoke test - this should not error
    for mol in mols_list:
        get_num_charged_functional_groups(mol)
