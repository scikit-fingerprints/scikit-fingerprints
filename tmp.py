from rdkit.Chem import AddHs
from tqdm import tqdm

from skfp.datasets.moleculenet import load_hiv
from skfp.preprocessing import MolFromSmilesTransformer


smiles_list, y = load_hiv()
mols_list = MolFromSmilesTransformer(n_jobs=-1).transform(smiles_list)

for mol in tqdm(mols_list):
    mol2 = AddHs(mol)
    assert mol.GetNumAtoms(onlyExplicit=False) == mol2.GetNumAtoms()
