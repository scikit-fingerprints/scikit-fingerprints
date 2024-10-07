import pytest
from rdkit.Chem import Mol, MolFromFASTA, MolToSmiles

from skfp.preprocessing import MolFromAminoseqTransformer


@pytest.fixture
def fasta_list():
    return [
        ">peptide_pm_1\nKWLRRVWRWWR\n",
        ">peptide_pm_2\nFLPAIGRVLSGIL\n",
        ">peptide_pm_3\nCGESCVWIPCISAVVGCSCKSKVCYKNGTLP\n",
        ">peptide_pm_4\nILGKLLSTAWGLLSKL\n",
        ">peptide_pm_5\nWKLFKKIPKFLHLAKKF\n",
        ">peptide_pm_6\nRAGLQFPVGRLLRRLLRRLLR\n",
        ">peptide_pm_7\nGLWSKIKTAGKSVAKAAAKAAVKAVTNAV\n",
        ">peptide_pm_8\nCGESCVYIPCLTSAIGCSCKSKVCYRNGIP\n",
    ]


@pytest.fixture
def sequence_list(fasta_list):
    return [fst.split("\n")[1] for fst in fasta_list]


@pytest.fixture
def peptide_list(fasta_list):
    return [MolFromFASTA(fst) for fst in fasta_list]


@pytest.fixture
def peptide_smiles(peptide_list):
    return [MolToSmiles(mol) for mol in peptide_list]


def test_mol_from_fasta(fasta_list):
    mol_from_fasta = MolFromAminoseqTransformer()
    mols_list = mol_from_fasta.transform(fasta_list)

    assert len(mols_list) == len(fasta_list)
    assert all(isinstance(x, Mol) for x in mols_list)


def test_mol_to_and_from_fasta(fasta_list, peptide_smiles):
    mol_from_fasta = MolFromAminoseqTransformer()

    peptide_list = mol_from_fasta.transform(fasta_list)
    smiles_list = [MolToSmiles(mol) for mol in peptide_list]

    assert smiles_list == peptide_smiles


def test_mol_from_fasta_and_sequence(fasta_list, sequence_list):
    mol_from_fasta = MolFromAminoseqTransformer()
    mol_from_sequence = MolFromAminoseqTransformer()

    mols_list_fasta = mol_from_fasta.transform(fasta_list)
    mols_list_sequence = mol_from_sequence.transform(sequence_list)

    smiles_list_fasta = [MolToSmiles(mol) for mol in mols_list_fasta]
    smiles_list_sequence = [MolToSmiles(mol) for mol in mols_list_sequence]

    assert smiles_list_fasta == smiles_list_sequence


def test_parallel_to_and_from_fasta(peptide_list, fasta_list):
    mol_from_fasta_seq = MolFromAminoseqTransformer()
    mols_list_seq = mol_from_fasta_seq.transform(fasta_list)
    smiles_list_seq = [MolToSmiles(mol) for mol in mols_list_seq]

    mol_from_fasta_parallel = MolFromAminoseqTransformer(n_jobs=-1)
    mols_list_parallel = mol_from_fasta_parallel.transform(fasta_list)
    smiles_list_parallel = [MolToSmiles(mol) for mol in mols_list_parallel]

    assert smiles_list_seq == smiles_list_parallel
