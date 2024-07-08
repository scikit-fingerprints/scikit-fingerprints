from rdkit.Chem import MolFromSmiles
from rdkit.rdBase import LogToPythonStderr

from skfp.utils import no_rdkit_logs


def test_no_rdkit_logs(capsys):
    LogToPythonStderr()

    MolFromSmiles("X")
    assert "SMILES Parse Error" in capsys.readouterr().err
    assert capsys.readouterr().out == ""

    with no_rdkit_logs():
        MolFromSmiles("X")

    assert capsys.readouterr().err == ""
    assert capsys.readouterr().out == ""
