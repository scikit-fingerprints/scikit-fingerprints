from numbers import Integral
from typing import Sequence

from rdkit.Chem import AddHs, Mol, MolToSmiles, RemoveHs
from rdkit.Chem.rdDistGeom import EmbedMolecule, ETKDGv3
from sklearn.utils import Interval

from skfp.preprocessing.base import BasePreprocessor


class ConformerGenerator(BasePreprocessor):
    _parameter_constraints: dict = {
        "max_conf_gen_attempts": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(self, max_conf_gen_attempts: int = 1000, random_state: int = 0):
        self.max_conf_gen_attempts = max_conf_gen_attempts
        self.random_state = random_state

    def transform(self, X: Sequence[Mol], copy: bool = False) -> list[Mol]:
        # adding hydrogens is recommended for conformer generation
        X = [AddHs(mol) for mol in X]

        conformer_ids = [self._embed_molecule(mol) for mol in X]

        X = [RemoveHs(mol) for mol in X]

        for i in range(len(X)):
            X[i].conf_id = conformer_ids[i]

        return X

    def _embed_molecule(self, mol: Mol) -> int:
        conf_id = -1

        # we create a new embedding params for each molecule, since it can
        # get modified if default settings fail to generate conformers
        embed_params = ETKDGv3()
        embed_params.useSmallRingTorsions = True
        embed_params.randomSeed = self.random_state

        try:
            # basic attempt
            conf_id = EmbedMolecule(mol, embed_params)
        except ValueError:
            pass

        if conf_id == -1:
            try:
                # more tries
                embed_params.maxIterations = self.max_conf_gen_attempts
                embed_params.useRandomCoords = True
                conf_id = EmbedMolecule(mol, embed_params)
            except ValueError:
                pass

        if conf_id == -1:
            try:
                # turn off conditions
                embed_params.enforceChirality = False
                embed_params.ignoreSmoothingFailures = True
                conf_id = EmbedMolecule(mol, embed_params)
            except ValueError:
                pass

        # we should not fail at this point
        if conf_id == -1:
            smiles = MolToSmiles(mol)
            raise ValueError(f"Could not generate conformer for {smiles}")

        return conf_id
