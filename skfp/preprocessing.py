from typing import List, Optional, Sequence, Tuple

from rdkit.Chem import AddHs, Mol, MolFromSmiles, MolToSmiles, RemoveHs
from rdkit.Chem.rdDistGeom import EmbedMolecule, ETKDGv3


class MolFromSmilesTransformer:
    def __init__(
        self,
        sanitize: bool = True,
        replacements: Optional[dict] = None,
    ):
        self.sanitize = sanitize
        self.replacements = replacements if replacements else {}

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X: Sequence[str]) -> List[Mol]:
        return [
            MolFromSmiles(x, sanitize=self.sanitize, replacements=self.replacements)
            for x in X
        ]


class MolToSmilesTransformer:
    def __init__(
        self,
        isomeric_smiles: bool = True,
        kekule_smiles: bool = False,
        rooted_at_atom: int = -1,
        canonical: bool = True,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        do_random: bool = False,
    ):
        self.isomeric_smiles = isomeric_smiles
        self.kekule_smiles = kekule_smiles
        self.rooted_at_atom = rooted_at_atom
        self.canonical = canonical
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.do_random = do_random

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X: Sequence[Mol]) -> List[str]:
        return [
            MolToSmiles(
                x,
                isomericSmiles=self.isomeric_smiles,
                kekuleSmiles=self.kekule_smiles,
                rootedAtAtom=self.rooted_at_atom,
                canonical=self.canonical,
                allBondsExplicit=self.all_bonds_explicit,
                allHsExplicit=self.all_hs_explicit,
                doRandom=self.do_random,
            )
            for x in X
        ]


class ConformerGenerator:
    def __init__(self, max_conf_gen_attempts: int = 1000, random_state: int = 0):
        self.max_conf_gen_attempts = max_conf_gen_attempts
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X: Sequence[Mol]) -> List[Mol]:
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
