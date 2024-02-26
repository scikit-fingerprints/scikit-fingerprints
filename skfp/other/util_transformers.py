from rdkit.Chem import MolFromSmiles, MolToSmiles


class MolFromSmilesTransformer:
    def __init__(
        self,
        sanitize: bool = True,
        replacements: dict = {},
    ):
        self.sanitize = sanitize
        self.replacements = replacements

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        return [
            MolFromSmiles(
                x, sanitize=self.sanitize, replacements=self.replacements
            )
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

    def transform(self, X):
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
