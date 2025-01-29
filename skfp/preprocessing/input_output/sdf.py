import os.path
import warnings
from collections.abc import Sequence
from copy import deepcopy

from rdkit.Chem import Mol, SDMolSupplier, SDWriter
from rdkit.Chem.PropertyMol import PropertyMol

from skfp.bases import BasePreprocessor
from skfp.utils import require_mols


class MolFromSDFTransformer(BasePreprocessor):
    """
    Creates RDKit ``Mol`` objects from SDF string or file.

    SDF (structure-data format) is processed for whole files, rather than individual
    molecules. For this reason ``.transform()`` either reads the SDF file directly
    from disk or takes a string input in that format.

    For details see RDKit documentation [1]_.

    Parameters
    ----------
    sanitize : bool, default=True
        Whether to perform sanitization [1]_, i.e. basic validity checks, on created
        molecules.

    remove_hydrogens : bool, default=True
        Remove explicit hydrogens from the molecule where possible, using RDKit
        implicit hydrogens instead.

    References
    ----------
    .. [1] `RDKit SDMolSupplier documentation
        <https://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.SDMolSupplier>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromSDFTransformer
    >>> sdf_file_path = "mols_in.sdf"
    >>> mol_from_sdf = MolFromSDFTransformer()  # doctest: +SKIP
    >>> mol_from_sdf  # doctest: +SKIP
    MolFromSDFTransformer()

    >>> mol_from_sdf.transform(sdf_file_path)  # doctest: +SKIP
        [<rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>]
    """

    _parameter_constraints: dict = {
        "sanitize": ["boolean"],
        "remove_hydrogens": ["boolean"],
    }

    def __init__(
        self,
        sanitize: bool = True,
        remove_hydrogens: bool = True,
    ):
        super().__init__()
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens

    def transform(self, X: str, copy: bool = False) -> list[Mol]:  # type: ignore[override] # noqa: ARG002
        """
        Create RDKit ``Mol`` objects from SDF file.

        Parameters
        ----------
        X : str
            Path to SDF file.

        copy : bool, default=False
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        X : list of shape (n_samples_conf_gen,)
            List with RDKit ``Mol`` objects.
        """
        self._validate_params()

        if X.endswith(".sdf"):
            if not os.path.exists(X):
                raise FileNotFoundError(f"SDF file at path '{X}' not found")

            with open(X) as file:
                X = file.read()

        supplier = SDMolSupplier()
        supplier.SetData(X, sanitize=self.sanitize, removeHs=self.remove_hydrogens)
        mols = list(supplier)

        if not mols:
            warnings.warn("No molecules detected in provided SDF file")

        return mols

    def _transform_batch(self, X):
        pass  # unused


class MolToSDFTransformer(BasePreprocessor):
    """
    Creates SDF file from RDKit ``Mol`` objects.

    SDF (structure-data format) is processed for whole files, rather than individual
    molecules. For this reason ``.transform()`` saves the results directly to file.

    If ``conf_id`` integer property is set for molecules, they are used to determine
    the conformer to save.

    For details see RDKit documentation [1]_.

    Parameters
    ----------
    filepath : string, default="mols.sdf"
        A string with file path location to save the SDF file. Tt should be a valid
        file path with ``.sdf`` extension.

    kekulize : bool, default=True
        Whether to kekulize molecules before writing them to SDF file.

    force_V3000 : bool, default=False
        Whether to force the V3000 format when writing to SDF file.

    References
    ----------
    .. [1] `RDKit SDWriter documentation
        <https://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.SDWriter>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromSDFTransformer, MolToSDFTransformer
    >>> sdf_file_path = "mols_in.sdf"
    >>> mol_from_sdf = MolFromSDFTransformer()
    >>> mol_to_sdf = MolToSDFTransformer(filepath="mols_out.sdf")
    >>> mol_to_sdf
    MolToSDFTransformer(filepath='mols_out.sdf')

    >>> mols = mol_from_sdf.transform(sdf_file_path)  # doctest: +SKIP
    >>> mol_to_sdf.transform(mols)  # doctest: +SKIP
    """

    _parameter_constraints: dict = {
        "filepath": [str],
        "kekulize": ["boolean"],
        "force_V3000": ["boolean"],
    }

    def __init__(
        self,
        filepath: str = "mols.sdf",
        kekulize: bool = True,
        force_V3000: bool = False,
    ):
        super().__init__()
        self.filepath = filepath
        self.kekulize = kekulize
        self.force_V3000 = force_V3000

    def transform(self, X: Sequence[Mol], copy: bool = False) -> None:
        """
        Write RDKit ``Mol`` objects to SDF file at location given by
        ``filepath`` parameter. File is created if necessary, and overwritten
        if it exists already.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit ``Mol`` objects.

        copy : bool, default=False
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
            None
        """
        self._validate_params()
        require_mols(X)

        if copy:
            X = deepcopy(X)

        with open(self.filepath, "w") as file:
            writer = SDWriter(file)
            writer.SetKekulize(self.kekulize)
            writer.SetForceV3000(self.force_V3000)

            for mol in X:
                if isinstance(mol, PropertyMol) and mol.HasProp("conf_id"):
                    writer.write(mol, confId=mol.GetIntProp("conf_id"))
                else:
                    writer.write(mol)

            writer.flush()

    def _transform_batch(self, X):
        pass  # unused
