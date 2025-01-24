# copyright notice for MACCS, which is based on RDKit:
#
#  Copyright (C) 2002-2012 Greg Landrum and Rational Discovery LLC
#   This file is part of the RDKit.
#   The contents are covered by the terms of the BSD license
#   which is included in the file license.txt, found at the root
#   of the RDKit source tree.
#
# we include copy of that license in skfp/fingerprints/data/RDKit_license.txt
# source: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol, MolToSmiles
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class MACCSFingerprint(BaseFingerprintTransformer):
    """
    MACCS fingerprint.

    The implementation uses RDKit. This is a substructure fingerprint, based on
    publicly available MDL definitions, and refined by Gregory Landrum for RDKit [1]_.
    Note that full public definitions are not available, and packages differ [2]_.

    Results differ slightly from RDKit, because it adds an artificial 0th feature with
    all zeros, to simulate 1-based indexing [3]_. We use exactly 166 features, removing
    said constant feature. First feature is also always zero, since it means "ISOTOPE",
    not supported by RDKit. Consider removing them before to further processing, e.g.
    using ``VarianceThreshold``.

    Count variant is an original one. It counts substructures instead of only checking
    for their existence. It also has fewer features, because RDKit MACCS has separate
    features checking e.g. the number of oxygens. The ordering of features also differs,
    and there are no constant zero features.

    Parameters
    ----------
    count : bool, default=False
        Whether to return binary (bit) features, or the count-based variant.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int = 166 or 158.
        Number of output features, size of fingerprints. Equal to 166 for the bit
        variant, and 158 for count.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] RDKit MACCS implementation
        https://github.com/rdkit/rdkit/blob/3457c1eb60846ea821e4a319f3505933027d3cf8/rdkit/Chem/MACCSkeys.py

    .. [2] `Andrew Dalke
        "MACCS key 44"
        <http://www.dalkescientific.com/writings/diary/archive/2014/10/17/maccs_key_44.html>`_

    .. [3] RDKit MACCS 167 features discussion
        https://github.com/rdkit/rdkit/issues/1726

    Examples
    --------
    >>> from skfp.fingerprints import MACCSFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = MACCSFingerprint()
    >>> fp
    MACCSFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 1, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 1, 0, 0]], dtype=uint8)
    """

    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        n_features_out = 158 if count else 166
        super().__init__(
            n_features_out=n_features_out,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to substructure
        names defined by Greg Landrum, based on publicly available MACCS Keys
        definitions.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            MACCS feature names.
        """
        if not self.count:
            # https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py
            feature_names = [
                "ISOTOPE",
                "atomic num >103",
                "Group IVa,Va,VIa Rows 4-6 ",
                "actinide",
                "Group IIIB,IVB",
                "Lanthanide",
                "Group VB,VIB,VIIB",
                "QAAA@1",
                "Group VIII (Fe...)",
                "Group IIa (Alkaline earth)",
                "4M Ring",
                "Group IB,IIB (Cu..)",
                "ON(C)C",
                "S-S",
                "OC(O)O",
                "QAA@1",
                "CTC",
                "Group IIIA (B...) ",
                "7M Ring",
                "Si",
                "C=C(Q)Q",
                "3M Ring",
                "NC(O)O",
                "N-O",
                "NC(N)N",
                "C$=C($A)$A",
                "I",
                "QCH2Q",
                "P",
                "CQ(C)(C)A",
                "QX",
                "CSN",
                "NS",
                "CH2=A",
                "Group IA (Alkali Metal)",
                "S Heterocycle",
                "NC(O)N",
                "NC(C)N",
                "OS(O)O",
                "S-O",
                "CTN",
                "F",
                "QHAQH",
                "OTHER",
                "C=CN",
                "BR",
                "SAN",
                "OQ(O)O",
                "CHARGE  ",
                "C=C(C)C",
                "CSO",
                "NN",
                "QHAAAQH",
                "QHAAQH",
                "OSO",
                "ON(O)C",
                "O Heterocycle",
                "QSQ",
                "Snot%A%A",
                "S=O",
                "AS(A)A",
                "A$!A$A",
                "N=O",
                "A$A!S",
                "C%N",
                "CC(C)(C)A",
                "QS",
                "QHQH",
                "QQH",
                "QNQ",
                "NO",
                "OAAO",
                "S=A",
                "CH3ACH3",
                "A!N$A",
                "C=C(A)A",
                "NAN",
                "C=N",
                "NAAN",
                "NAAAN",
                "SA(A)A",
                "ACH2QH",
                "QAAAA@1",
                "NH2",
                "CN(C)C",
                "CH2QCH2",
                "X!A$A",
                "S",
                "OAAAO",
                "QHAACH2A",
                "QHAAACH2A",
                "OC(N)C",
                "QCH3",
                "QN",
                "NAAO",
                "5 M ring",
                "NAAAO",
                "QAAAAA@1",
                "C=C",
                "ACH2N",
                "8M Ring or larger",
                "QO",
                "CL",
                "QHACH2A",
                "A$A($A)$A",
                "QA(Q)Q",
                "XA(A)A",
                "CH3AAACH2A",
                "ACH2O",
                "NCO",
                "NACH2A",
                "AA(A)(A)A",
                "Onot%A%A",
                "CH3CH2A",
                "CH3ACH2A",
                "CH3AACH2A",
                "NAO",
                "ACH2CH2A > 1",
                "N=A",
                "Heterocyclic atom > 1",
                "N Heterocycle",
                "AN(A)A",
                "OCO",
                "QQ",
                "Aromatic Ring > 1",
                "A!O!A",
                "A$A!O > 1",
                "ACH2AAACH2A",
                "ACH2AACH2A",
                "QQ > 1",
                "QH > 1",
                "OACH2A",
                "A$A!N",
                "X (HALOGEN)",
                "Nnot%A%A",
                "O=A>1 ",
                "Heterocycle",
                "QCH2A>1",
                "OH",
                "O > 3",
                "CH3 > 2",
                "N > 1",
                "A$A!O",
                "Anot%A%Anot%A",
                "6M ring > 1",
                "O > 2",
                "ACH2CH2A",
                "AQ(A)A",
                "CH3 > 1",
                "A!A$A!A",
                "NH",
                "OC(C)C",
                "QCH2A",
                "C=O",
                "A!CH2!A",
                "NA(A)A",
                "C-O",
                "C-N",
                "O>1",
                "CH3",
                "N",
                "Aromatic",
                "6M Ring",
                "O",
                "Ring",
                "Fragments",
            ]
        else:
            feature_names = [
                "fragments",
                "atomic num >103",
                "N",
                "O",
                "F",
                "Si",
                "P",
                "S",
                "Cl",
                "Br",
                "I",
                "Aromatic",
                "X (HALOGEN)",
                "CHARGE",
                "QH",
                "OTHER",
                "Group IA (Alkali Metal)",
                "Group IB,IIB (Cu..)",
                "Group IIa (Alkaline earth)",
                "Group IIIA (B...)",
                "Group IIIB,IVB (Sc...)",
                "Group IVa,Va,VIa Rows 4-6",
                "Group VB,VIB,VIIB",
                "Group VIII (Fe...)",
                "actinide",
                "Lanthanide",
                "Ring",
                "Aromatic Ring",
                "Heterocyclic atom",
                "Heterocycle",
                "S Heterocycle",
                "O Heterocycle",
                "N Heterocycle",
                "3-membered ring",
                "4-membered ring",
                "5-membered ring",
                "6-membered ring",
                "7-membered ring",
                "8-membered ring or larger",
                "S-S",
                "N-O",
                "NS",
                "S-O",
                "NN",
                "S=O",
                "N=O",
                "C%N",
                "QS",
                "NO",
                "S=A",
                "C=N",
                "NH2",
                "QN",
                "C=C",
                "QO",
                "QX",
                "N=A",
                "QQ",
                "O=A>1",
                "OH",
                "CH3",
                "C;H3,H4",
                "NH",
                "C=O",
                "C-O",
                "C-N",
                "CH3",
                "CTC",
                "CTN",
                "CSO",
                "CSN",
                "CH2=A",
                "C=CN",
                "SAN",
                "OSO",
                "QSQ",
                "Snot%A%A",
                "QQH",
                "QNQ",
                "NAN",
                "NCO",
                "NAO",
                "OCO",
                "A!O!A",
                "A$!A$A",
                "A$A!S",
                "X!A$A",
                "A$A!O",
                "A$A!N",
                "Nnot%A%A",
                "A$A!O",
                "NC(O)N",
                "QAAA@1",
                "ON(C)C",
                "OC(O)O",
                "QAA@1",
                "C=C(Q)Q",
                "NC(O)O",
                "NC(N)N",
                "C$=C($A)$A",
                "QCH2Q",
                "CQ(C)(C)A",
                "OS(O)O",
                "QHAQH",
                "OQ(O)O",
                "C=C(C)C",
                "QHAAAQH",
                "QHAAQH",
                "ON(O)C",
                "AS(A)A",
                "CC(C)(C)A",
                "QHQH",
                "OAAO",
                "CH3ACH3",
                "A!N$A",
                "C=C(A)A",
                "NAAN",
                "NAAAN",
                "SA(A)A",
                "ACH2QH",
                "QAAAA@1",
                "CN(C)C",
                "CH2QCH2",
                "OAAAO",
                "QHAACH2A",
                "QHAAACH2A",
                "OC(N)C",
                "QCH3",
                "NAAO",
                "NAAAO",
                "QAAAAA@1",
                "ACH2N",
                "QHACH2A",
                "A$A($A)$A",
                "QA(Q)Q",
                "XA(A)A",
                "CH3AAACH2A",
                "ACH2O",
                "NACH2A",
                "AA(A)(A)A",
                "Onot%A%A",
                "CH3CH2A",
                "CH3ACH2A",
                "CH3AACH2A",
                "ACH2CH2A",
                "AN(A)A",
                "ACH2AAACH2A",
                "ACH2AACH2A",
                "OACH2A",
                "QCH2A",
                "Anot%A%Anot%A",
                "ACH2CH2A",
                "AQ(A)A",
                "A!A$A!A",
                "OC(C)C",
                "QCH2A",
                "A!CH2!A",
                "NA(A)A",
            ]
        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute MACCS fingerprints. Output shape depends on ``count``
        parameter.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.n_features_out)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

        X = ensure_mols(X)

        if self.count:
            X = [self._get_maccs_patterns_counts(mol) for mol in X]
        else:
            X = [GetMACCSKeysFingerprint(mol) for mol in X]
            X = np.array(X)[:, 1:]  # remove constant zeros column

        dtype = np.uint32 if self.count else np.uint8
        return csr_array(X, dtype=dtype) if self.sparse else np.array(X, dtype=dtype)

    def _get_maccs_patterns_counts(self, mol: Mol) -> list[int]:
        smarts_list = [
            None,  # fragments
            None,  # atomic num >103
            "[#7]",  # N
            "[#8]",  # O
            "[#9]",  # F
            "[#14]",  # Si
            "[#15]",  # P
            "[#16]",  # S
            "[#17]",  # Cl
            "[#35]",  # Br
            "[#53]",  # I
            "a",  # Aromatic
            "[F,Cl,Br,I]",  # X (HALOGEN)
            "[!+0]",  # CHARGE
            "[!#6;!#1;!H0]",  # QH
            "[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]",  # OTHER
            "[Li,Na,K,Rb,Cs,Fr]",  # Group IA (Alkali Metal)
            "[Cu,Zn,Ag,Cd,Au,Hg]",  # Group IB,IIB (Cu..)
            "[Be,Mg,Ca,Sr,Ba,Ra]",  # Group IIa (Alkaline earth)
            "[#5,#13,#31,#49,#81]",  # Group IIIA (B...)
            "[Sc,Ti,Y,Zr,Hf]",  # Group IIIB,IVB (Sc...)
            "[#32,#33,#34,#50,#51,#52,#82,#83,#84]",  # Group IVa,Va,VIa Rows 4-6
            "[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]",  # Group VB,VIB,VIIB
            "[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]",  # Group VIII (Fe...)
            "[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]",  # actinide
            "[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]",  # Lanthanide
            "[R]",  # Ring
            None,  # Aromatic Ring
            "[!#6;R]",  # Heterocyclic atom (&...) Spec Incomplete
            "[!C;!c;R]",  # Heterocycle
            "[#16R]",  # S Heterocycle
            "[#8R]",  # O Heterocycle
            "[#7;R]",  # N Heterocycle
            "*1~*~*~1",  # 3-membered ring
            "*1~*~*~*~1",  # 4-membered ring
            "*1~*~*~*~*~1",  # 5-membered ring
            "*1~*~*~*~*~*~1",  # 6-membered ring
            "*1~*~*~*~*~*~*~1",  # 7-membered ring
            None,  # 8-membered ring or larger - idx 38
            "[#16]-[#16]",  # S-S
            "[#7]-[#8]",  # N-O
            "[#7]~[#16]",  # NS
            "[#16]-[#8]",  # S-O
            "[#7]~[#7]",  # NN
            "[#16]=[#8]",  # S=O
            "[#7]=[#8]",  # N=O
            "c:n",  # C%N
            "[!#6;!#1]~[#16]",  # QS
            "[#7]~[#8]",  # NO
            "[#16]=*",  # S=A
            "[#6]=[#7]",  # C=N
            "[NH2]",  # NH2
            "[!#6;!#1]~[#7]",  # QN
            "[#6]=[#6]",  # C=C
            "[!#6;!#1]~[#8]",  # QO
            "[!#6;!#1]~[F,Cl,Br,I]",  # QX
            "[#7]=*",  # N=A
            "[!#6;!#1]~[!#6;!#1]",  # QQ
            "[#8]=*",  # O=A>1
            "[O;!H0]",  # OH
            "[CH3]",  # CH3 (&...) Spec Incomplete
            "[C;H3,H4]",  # CH3
            "[#7;!H0]",  # NH
            "[#6]=[#8]",  # C=O
            "[#6]-[#8]",  # C-O
            "[#6]-[#7]",  # C-N
            "[C;H3,H4]",  # CH3
            "[#6]#[#6]",  # CTC
            "[#6]#[#7]",  # CTN
            "[#6]~[#16]~[#8]",  # CSO
            "[#6]~[#16]~[#7]",  # CSN
            "[CH2]=*",  # CH2=A
            "[#6]=[#6]~[#7]",  # C=CN
            "[#16]~*~[#7]",  # SAN
            "[#8]~[#16]~[#8]",  # OSO
            "[!#6;!#1]~[#16]~[!#6;!#1]",  # QSQ
            "[#16]!:*:*",  # Snot%A%A
            "[!#6;!#1]~[!#6;!#1;!H0]",  # QQH
            "[!#6;!#1]~[#7]~[!#6;!#1]",  # QNQ
            "[#7]~*~[#7]",  # NAN
            "[#7]~[#6]~[#8]",  # NCO
            "[#7]~*~[#8]",  # NAO
            "[#8]~[#6]~[#8]",  # OCO
            "*!@[#8]!@*",  # A!O!A
            "*@*!@*@*",  # A$!A$A
            "*@*!@[#16]",  # A$A!S
            "[F,Cl,Br,I]!@*@*",  # X!A$A
            "*@*!@[#8]",  # A$A!O (&...) Spec Incomplete
            "*@*!@[#7]",  # A$A!N
            "[#7]!:*:*",  # Nnot%A%A
            "*@*!@[#8]",  # A$A!O
            "[#7]~[#6](~[#8])~[#7]",  # NC(O)N
            "[!#6;!#1]1~*~*~*~1",  # QAAA@1
            "[#8]~[#7](~[#6])~[#6]",  # ON(C)C
            "[#8]~[#6](~[#8])~[#8]",  # OC(O)O
            "[!#6;!#1]1~*~*~1",  # QAA@1
            "[#6]=[#6](~[!#6;!#1])~[!#6;!#1]",  # C=C(Q)Q
            "[#7]~[#6](~[#8])~[#8]",  # NC(O)O
            "[#7]~[#6](~[#7])~[#7]",  # NC(N)N
            "[#6]=;@[#6](@*)@*",  # C$=C($A)$A
            "[!#6;!#1]~[CH2]~[!#6;!#1]",  # QCH2Q
            "[#6]~[!#6;!#1](~[#6])(~[#6])~*",  # CQ(C)(C)A
            "[#8]~[#16](~[#8])~[#8]",  # OS(O)O
            "[!#6;!#1;!H0]~*~[!#6;!#1;!H0]",  # QHAQH
            "[#8]~[!#6;!#1](~[#8])(~[#8])",  # OQ(O)O
            "[#6]=[#6](~[#6])~[#6]",  # C=C(C)C
            "[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]",  # QHAAAQH
            "[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]",  # QHAAQH
            "[#8]~[#7](~[#8])~[#6]",  # ON(O)C
            "*~[#16](~*)~*",  # AS(A)A
            "[#6]~[#6](~[#6])(~[#6])~*",  # CC(C)(C)A
            "[!#6;!#1;!H0]~[!#6;!#1;!H0]",  # QHQH (&...) SPEC Incomplete
            "[#8]~*~*~[#8]",  # OAAO
            "[CH3]~*~[CH3]",  # CH3ACH3
            "*!@[#7]@*",  # A!N$A
            "[#6]=[#6](~*)~*",  # C=C(A)A
            "[#7]~*~*~[#7]",  # NAAN
            "[#7]~*~*~*~[#7]",  # NAAAN
            "[#16]~*(~*)~*",  # SA(A)A
            "*~[CH2]~[!#6;!#1;!H0]",  # ACH2QH
            "[!#6;!#1]1~*~*~*~*~1",  # QAAAA@1
            "[#6]~[#7](~[#6])~[#6]",  # CN(C)C
            "[C;H2,H3][!#6;!#1][C;H2,H3]",  # CH2QCH2
            "[#8]~*~*~*~[#8]",  # OAAAO
            # QHAACH2A
            "[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]",
            # QHAAACH2A
            "[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]",
            "[#8]~[#6](~[#7])~[#6]",  # OC(N)C
            "[!#6;!#1]~[CH3]",  # QCH3
            "[#7]~*~*~[#8]",  # NAAO
            "[#7]~*~*~*~[#8]",  # NAAAO
            "[!#6;!#1]1~*~*~*~*~*~1",  # QAAAAA@1
            "*~[CH2]~[#7]",  # ACH2N
            "[!#6;!#1;!H0]~*~[CH2]~*",  # QHACH2A
            "*@*(@*)@*",  # A$A($A)$A
            "[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]",  # QA(Q)Q
            "[F,Cl,Br,I]~*(~*)~*",  # XA(A)A
            "[CH3]~*~*~*~[CH2]~*",  # CH3AAACH2A
            "*~[CH2]~[#8]",  # ACH2O
            "[#7]~*~[CH2]~*",  # NACH2A
            "*~*(~*)(~*)~*",  # AA(A)(A)A
            "[#8]!:*:*",  # Onot%A%A
            "[CH3]~[CH2]~*",  # CH3CH2A
            "[CH3]~*~[CH2]~*",  # CH3ACH2A
            "[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]",  # CH3AACH2A
            "[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]",  # ACH2CH2A
            "*~[#7](~*)~*",  # AN(A)A
            "[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]",
            # ACH2AAACH2A
            "[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]",  # ACH2AACH2A
            "[#8]~*~[CH2]~*",  # OACH2A
            "[!#6;!#1]~[CH2]~*",  # QCH2A (&...) Spec Incomplete
            "*!:*:*!:*",  # Anot%A%Anot%A
            "[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]",  # ACH2CH2A
            "*~[!#6;!#1](~*)~*",  # AQ(A)A
            "*!@*@*!@*",  # A!A$A!A
            "[#8]~[#6](~[#6])~[#6]",  # OC(C)C
            "[!#6;!#1]~[CH2]~*",  # QCH2A
            "*!@[CH2]!@*",  # A!CH2!A
            "[#7]~*(~*)~*",  # NA(A)A
        ]
        counts = self._get_smarts_match_counts(mol, smarts_list)

        # here we fix positions that can't be easily written as SMARTS

        # number of fragments
        counts[0] = MolToSmiles(mol).count(".") + 1

        # atomic number over 103
        counts[1] = sum(atom.GetAtomicNum() > 103 for atom in mol.GetAtoms())

        # aromatic rings
        counts[28] = sum(
            all(mol.GetBondWithIdx(bond_idx).GetIsAromatic() for bond_idx in ring)
            for ring in mol.GetRingInfo().BondRings()
        )

        # 8-membered rings or larger
        counts[39] = sum(1 for ring in mol.GetRingInfo().BondRings() if len(ring) >= 8)

        return counts

    def _get_smarts_match_counts(
        self, mol: Mol, smarts_list: list[Optional[str]]
    ) -> list[int]:
        from rdkit.Chem import MolFromSmarts

        return [
            (
                len(mol.GetSubstructMatches(MolFromSmarts(smarts)))
                if smarts is not None
                else 0
            )
            for smarts in smarts_list
        ]
