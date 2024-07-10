from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseSubstructureFingerprint
from skfp.utils import ensure_smiles


class LaggnerFingerprint(BaseSubstructureFingerprint):
    """
    Substructure fingerprint with definitions by Christian Laggner.

    A substructure fingerprint based on SMARTS patterns for functional group
    classification, proposed by Christian Laggner [1]_. It is also known as
    SubstructureFingerprint in Chemistry Development Kit (CDK) [2]_.
    It tests for presence of 307 predefined substructures, designed for functional
    groups of organic compounds, for use in similarity searching.

    Parameters
    ----------
    count : bool, default=False
        Whether to return binary (bit) features, or their counts.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See Scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int, default=0
        Controls the verbosity when computing fingerprints.

    Attributes
    ----------
    n_features_out : int = 307
        Number of output features, size of fingerprints.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] `Christian Laggner
        "SMARTS Patterns for Functional Group Classification"
        OpenBabel
        <https://raw.githubusercontent.com/openbabel/openbabel/master/data/SMARTS_InteLigand.txt>`_

    .. [2] `egonw
        "SubstructureFingerprinter"
        Chemistry Development Kit (CDK) API reference
        <https://cdk.github.io/cdk/1.5/docs/api/org/openscience/cdk/fingerprint/SubstructureFingerprinter.html>`_

    Examples
    --------
    >>> from skfp.fingerprints import LaggnerFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = LaggnerFingerprint()
    >>> fp
    LaggnerFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [1, 0, 0, ..., 1, 0, 1]], dtype=uint8)
    """

    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        # RDKit does not support multi-component SMARTS (with a dot), so we can't match
        # the salts pattern "([-1,-2,-3,-4,-5,-6,-7]).([+1,+2,+3,+4,+5,+6,+7])"
        # (position 299); we compute it manually in .transform(), and here temporarily
        # replace it with a wildcard pattern "*"

        # flake8: noqa: E501
        patterns = [
            "[CX4H3][#6]",
            "[CX4H2]([#6])[#6]",
            "[CX4H1]([#6])([#6])[#6]",
            "[CX4]([#6])([#6])([#6])[#6]",
            "[CX3;$([H2]),$([H1][#6]),$(C([#6])[#6])]=[CX3;$([H2]),$([H1][#6]),$(C([#6])[#6])]",
            "[CX2]#[CX2]",
            "[CX3]=[CX2]=[CX3]",
            "[ClX1][CX4]",
            "[FX1][CX4]",
            "[BrX1][CX4]",
            "[IX1][CX4]",
            "[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]",
            "[OX2H][CX4H2;!$(C([OX2H])[O,S,#7,#15])]",
            "[OX2H][CX4H;!$(C([OX2H])[O,S,#7,#15])]",
            "[OX2H][CX4D4;!$(C([OX2H])[O,S,#7,#15])]",
            "[OX2]([CX4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])])[CX4;!$(C([OX2])[O,S,#7,#15])]",
            "[SX2]([CX4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])])[CX4;!$(C([OX2])[O,S,#7,#15])]",
            "[OX2](c)[CX4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])]",
            "[c][OX2][c]",
            "[SX2](c)[CX4;!$(C([OX2])[O,S,#7,#15,F,Cl,Br,I])]",
            "[c][SX2][c]",
            "[O+;!$([O]~[!#6]);!$([S]*~[#7,#8,#15,#16])]",
            "[NX3+0,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX3H2+0,NX4H3+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX3H1+0,NX4H2+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX3H0+0,NX4H1+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX4H0+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX3H2+0,NX4H3+]c",
            "[NX3H1+0,NX4H2+;!$([N][!c]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX3H0+0,NX4H1+;!$([N][!c]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX4H0+;!$([N][!c]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX3H1+0,NX4H2+;$([N]([c])[C]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX3H0+0,NX4H1+;$([N]([c])([C])[#6]);!$([N]*~[#7,#8,#15,#16])]",
            "[NX4H0+;$([N]([c])([C])[#6][#6]);!$([N]*~[#7,#8,#15,#16])]",
            "[N+;!$([N]~[!#6]);!$(N=*);!$([N]*~[#7,#8,#15,#16])]",
            "[SX2H][CX4;!$(C([SX2H])~[O,S,#7,#15])]",
            "[SX2]([CX4;!$(C([SX2])[O,S,#7,#15,F,Cl,Br,I])])[CX4;!$(C([SX2])[O,S,#7,#15])]",
            "[SX2](c)[CX4;!$(C([SX2])[O,S,#7,#15])]",
            "[SX2D2][SX2D2]",
            "[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15,F,Cl,Br,I])][CX4;!$(C([N])[O,S,#7,#15])][NX3;!$(NC=[O,S,N])]",
            "[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])][CX4;!$(C([OX2H])[O,S,#7,#15])][OX2H]",
            "[OX2H][CX4;!$(C([OX2H])([OX2H])[O,S,#7,#15])][OX2H]",
            "[OX2H][OX2]",
            "[OX2D2][OX2D2]",
            "[LiX1][#6,#14]",
            "[MgX2][#6,#14]",
            "[!#1;!#5;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#33;!#34;!#35;!#52;!#53;!#85]~[#6;!-]",
            "[$([CX3H][#6]),$([CX3H2])]=[OX1]",
            "[#6][CX3](=[OX1])[#6]",
            "[$([CX3H][#6]),$([CX3H2])]=[SX1]",
            "[#6][CX3](=[SX1])[#6]",
            "[NX2;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16])]=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6])]",
            "[NX3+;!$([N][!#6]);!$([N][CX3]=[#7,#8,#15,#16])]",
            "[NX2](=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6])])[OX2H]",
            "[NX2](=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6])])[OX2][#6;!$(C=[#7,#8])]",
            "[OX2]([#6;!$(C=[O,S,N])])[CX4;!$(C(O)(O)[!#6])][OX2][#6;!$(C=[O,S,N])]",
            "[OX2H][CX4;!$(C(O)(O)[!#6])][OX2][#6;!$(C=[O,S,N])]",
            "[NX3v3;!$(NC=[#7,#8,#15,#16])]([#6])[CX4;!$(C(N)(N)[!#6])][NX3v3;!$(NC=[#7,#8,#15,#16])][#6]",
            "[NX3v3;!$(NC=[#7,#8,#15,#16])]([#6])[CX4;!$(C(N)(N)[!#6])][OX2H]",
            "[SX2]([#6;!$(C=[O,S,N])])[CX4;!$(C(S)(S)[!#6])][SX2][#6;!$(C=[O,S,N])]",
            "[SX2]([#6;!$(C=[O,S,N])])[CX4;!$(C(S)(S)[!#6])][OX2H]",
            "[NX3v3,SX2,OX2;!$(*C=[#7,#8,#15,#16])][CX4;!$(C([N,S,O])([N,S,O])[!#6])][FX1,ClX1,BrX1,IX1]",
            "[NX3v3,SX2,OX2;!$(*C=[#7,#8,#15,#16])][CX4;!$(C([N,S,O])([N,S,O])[!#6])][FX1,ClX1,BrX1,IX1,NX3v3,SX2,OX2;!$(*C=[#7,#8,#15,#16])]",
            "[NX3v3,SX2,OX2;$(**=[#7,#8,#15,#16])][CX4;!$(C([N,S,O])([N,S,O])[!#6])][FX1,ClX1,BrX1,IX1]",
            "[NX3v3,SX2,OX2;$(**=[#7,#8,#15,#16])][CX4;!$(C([N,S,O])([N,S,O])[!#6])][NX3v3,SX2,OX2;!$(*C=[#7,#8,#15,#16])]",
            "[NX3v3,SX2,OX2;$(**=[#7,#8,#15,#16])][CX4;!$(C([N,S,O])([N,S,O])[!#6])][FX1,ClX1,BrX1,IX1,NX3v3,SX2,OX2;!$(*C=[#7,#8,#15,#16])]",
            "[NX1]#[CX2][CX4;$([CH2]),$([CH]([CX2])[#6]),$(C([CX2])([#6])[#6])][OX2H]",
            "[ClX1][CX3]=[CX3]",
            "[FX1][CX3]=[CX3]",
            "[BrX1][CX3]=[CX3]",
            "[IX1][CX3]=[CX3]",
            "[OX2H][CX3;$([H1]),$(C[#6])]=[CX3]",
            "[OX2H][CX3;$([H1]),$(C[#6])]=[CX3;$([H1]),$(C[#6])][OX2H]",
            "[OX2]([#6;!$(C=[N,O,S])])[CX3;$([H0][#6]),$([H1])]=[CX3]",
            "[OX2]([CX3]=[OX1])[#6X3;$([#6][#6]),$([H1])]=[#6X3;!$(C[OX2H])]",
            "[NX3;$([NH2][CX3]),$([NH1]([CX3])[#6]),$([N]([CX3])([#6])[#6]);!$([N]*=[#7,#8,#15,#16])][CX3;$([CH]),$([C][#6])]=[CX3]",
            "[SX2H][CX3;$([H1]),$(C[#6])]=[CX3]",
            "[SX2]([#6;!$(C=[N,O,S])])[CX3;$(C[#6]),$([CH])]=[CX3]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[ClX1]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[FX1]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[BrX1]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[IX1]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[FX1,ClX1,BrX1,IX1]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[$([OX2H]),$([OX1-])]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[OX2][#6;!$(C=[O,N,S])]",
            "[#6][#6X3R](=[OX1])[#8X2][#6;!$(C=[O,N,S])]",
            "[CX3;$([H0][#6]),$([H1])](=[OX1])[#8X2][CX3;$([H0][#6]),$([H1])](=[OX1])",
            "[$([#6X3H0][#6]),$([#6X3H])](=[!#6])[!#6]",
            "[CX3;!R;$([C][#6]),$([CH]);$([C](=[OX1])[$([SX2H]),$([SX1-])]),$([C](=[SX1])[$([OX2H]),$([OX1-])])]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[SX2][#6;!$(C=[O,N,S])]",
            "[#6][#6X3R](=[OX1])[#16X2][#6;!$(C=[O,N,S])]",
            "[CX3;$([H0][#6]),$([H1])](=[SX1])[OX2][#6;!$(C=[O,N,S])]",
            "[#6][#6X3R](=[SX1])[#8X2][#6;!$(C=[O,N,S])]",
            "[CX3;$([H0][#6]),$([H1])](=[SX1])[FX1,ClX1,BrX1,IX1]",
            "[CX3;!R;$([C][#6]),$([CH]);$([C](=[SX1])[SX2H])]",
            "[CX3;!R;$([C][#6]),$([CH]);$([C](=[SX1])[SX2][#6;!$(C=[O,N,S])])]",
            "[#6][#6X3R](=[SX1])[#16X2][#6;!$(C=[O,N,S])]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[NX3H2]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3H1][#6;!$(C=[O,N,S])]",
            "[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3H0]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])]",
            "[#6R][#6X3R](=[OX1])[#7X3;$([H1][#6;!$(C=[O,N,S])]),$([H0]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[#6X3;$([H0][#6]),$([H1])](=[OX1])[#7X3H0]([#6])[#6X3;$([H0][#6]),$([H1])](=[OX1])",
            "[#6X3;$([H0][#6]),$([H1])](=[OX1])[#7X3H0]([!#6])[#6X3;$([H0][#6]),$([H1])](=[OX1])",
            "[#6X3;$([H0][#6]),$([H1])](=[OX1])[#7X3H1][#6X3;$([H0][#6]),$([H1])](=[OX1])",
            "[$([CX3;!R][#6]),$([CX3H;!R])](=[SX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[#6R][#6X3R](=[SX1])[#7X3;$([H1][#6;!$(C=[O,N,S])]),$([H0]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[#6X3;$([H0][#6]),$([H1])](=[OX1])[#8X2][#7X2]=,:[#6X3;$([H0]([#6])[#6]),$([H1][#6]),$([H2])]",
            "[NX3;!$(NC=[O,S])][CX3;$([CH]),$([C][#6])]=[NX2;!$(NC=[O,S])]",
            "[CX3;$([H0][#6]),$([H1])](=[OX1])[#7X3;$([H1]),$([H0][#6;!$(C=[O,N,S])])][$([OX2H]),$([OX1-])]",
            "[CX3;$([H0][#6]),$([H1])](=[OX1])[#7X3;$([H1]),$([H0][#6;!$(C=[O,N,S])])][OX2][#6;!$(C=[O,N,S])]",
            "[CX3R0;$([H0][#6]),$([H1])](=[NX2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[$([OX2H]),$([OX1-])]",
            "[#6R][#6X3R](=,:[#7X2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[$([OX2H]),$([OX1-])]",
            "[CX3R0;$([H0][#6]),$([H1])](=[NX2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[OX2][#6;!$(C=[O,N,S])]",
            "[#6R][#6X3R](=,:[#7X2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[OX2][#6;!$(C=[O,N,S])]",
            "[CX3R0;$([H0][#6]),$([H1])](=[NX2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[$([SX2H]),$([SX1-])]",
            "[#6R][#6X3R](=,:[#7X2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[$([SX2H]),$([SX1-])]",
            "[CX3R0;$([H0][#6]),$([H1])](=[NX2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[SX2][#6;!$(C=[O,N,S])]",
            "[#6R][#6X3R](=,:[#7X2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[SX2][#6;!$(C=[O,N,S])]",
            "[#7X3v3;!$(N([#6X3]=[#7X2])C=[O,S])][CX3R0;$([H1]),$([H0][#6])]=[NX2v3;!$(N(=[#6X3][#7X3])C=[O,S])]",
            "[#6][#6X3R;$([H0](=[NX2;!$(N(=[#6X3][#7X3])C=[O,S])])[#7X3;!$(N([#6X3]=[#7X2])C=[O,S])]),$([H0](-[NX3;!$(N([#6X3]=[#7X2])C=[O,S])])=,:[#7X2;!$(N(=[#6X3][#7X3])C=[O,S])])]",
            "[CX3R0;$([H0][#6]),$([H1])](=[NX2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[FX1,ClX1,BrX1,IX1]",
            "[#6R][#6X3R](=,:[#7X2;$([H1]),$([H0][#6;!$(C=[O,N,S])])])[FX1,ClX1,BrX1,IX1]",
            "[$([$([#6X3][#6]),$([#6X3H])](=[#7X2v3])[#7X3v3][#7X3v3]),$([$([#6X3][#6]),$([#6X3H])]([#7X3v3])=[#7X2v3][#7X3v3])]",
            "[NX3,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])][C][CX3](=[OX1])[OX2H,OX1-]",
            "[OX2H][C][CX3](=[OX1])[OX2H,OX1-]",
            "[NX3;$([N][CX3](=[OX1])[C][NX3,NX4+])][C][CX3](=[OX1])[NX3;$([N][C][CX3](=[OX1])[NX3,OX2,OX1-])]",
            "[NX3;$([N][CX3](=[OX1])[C][NX3,NX4+])][C][CX3](=[OX1])[OX2H,OX1-]",
            "[NX3,NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])][C][CX3](=[OX1])[NX3;$([N][C][CX3](=[OX1])[NX3,OX2,OX1-])]",
            "[#6][OX2][CX4;$(C[#6]),$([CH])]([OX2][#6])[OX2][#6]",
            "[CX3]=[CX2]=[OX1]",
            "[#7X2,#8X3,#16X2;$(*[#6,#14])][#6X3]([#7X2,#8X3,#16X2;$(*[#6,#14])])=[#6X3]",
            "[NX1]#[CX2]",
            "[CX1-]#[NX2+]",
            "[#6X3](=[OX1])[#6X3]=,:[#6X3][#7,#8,#16,F,Cl,Br,I]",
            "[#6X3](=[OX1])[#6X3]=,:[#6X3][$([OX2H]),$([OX1-])]",
            "[#6X3](=[OX1])[#6X3]=,:[#6X3][#6;!$(C=[O,N,S])]",
            "[#6X3](=[OX1])[#6X3]=,:[#6X3][#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[#6X3](=[OX1])[#6X3]=,:[#6X3][FX1,ClX1,BrX1,IX1]",
            "[#6;!$(C=[O,N,S])][#8X2][#6X3](=[OX1])[#8X2][#6;!$(C=[O,N,S])]",
            "[#6;!$(C=[O,N,S])][OX2;!R][CX3](=[OX1])[OX2][FX1,ClX1,BrX1,IX1]",
            "[#6;!$(C=[O,N,S])][OX2;!R][CX3](=[OX1])[$([OX2H]),$([OX1-])]",
            "[!#6][#6X3](=[!#6])[!#6]",
            "[#6;!$(C=[O,N,S])][#8X2][#6X3](=[SX1])[#8X2][#6;!$(C=[O,N,S])]",
            "[#6;!$(C=[O,N,S])][OX2;!R][CX3](=[SX1])[OX2][FX1,ClX1,BrX1,IX1]",
            "[#6;!$(C=[O,N,S])][OX2;!R][CX3](=[SX1])[$([OX2H]),$([OX1-])]",
            "[#7X3;!$([#7][!#6])][#6X3](=[OX1])[#7X3;!$([#7][!#6])]",
            "[#7X3;!$([#7][!#6])][#6X3](=[SX1])[#7X3;!$([#7][!#6])]",
            "[#7X2;!$([#7][!#6])]=,:[#6X3]([#8X2&!$([#8][!#6]),OX1-])[#7X3;!$([#7][!#6])]",
            "[#7X2;!$([#7][!#6])]=,:[#6X3]([#16X2&!$([#16][!#6]),SX1-])[#7X3;!$([#7][!#6])]",
            "[N;v3X3,v4X4+][CX3](=[N;v3X2,v4X3+])[N;v3X3,v4X4+]",
            "[NX3]C(=[OX1])[O;X2H,X1-]",
            "[#7X3][#6](=[OX1])[#8X2][#6]",
            "[#7X3][#6](=[OX1])[#7X3][#6](=[OX1])[#7X3]",
            "[#7X3][#7X3][#6X3]([#7X3;!$([#7][#7])])=[OX1]",
            "[#7X3][#7X3][#6X3]([#7X3][#7X3])=[OX1]",
            "[#7X2](=[#6])[#7X3][#6X3]([#7X3;!$([#7][#7])])=[OX1]",
            "[#7X2](=[#6])[#7X3][#6X3]([#7X3][#7X3])=[OX1]",
            "[#7X3][#7X3][#6X3]([#7X3;!$([#7][#7])])=[SX1]",
            "[#7X3][#7X3][#6X3]([#7X3][#7X3])=[SX1]",
            "[#7X2](=[#6])[#7X3][#6X3]([#7X3;!$([#7][#7])])=[SX1]",
            "[#7X2](=[#6])[#7X3][#6X3]([#7X3][#7X3])=[SX1]",
            "[NX2]=[CX2]=[OX1]",
            "[OX2][CX2]#[NX1]",
            "[NX2]=[CX2]=[SX1]",
            "[SX2][CX2]#[NX1]",
            "[NX2]=[CX2]=[NX2]",
            "[CX4H0]([O,S,#7])([O,S,#7])([O,S,#7])[O,S,#7,F,Cl,Br,I]",
            "[OX2H][c]",
            "[OX2H][c][c][OX2H]",
            "[Cl][c]",
            "[F][c]",
            "[Br][c]",
            "[I][c]",
            "[SX2H][c]",
            "[c]=[NX2;$([H1]),$([H0][#6;!$([C]=[N,S,O])])]",
            "[c]=[OX1]",
            "[c]=[SX1]",
            "[nX3H1+0]",
            "[nX3H0+0]",
            "[nX2,nX3+]",
            "[o]",
            "[sX2]",
            "[a;!c]",
            "[NX2](=[OX1])[O;$([X2]),$([X1-])]",
            "[SX2][NX2]=[OX1]",
            "[$([NX3](=[OX1])(=[OX1])[O;$([X2]),$([X1-])]),$([NX3+]([OX1-])(=[OX1])[O;$([X2]),$([X1-])])]",
            "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
            "[NX2](=[OX1])[!#7;!#8]",
            "[NX1]~[NX2]~[NX2,NX1]",
            "[CX3](=[OX1])[NX2]~[NX2]~[NX1]",
            "[$([#6]=[NX2+]=[NX1-]),$([#6-]-[NX2+]#[NX1])]",
            "[#6][NX2+]#[NX1]",
            "[#7;!$(N*=O)][NX2]=[OX1]",
            "[NX2](=[OX1])N-*=O",
            "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
            "[NX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6]);!$(NC=[O,N,S])][NX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6]);!$(NC=[O,N,S])]",
            "[NX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6]);!$(NC=[O,N,S])][NX2]=[#6]",
            "[NX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6]);!$(NC=[O,N,S])][OX2;$([H1]),$(O[#6;!$(C=[N,O,S])])]",
            "[$([SX4](=[OX1])(=[OX1])([#6])[#6]),$([SX4+2]([OX1-])([OX1-])([#6])[#6])]",
            "[$([SX3](=[OX1])([#6])[#6]),$([SX3+]([OX1-])([#6])[#6])]",
            "[S+;!$([S]~[!#6]);!$([S]*~[#7,#8,#15,#16])]",
            "[SX4](=[OX1])(=[OX1])([$([OX2H]),$([OX1-])])[$([OX2H]),$([OX1-])]",
            "[SX4](=[OX1])(=[OX1])([$([OX2H]),$([OX1-])])[OX2][#6;!$(C=[O,N,S])]",
            "[SX4](=[OX1])(=[OX1])([OX2][#6;!$(C=[O,N,S])])[OX2][#6;!$(C=[O,N,S])]",
            "[SX4](=[OX1])(=[OX1])([#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])])[$([OX2H]),$([OX1-])]",
            "[SX4](=[OX1])(=[OX1])([#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[SX4](=[OX1])(=[OX1])([#7X3][#6;!$(C=[O,N,S])])[OX2][#6;!$(C=[O,N,S])]",
            "[SX4D4](=[!#6])(=[!#6])([!#6])[!#6]",
            "[SX4;$([H1]),$([H0][#6])](=[OX1])(=[OX1])[$([OX2H]),$([OX1-])]",
            "[SX4;$([H1]),$([H0][#6])](=[OX1])(=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[SX4;$([H1]),$([H0][#6])](=[OX1])(=[OX1])[OX2][#6;!$(C=[O,N,S])]",
            "[SX4;$([H1]),$([H0][#6])](=[OX1])(=[OX1])[FX1,ClX1,BrX1,IX1]",
            "[SX4;$([H1]),$([H0][#6])](=[!#6])(=[!#6])[!#6]",
            "[SX3;$([H1]),$([H0][#6])](=[OX1])[$([OX2H]),$([OX1-])]",
            "[SX3;$([H1]),$([H0][#6])](=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[SX3;$([H1]),$([H0][#6])](=[OX1])[OX2][#6;!$(C=[O,N,S])]",
            "[SX3;$([H1]),$([H0][#6])](=[OX1])[FX1,ClX1,BrX1,IX1]",
            "[SX3;$([H1]),$([H0][#6])](=[!#6])[!#6]",
            "[SX2;$([H1]),$([H0][#6])][$([OX2H]),$([OX1-])]",
            "[SX2;$([H1]),$([H0][#6])][#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[SX2;$([H1]),$([H0][#6])][OX2][#6;!$(C=[O,N,S])]",
            "[SX2;$([H1]),$([H0][#6])][FX1,ClX1,BrX1,IX1]",
            "[SX2;$([H1]),$([H0][#6])][!#6]",
            "[PX3;$([H3]),$([H2][#6]),$([H1]([#6])[#6]),$([H0]([#6])([#6])[#6])]",
            "[PX4;$([H3]=[OX1]),$([H2](=[OX1])[#6]),$([H1](=[OX1])([#6])[#6]),$([H0](=[OX1])([#6])([#6])[#6])]",
            "[P+;!$([P]~[!#6]);!$([P]*~[#7,#8,#15,#16])]",
            "[PX4;$([H3]=[CX3]),$([H2](=[CX3])[#6]),$([H1](=[CX3])([#6])[#6]),$([H0](=[CX3])([#6])([#6])[#6])]",
            "[PX4;$([H1]),$([H0][#6])](=[OX1])([$([OX2H]),$([OX1-])])[$([OX2H]),$([OX1-])]",
            "[PX4;$([H1]),$([H0][#6])](=[OX1])([$([OX2H]),$([OX1-])])[OX2][#6;!$(C=[O,N,S])]",
            "[PX4;$([H1]),$([H0][#6])](=[OX1])([OX2][#6;!$(C=[O,N,S])])[OX2][#6;!$(C=[O,N,S])]",
            "[PX4;$([H1]),$([H0][#6])](=[OX1])([$([OX2H]),$([OX1-])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4;$([H1]),$([H0][#6])](=[OX1])([#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4;$([H1]),$([H0][#6])](=[OX1])([OX2][#6;!$(C=[O,N,S])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4;$([H1]),$([H0][#6])](=[!#6])([!#6])[!#6]",
            "[PX4D4](=[OX1])([$([OX2H]),$([OX1-])])([$([OX2H]),$([OX1-])])[$([OX2H]),$([OX1-])]",
            "[PX4D4](=[OX1])([$([OX2H]),$([OX1-])])([$([OX2H]),$([OX1-])])[OX2][#6;!$(C=[O,N,S])]",
            "[PX4D4](=[OX1])([$([OX2H]),$([OX1-])])([OX2][#6;!$(C=[O,N,S])])[OX2][#6;!$(C=[O,N,S])]",
            "[PX4D4](=[OX1])([OX2][#6;!$(C=[O,N,S])])([OX2][#6;!$(C=[O,N,S])])[OX2][#6;!$(C=[O,N,S])]",
            "[PX4D4](=[OX1])([$([OX2H]),$([OX1-])])([$([OX2H]),$([OX1-])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4D4](=[OX1])([$([OX2H]),$([OX1-])])([#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4D4](=[OX1])([#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])])([#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4D4](=[OX1])([$([OX2H]),$([OX1-])])([OX2][#6;!$(C=[O,N,S])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4D4](=[OX1])([OX2][#6;!$(C=[O,N,S])])([OX2][#6;!$(C=[O,N,S])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4D4](=[OX1])([OX2][#6;!$(C=[O,N,S])])([#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4D4](=[!#6])([!#6])([!#6])[!#6]",
            "[PX4;$([H2]),$([H1][#6]),$([H0]([#6])[#6])](=[OX1])[$([OX2H]),$([OX1-])]",
            "[PX4;$([H2]),$([H1][#6]),$([H0]([#6])[#6])](=[OX1])[OX2][#6;!$(C=[O,N,S])]",
            "[PX4;$([H2]),$([H1][#6]),$([H0]([#6])[#6])](=[OX1])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX4;$([H2]),$([H1][#6]),$([H0]([#6])[#6])](=[!#6])[!#6]",
            "[PX3;$([H1]),$([H0][#6])]([$([OX2H]),$([OX1-])])[$([OX2H]),$([OX1-])]",
            "[PX3;$([H1]),$([H0][#6])]([$([OX2H]),$([OX1-])])[OX2][#6;!$(C=[O,N,S])]",
            "[PX3;$([H1]),$([H0][#6])]([OX2][#6;!$(C=[O,N,S])])[OX2][#6;!$(C=[O,N,S])]",
            "[PX3;$([H1]),$([H0][#6])]([$([OX2H]),$([OX1-])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX3;$([H1]),$([H0][#6])]([#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX3;$([H1]),$([H0][#6])]([OX2][#6;!$(C=[O,N,S])])[#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX3;$([D2]),$([D3][#6])]([!#6])[!#6]",
            "[PX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6])][$([OX2H]),$([OX1-])]",
            "[PX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6])][OX2][#6;!$(C=[O,N,S])]",
            "[PX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6])][#7X3;$([H2]),$([H1][#6;!$(C=[O,N,S])]),$([#7]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])])]",
            "[PX3;$([H2]),$([H1][#6]),$([H0]([#6])[#6])][!#6]",
            "[SiX4]([#6])([#6])([#6])[#6]",
            "[SiX4;$([H1]([#6])([#6])[#6]),$([H2]([#6])[#6]),$([H3][#6]),$([H4])]",
            "[SiX4]([FX1,ClX1,BrX1,IX1])([#6])([#6])[#6]",
            "[SiX4]([!#6])([#6])([#6])[#6]",
            "[SiX4]([!#6])([!#6])([#6])[#6]",
            "[SiX4]([!#6])([!#6])([!#6])[#6]",
            "[SiX4]([!#6])([!#6])([!#6])[!#6]",
            "[BX3]([#6])([#6])[#6]",
            "[BX3]([!#6])([!#6])[!#6]",
            "[BX3]([!#6])([!#6])[!#6]",
            "[BH1,BH2,BH3,BH4]",
            "[BX4]",
            "a",
            "[!#6;!R0]",
            "[OX2r3]1[#6r3][#6r3]1",
            "[NX3H1r3]1[#6r3][#6r3]1",
            "[D4R;$(*(@*)(@*)(@*)@*)]",
            "[R;$(*(@*)(@*)@*);!$([R2;$(*(@*)(@*)(@*)@*)])]@[R;$(*(@*)(@*)@*);!$([R2;$(*(@*)(@*)(@*)@*)])]",
            "[R;$(*(@*)(@*)@*);!$([D4R;$(*(@*)(@*)(@*)@*)]);!$([R;$(*(@*)(@*)@*);!$([R2;$(*(@*)(@*)(@*)@*)])]@[R;$(*(@*)(@*)@*);!$([R2;$(*(@*)(@*)(@*)@*)])])]",
            "[OX2;$([r5]1@C@C@C(O)@C1),$([r6]1@C@C@C(O)@C(O)@C1)]",
            "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
            "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C(O)@C1)]",
            "[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),$([r6]1@C(!@[OX2H1])@C@C@C@C1)]",
            "[OX2;$([r5]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
            "[OX2;$([r5]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
            "*=*[*]=,#,:[*]",
            "*#*[*]=,#,:[*]",
            r"*/[D2]=[D2]\*",
            "*/[D2]=[D2]/*",
            "[$(*=O),$([#16,#14,#5]),$([#7]([#6]=[OX1]))][#8X2][$(*=O),$([#16,#14,#5]),$([#7]([#6]=[OX1]))]",
            "[FX1,ClX1,BrX1,IX1][!#6]",
            "[F,Cl,Br,I;!$([X1]);!$([X0-])]",
            "[FX1][CX4;!$([H0][Cl,Br,I]);!$([F][C]([F])([F])[F])]([FX1])([FX1])",
            "[#6]~[#7,#8,#16]",
            "[!+0]",
            "[-1,-2,-3,-4,-5,-6,-7]",
            "[+1,+2,+3,+4,+5,+6,+7]",
            "*",  # temporarily replaces "([-1,-2,-3,-4,-5,-6,-7]).([+1,+2,+3,+4,+5,+6,+7])",
            "[$([#7X2,OX1,SX1]=*[!H0;!$([a;!n])]),$([#7X3,OX2,SX2;!H0]*=*),$([#7X3,OX2,SX2;!H0]*:n)]",
            "[$([#7X2,OX1,SX1]=,:**=,:*[!H0;!$([a;!n])]),$([#7X3,OX2,SX2;!H0]*=**=*),$([#7X3,OX2,SX2;!H0]*=,:**:n)]",
            "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]",
            "[CX3]=[CX3][$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-])]",
            "[CX3](=[OX1])[NX2]=[NX2][CX3](=[OX1])",
            "[$([CX4;!$([H0]);!$(C[!#6;!$([P,S]=O);!$(N(~O)~O)])][$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-]);!$(*[S,O,N;H1,H2]);!$([*+0][S,O;X1-])]),$([CX4;!$([H0])]1[CX3]=[CX3][CX3]=[CX3]1)]",
            "[CX4;!$([H0]);!$(C[!#6;!$([P,S]=O);!$(N(~O)~O)])]([$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-]);!$(*[S,O,N;H1,H2]);!$([*+0][S,O;X1-])])[$([CX3]=[O,N,S]),$(C#[N]),$([S,P]=[OX1]),$([NX3]=O),$([NX3+](=O)[O-]);!$(*[S,O,N;H1,H2]);!$([*+0][S,O;X1-])]",
            "[$([*@](~*)(~*)(*)*),$([*@H](*)(*)*),$([*@](~*)(*)*),$([*@H](~*)~*)]",
        ]
        # flake8: noqa
        super().__init__(
            patterns=patterns,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        """
        Compute Laggner fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit Mol objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 307)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        # temporarily use dense array, setting bits on CSR array is slow
        sparse = self.sparse
        try:
            self.sparse = False
            fps = super()._calculate_fingerprint(X)
        finally:
            self.sparse = sparse

        smiles_list = ensure_smiles(X)

        for idx, smiles in enumerate(smiles_list):
            # salt = at least two components, has anion and cation
            # this can only be 0 or 1
            multi_component = "." in smiles
            anion = fps[idx, 297]
            cation = fps[idx, 298]
            salt = multi_component & anion & cation
            fps[idx, 299] = salt

        return csr_array(fps) if self.sparse else fps
