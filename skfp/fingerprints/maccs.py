from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol, MolToSmiles
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class MACCSFingerprint(BaseFingerprintTransformer):
    """MACCS fingerprint."""

    def __init__(
        self,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        n_features_out = 159 if count else 166
        super().__init__(
            n_features_out=n_features_out,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

        X = ensure_mols(X)

        if self.count:
            X = [self._get_maccs_patterns_counts(mol) for mol in X]
        else:
            X = [GetMACCSKeysFingerprint(mol) for mol in X]

        dtype = np.uint32 if self.count else np.uint8
        return csr_array(X, dtype=dtype) if self.sparse else np.array(X, dtype=dtype)

    def _get_maccs_patterns_counts(self, mol: Mol) -> list[int]:
        smarts_list = [
            None,  # fragments
            None,  # atomic num >103 - idx 0
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
            "[!#6;!#1]~[!#6;!#1]",  # QQ (&...)  Spec Incomplete
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
            "[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]",  # noqa: E501
            # QHAAACH2A
            "[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]",  # noqa: E501
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
            "[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]",  # noqa: E501
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
