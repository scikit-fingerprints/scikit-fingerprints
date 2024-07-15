import joblib

import skfp.fingerprints as fps
from skfp.datasets.moleculenet import load_pcba
from skfp.preprocessing import MolFromSmilesTransformer

if __name__ == "__main__":
    X, y = load_pcba()
    mol_from_smiles = MolFromSmilesTransformer()

    mols = mol_from_smiles.transform(X)

    n_cores = joblib.cpu_count(only_physical_cores=True)
    fp_name_to_fp = {
        "AtomPairs": fps.AtomPairFingerprint,
        "Avalon": fps.AvalonFingerprint,
        "ECFP": fps.ECFPFingerprint,
        "ERG": fps.ERGFingerprint,
        "EState": fps.EStateFingerprint,
        "FCFP": fps.ECFPFingerprint,
        "GhoseCrippen": fps.GhoseCrippenFingerprint,
        "KlekotaRoth": fps.KlekotaRothFingerprint,
        "Laggner": fps.LaggnerFingerprint,
        "Layered": fps.LayeredFingerprint,
        "Lingo": fps.LingoFingerprint,
        "MACCS": fps.MACCSFingerprint,
        "MAP": fps.MAPFingerprint,
        "Pattern": fps.PatternFingerprint,
        "PhysiochemicalProperties": fps.PhysiochemicalPropertiesFingerprint,
        "PubChem": fps.PubChemFingerprint,
        "RDKit": fps.RDKitFingerprint,
        "SECFP": fps.SECFPFingerprint,
        "TopologicalTorsion": fps.TopologicalTorsionFingerprint,
    }

    fp_kwargs = {"sparse": True, "n_jobs": joblib.cpu_count(only_physical_cores=True)}

    print("Fingerprint name\tDense size (MB)\tSparse size (MB)\tMemory reduction")
    mb = 1024 * 1024
    for fp_name, fp_cls in fp_name_to_fp.items():
        if fp_name == "FCFP":
            fp = fp_cls(use_fcfp=True, **fp_kwargs)
        else:
            fp = fp_cls(**fp_kwargs)

        X = fp.transform(mols)
        sparse_mem = X.data.nbytes // mb
        dense_mem = X.todense().nbytes // mb
        reduction = round(dense_mem / sparse_mem, 1)
        print(f"{fp_name}\t{dense_mem}\t{sparse_mem}\t{reduction}")
