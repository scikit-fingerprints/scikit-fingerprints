import numpy as np
from ogb.graphproppred import GraphPropPredDataset
from rdkit.Chem import Mol
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import skfp.fingerprints as fps
from skfp.bases import BaseFingerprintTransformer
from skfp.datasets.moleculenet import load_moleculenet_benchmark
from skfp.preprocessing import MolFromSmilesTransformer
from skfp.utils import no_rdkit_logs


def fp_name_to_fp(fp_name: str) -> tuple[BaseFingerprintTransformer, dict]:
    if fp_name == "AtomPairs":
        fingerprint = fps.AtomPairFingerprint(n_jobs=-1)
        fp_params_grid = {
            "fp_size": [1024, 2048, 4096],
            "scale_by_hac": [False, True],
            "include_chirality": [False, True],
            "count": [False, True],
        }
    elif fp_name == "Avalon":
        fingerprint = fps.AvalonFingerprint(n_jobs=-1)
        fp_params_grid = {
            "fp_size": [256, 512, 1024, 2048],
            "count": [False, True],
        }
    elif fp_name == "ECFP":
        fingerprint = fps.ECFPFingerprint(n_jobs=-1)
        fp_params_grid = {
            "fp_size": [1024, 2048, 4096],
            "radius": [2, 3],
            "include_chirality": [False, True],
            "count": [False, True],
        }
    elif fp_name == "ERG":
        fingerprint = fps.ERGFingerprint(n_jobs=-1)
        fp_params_grid = {"max_path": list(range(5, 26))}
    elif fp_name == "EState":
        fingerprint = fps.EStateFingerprint(n_jobs=-1)
        fp_params_grid = {"variant": ["sum", "bit", "count"]}
    elif fp_name == "FCFP":
        fingerprint = fps.ECFPFingerprint(use_fcfp=True, n_jobs=-1)
        fp_params_grid = {
            "fp_size": [1024, 2048, 4096],
            "radius": [2, 3],
            "include_chirality": [False, True],
            "count": [False, True],
        }
    elif fp_name == "GhoseCrippen":
        fingerprint = fps.GhoseCrippenFingerprint(n_jobs=-1)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "KlekotaRoth":
        fingerprint = fps.KlekotaRothFingerprint(n_jobs=-1)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "Laggner":
        fingerprint = fps.LaggnerFingerprint(n_jobs=-1)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "Layered":
        fingerprint = fps.LayeredFingerprint(n_jobs=-1)
        fp_params_grid = {
            "fp_size": [1024, 2048, 4096],
            "max_path": [5, 6, 7, 8, 9],
        }
    elif fp_name == "Lingo":
        fingerprint = fps.LingoFingerprint(n_jobs=-1)
        fp_params_grid = {
            "substring_length": [3, 4, 5, 6],
            "count": [False, True],
        }
    elif fp_name == "MACCS":
        fingerprint = fps.MACCSFingerprint(n_jobs=-1)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "MAP":
        fingerprint = fps.MAPFingerprint(n_jobs=-1)
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "radius": [2, 3],
            "variant": ["bit", "count"],
        }
    elif fp_name == "Pattern":
        fingerprint = fps.PatternFingerprint()
        fp_params_grid = {
            "fp_size": [1024, 2048, 4096],
            "tautomers": [False, True],
        }
    elif fp_name == "PhysiochemicalProperties":
        fingerprint = fps.PhysiochemicalPropertiesFingerprint(n_jobs=-1)
        fp_params_grid = {
            "fp_size": [1024, 2048, 4096],
            "variant": ["BP", "BT"],
        }
    elif fp_name == "PubChem":
        fingerprint = fps.PubChemFingerprint(n_jobs=-1)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "RDKit":
        fingerprint = fps.RDKitFingerprint(n_jobs=-1)
        fp_params_grid = {
            "fp_size": [1024, 2048, 4096],
            "max_path": [5, 6, 7, 8, 9],
            "count": [False, True],
        }
    elif fp_name == "SECFP":
        fingerprint = fps.SECFPFingerprint(n_jobs=-1)
        fp_params_grid = {
            "fp_size": [1024, 2048, 4096],
            "radius": [2, 3, 4],
        }
    elif fp_name == "TopologicalTorsion":
        fingerprint = fps.TopologicalTorsionFingerprint(n_jobs=-1)
        fp_params_grid = {
            "fp_size": [1024, 2048, 4096],
            "include_chirality": [False, True],
            "count": [False, True],
        }
    else:
        raise ValueError(f"Fingerprint name '{fp_name}' not recognized")

    return fingerprint, fp_params_grid


def train_and_tune_fp_classifier(
    mols_train: list[Mol],
    mols_test: list[Mol],
    y_train: np.ndarray,
    y_test: np.ndarray,
    fp: BaseFingerprintTransformer,
    fp_params_grid: dict,
) -> tuple[float, float, float]:
    pipeline = Pipeline(
        [("fp", fp), ("clf", RandomForestClassifier(n_jobs=-1, random_state=0))]
    )
    pipeline.fit(mols_train, y_train)

    y_pred_default = pipeline.predict_proba(mols_test)[:, 1]

    grid = {f"fp__{key}": val for key, val in fp_params_grid.items()}
    cv = GridSearchCV(estimator=pipeline, param_grid=grid, scoring="roc_auc")
    cv.fit(mols_train, y_train)

    y_pred_tuned = cv.predict_proba(mols_test)[:, 1]

    auroc_default = roc_auc_score(y_test, y_pred_default)
    auroc_tuned = roc_auc_score(y_test, y_pred_tuned)
    diff = auroc_tuned - auroc_default

    return auroc_default, auroc_tuned, diff


if __name__ == "__main__":
    datasets = load_moleculenet_benchmark(subset="classification_single_task")
    mol_from_smiles = MolFromSmilesTransformer()

    for dataset_name, X, y in datasets:
        print("DATASET", dataset_name)
        X = np.array(X)

        dataset = GraphPropPredDataset(
            name=f"ogbg-mol{dataset_name.lower()}", root=".tmp"
        )
        split_idx = dataset.get_idx_split()

        train_idxs = list(split_idx["train"]) + list(split_idx["valid"])
        test_idxs = list(split_idx["test"])

        smiles_train = X[train_idxs]
        smiles_test = X[test_idxs]

        mols_train = mol_from_smiles.transform(smiles_train)
        mols_test = mol_from_smiles.transform(smiles_test)

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        for fp_name in [
            "AtomPairs",
            "Avalon",
            "ECFP",
            "ERG",
            "EState",
            "FCFP",
            "GhoseCrippen",
            "KlekotaRoth",
            "Laggner",
            "Layered",
            "Lingo",
            "MACCS",
            "MAP",
            "Pattern",
            "PhysiochemicalProperties",
            "PubChem",
            "RDKit",
            "SECFP",
            "TopologicalTorsion",
        ]:
            print(fp_name)
            fp, fp_params_grid = fp_name_to_fp(fp_name)
            with no_rdkit_logs():
                auroc_default, auroc_tuned, diff = train_and_tune_fp_classifier(
                    mols_train=mols_train,
                    mols_test=mols_test,
                    y_train=y_train,
                    y_test=y_test,
                    fp=fp,
                    fp_params_grid=fp_params_grid,
                )
            print(
                f"AUROC default {auroc_default:.1%}, tuned {auroc_tuned:.1%}, diff: {diff:.1%}"
            )
