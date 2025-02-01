import inspect
import os
from time import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import skfp.fingerprints as fps
from skfp.datasets.moleculenet import load_moleculenet_benchmark, load_ogb_splits
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

fingerprint_classes = [
    cls for name, cls in inspect.getmembers(fps, predicate=inspect.isclass)
]

descriptor_fingerprints = [
    fps.AutocorrFingerprint,
    fps.BCUT2DFingerprint,
    fps.ElectroShapeFingerprint,
    fps.EStateFingerprint,
    fps.GETAWAYFingerprint,
    fps.MordredFingerprint,
    fps.MORSEFingerprint,
    fps.MQNsFingerprint,
    fps.RDFFingerprint,
    fps.RDKit2DDescriptorsFingerprint,
    fps.USRFingerprint,
    fps.USRCATFingerprint,
    fps.VSAFingerprint,
    fps.WHIMFingerprint,
]

LIMIT_SIZE = None

SCRIPT_PATH = os.path.abspath(__file__)


classifier_parameters = [
    (RandomForestClassifier, {"n_jobs": -1, "class_weight": "balanced"}),
    (
        LogisticRegressionCV,
        {"class_weight": "balanced", "max_iter": 150, "n_jobs": -1},
    ),
    (lgb.LGBMClassifier, {"n_jobs": -1, "class_weight": "balanced", "verbose": -1}),
]


# BACE, BBBP, HIV
for dataset_name, smiles_list, y in load_moleculenet_benchmark(
    "classification_single_task"
):
    smiles_list = np.array(smiles_list)

    train_idxs, valid_idxs, test_idxs = load_ogb_splits(dataset_name)

    if LIMIT_SIZE:
        train_idxs = train_idxs[:LIMIT_SIZE]
        valid_idxs = valid_idxs[:LIMIT_SIZE]
        test_idxs = test_idxs[:LIMIT_SIZE]

    print(f"Number of molecules: {len(smiles_list)}")

    smiles_train = smiles_list[train_idxs]
    smiles_valid = smiles_list[valid_idxs]
    smiles_test = smiles_list[test_idxs]

    y_train = y[train_idxs]
    y_valid = y[valid_idxs]
    y_test = y[test_idxs]

    mol_from_smiles = MolFromSmilesTransformer()
    mols_train = mol_from_smiles.transform(smiles_train)
    mols_valid = mol_from_smiles.transform(smiles_valid)
    mols_test = mol_from_smiles.transform(smiles_test)

    conf_gen = ConformerGenerator(n_jobs=-1, errors="filter")
    mols_train, y_train = conf_gen.transform_x_y(mols_train, y_train)
    mols_valid, y_valid = conf_gen.transform_x_y(mols_valid, y_valid)
    mols_test, y_test = conf_gen.transform_x_y(mols_test, y_test)

    records: list[dict] = []

    np.random.seed(42)

    for fingerprint in fingerprint_classes:
        fp_record = {}

        fp_name = fingerprint.__name__
        fp_record["fp_name"] = fp_name
        print(fp_name)

        start = time()
        fp_transformer = fingerprint(n_jobs=-1)
        X_fp_train = fp_transformer.transform(mols_train)
        X_fp_valid = fp_transformer.transform(mols_valid)
        X_fp_test = fp_transformer.transform(mols_test)
        end = time()

        execution_time = end - start
        print(f" - Time of fingerprints computing : {round(execution_time, 2)}s")
        fp_record["execution_time"] = execution_time

        if fp_transformer in descriptor_fingerprints:
            preproc_pipeline = make_pipeline(
                SimpleImputer(strategy="median"), RobustScaler()
            )
            X_fp_train = preproc_pipeline.fit_transform(X_fp_train)
            X_fp_valid = preproc_pipeline.transform(X_fp_valid)
            X_fp_test = preproc_pipeline.transform(X_fp_test)

        for classifier, clf_kwargs in classifier_parameters:
            clf_name = classifier.__name__
            scores_valid = []
            scores_test = []
            try:
                for random_state in range(10):
                    clf = classifier(random_state=random_state, **clf_kwargs)
                    clf.fit(X_fp_train, y_train)
                    auroc_valid = roc_auc_score(
                        y_valid, clf.predict_proba(X_fp_valid)[:, 1]
                    )
                    auroc_test = roc_auc_score(
                        y_test, clf.predict_proba(X_fp_test)[:, 1]
                    )
                    scores_valid.append(auroc_valid)
                    scores_valid.append(auroc_test)
            except ValueError as e:
                print(f" - Error: {e}")
                fp_record[f"{clf_name}_error"] = str(e)
                continue

            score_valid = np.mean(scores_valid)
            std_valid = np.std(scores_valid)

            score_test = np.mean(scores_test)
            std_test = np.std(scores_test)

            print(f" - - ROC AUC score for {clf_name} : {int(100 * score_test)}%")
            fp_record[f"{clf_name}_valid_mean"] = score_valid
            fp_record[f"{clf_name}_valid_std"] = std_valid
            fp_record[f"{clf_name}_test_mean"] = score_test
            fp_record[f"{clf_name}_test_std"] = std_test

        records.append(fp_record)

    df = pd.DataFrame.from_records(records)

    if os.path.exists(f"classification-scores-{dataset_name}.csv"):
        os.remove(f"classification-scores-{dataset_name}.csv")
    df.to_csv(f"classification-scores-{dataset_name}.csv")
