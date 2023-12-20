import os
from time import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from ogb.graphproppred import GraphPropPredDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from skfp import (
    ECFP,
    MHFP,
    AtomPairFingerprint,
    ERGFingerprint,
    MACCSKeysFingerprint,
    MAP4Fingerprint,
    TopologicalTorsionFingerprint,
)

dataset_names = [
    "ogbg-molhiv",
    "ogbg-molbace",
    "ogbg-bbbp",
]
property_names = [
    "HIV_active",
    "Class",
    "p_np",
]

for dataset_name, property_name in zip(dataset_names, property_names):
    dataset = GraphPropPredDataset(name=dataset_name, root="../dataset")
    split_idx = dataset.get_idx_split()
    train_idx = np.array(split_idx["train"])
    valid_idx = np.array(split_idx["valid"])
    test_idx = np.array(split_idx["test"])

    dataframe = pd.read_csv(
        f"../dataset/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )

    print(dataframe.columns)

    X = dataframe["smiles"]
    y = dataframe[property_name]

    n_molecules = X.shape[0]
    print("Number of molecules:", n_molecules)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    records = []

    fp_names = [
        "ECFP",
        "Atom Pairs",
        "Topological Torsion",
        "MACCS Keys",
        "ERG",
        "MAP4",
        "MHFP",
    ]
    fprints = [
        ECFP,
        AtomPairFingerprint,
        TopologicalTorsionFingerprint,
        MACCSKeysFingerprint,
        ERGFingerprint,
        MAP4Fingerprint,
        MHFP,
    ]

    clf_names = ["RF", "LogReg", "LGBM"]
    classifiers = [
        RandomForestClassifier,
        LogisticRegression,
        lgb.LGBMClassifier,
    ]
    classifier_kwargs = [
        {"n_jobs": -1},
        {
            "class_weight": "balanced",
            "penalty": None,
            "n_jobs": -1,
        },
        {"n_jobs": -1, "verbose": 0},
    ]

    np.random.seed(42)

    for fingerprint, fp_name in zip(fprints, fp_names):
        records.append({})
        records[-1]["fp_name"] = fp_name
        print(fp_name, "Fingerprint")
        start = time()
        fp_transformer = fingerprint(n_jobs=-1)
        X_fp_train = fp_transformer.transform(X_train)
        X_fp_valid = fp_transformer.transform(X_valid)
        X_fp_test = fp_transformer.transform(X_test)

        scaler = MinMaxScaler()
        X_fp_train = scaler.fit_transform(X_fp_train)
        X_fp_valid = scaler.fit_transform(X_fp_valid)
        X_fp_test = scaler.transform(X_fp_test)

        end = time()
        execution_time = end - start
        print(
            f" - Time of fingerprints computing : {round(execution_time,2)}s"
        )
        records[-1]["execution_time"] = execution_time
        for classifier, clf_name, clf_kwargs in zip(
            classifiers, clf_names, classifier_kwargs
        ):
            scores = []
            scores_valid = []
            for epoch in range(10):
                clf = classifier(
                    random_state=np.random.randint(0, 2**31), **clf_kwargs
                )
                clf.fit(X_fp_train, y_train)
                scores.append(
                    roc_auc_score(y_test, clf.predict_proba(X_fp_test)[:, 1])
                )
                scores_valid.append(
                    roc_auc_score(y_valid, clf.predict_proba(X_fp_valid)[:, 1])
                )
            score = np.average(scores)
            std = np.std(scores)
            score_valid = np.average(scores_valid)
            std_valid = np.std(scores_valid)

            print(f" - - ROC AUC score for {clf_name} : {int(100*score)}%")
            records[-1][clf_name + "_mean"] = score
            records[-1][clf_name + "_std"] = std
            records[-1][clf_name + "_valid_mean"] = score_valid
            records[-1][clf_name + "_valid_std"] = std_valid

    df = pd.DataFrame.from_records(records)
    if os.path.exists("classification-scores-" + dataset_name + ".csv"):
        os.remove("classification-scores-" + dataset_name + ".csv")
    df.to_csv("classification-scores-" + dataset_name + ".csv")
