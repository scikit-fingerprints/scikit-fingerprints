from time import time

import lightgbm as lgb
import pandas as pd
from ogb.graphproppred import GraphPropPredDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from skfp import (
    MHFP,
    AtomPairFingerprint,
    ERGFingerprint,
    MACCSKeysFingerprint,
    MAP4Fingerprint,
    ECFP,
    TopologicalTorsionFingerprint,
)

dataset_name = "ogbg-molhiv"
GraphPropPredDataset(name=dataset_name, root="../dataset")
dataset = pd.read_csv(
    f"../dataset/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
)

X = dataset["smiles"]
y = dataset["HIV_active"]

n_molecules = X.shape[0]
print("Number of molecules:", n_molecules)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

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
classifiers = [RandomForestClassifier, LogisticRegression, lgb.LGBMClassifier]
classifier_kwargs = [
    {"random_state": 42, "n_jobs": -1},
    {
        "random_state": 42,
        "class_weight": "balanced",
        "max_iter": 1000,
        "penalty": None,
        "n_jobs": -1,
    },
    {"random_state": 42, "n_jobs": -1, "verbose": 0},
]


for fingerprint, fp_name in zip(fprints, fp_names):
    records.append({})
    records[-1]["fp_name"] = fp_name
    print(fp_name, "Fingerprint")
    start = time()
    fp_transformer = fingerprint(n_jobs=-1)
    X_fp_train = fp_transformer.transform(X_train)
    X_fp_test = fp_transformer.transform(X_test)

    scaler = MinMaxScaler()
    X_fp_train = scaler.fit_transform(X_fp_train)
    X_fp_test = scaler.transform(X_fp_test)

    end = time()
    execution_time = end - start
    print(f" - Time of fingerprints computing : {round(execution_time,2)}s")
    records[-1]["execution_time"] = execution_time
    for classifier, clf_name, clf_kwargs in zip(
        classifiers, clf_names, classifier_kwargs
    ):
        clf = classifier(**clf_kwargs)
        clf.fit(X_fp_train, y_train)
        score = roc_auc_score(y_test, clf.predict_proba(X_fp_test)[:, 1])
        print(f" - - ROC AUC score for {clf_name} : {int(100*score)}%")
        records[-1][clf_name] = score

df = pd.DataFrame.from_records(records)
df.to_csv("classification_scores.csv")
