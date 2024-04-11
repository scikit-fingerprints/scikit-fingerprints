import inspect
import os
from time import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from ogb.graphproppred import GraphPropPredDataset
from rdkit.Chem import MolFromSmiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import skfp.fingerprints
from skfp.preprocessing import ConformerGenerator

fingerprint_classes = [
    cls
    for name, cls in inspect.getmembers(skfp.fingerprints, predicate=inspect.isclass)
]

SIZE = -1

SCRIPT_PATH = os.path.abspath(__file__)
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "dataset")


dataset_params = [
    ("ogbg-molhiv", "HIV_active"),
    ("ogbg-molbace", "Class"),
    ("ogbg-molbbbp", "p_np"),
]


classifier_parameters = [
    (RandomForestClassifier, {"n_jobs": -1, "class_weight": "balanced"}),
    (
        LogisticRegressionCV,
        {"class_weight": "balanced", "max_iter": 150, "n_jobs": -1},
    ),
    (lgb.LGBMClassifier, {"n_jobs": -1, "class_weight": "balanced", "verbose": -1}),
]

for dataset_name, property_name in dataset_params:
    dataset = GraphPropPredDataset(name=dataset_name, root=DATASET_DIR)
    split_idx = dataset.get_idx_split()
    train_idx = np.array(split_idx["train"])[:SIZE]
    valid_idx = np.array(split_idx["valid"])[:SIZE]
    test_idx = np.array(split_idx["test"])[:SIZE]

    dataframe = pd.read_csv(
        f"{DATASET_DIR}/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )

    print(dataframe.columns)

    X = dataframe["smiles"]
    y = dataframe[property_name]

    n_molecules = X.shape[0]
    print("Number of molecules:", n_molecules)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    X_train_mols = [MolFromSmiles(smiles) for smiles in X_train]
    X_valid_mols = [MolFromSmiles(smiles) for smiles in X_valid]
    X_test_mols = [MolFromSmiles(smiles) for smiles in X_test]

    records = []

    np.random.seed(42)

    for fingerprint in fingerprint_classes:
        fp_name = fingerprint.__name__
        records.append({})
        records[-1]["fp_name"] = fp_name
        print(fp_name)
        start = time()
        fp_transformer = fingerprint(n_jobs=-1)

        try:
            X_fp_train = fp_transformer.fit_transform(X_train)
            X_fp_valid = fp_transformer.transform(X_valid)
            X_fp_test = fp_transformer.transform(X_test)
        except RuntimeError as e:
            print(f" - Error: {e}")
            records[-1]["error"] = str(e)
            continue
        except ValueError as e:
            try:
                conf_gen = ConformerGenerator()
                X_train_conf = conf_gen.transform(X_train_mols)
                X_valid_conf = conf_gen.transform(X_valid_mols)
                X_test_conf = conf_gen.transform(X_test_mols)

                X_fp_train = fp_transformer.fit_transform(X_train_conf)
                X_fp_valid = fp_transformer.transform(X_valid_conf)
                X_fp_test = fp_transformer.transform(X_test_conf)
            except ValueError as e:
                print(e.__class__.__name__)
                print(f" - Error: {e}")
                records[-1]["error"] = str(e)
                continue

        scaler = MinMaxScaler()
        X_fp_train = scaler.fit_transform(X_fp_train)
        X_fp_valid = scaler.fit_transform(X_fp_valid)
        X_fp_test = scaler.transform(X_fp_test)

        end = time()
        execution_time = end - start
        print(f" - Time of fingerprints computing : {round(execution_time,2)}s")
        records[-1]["execution_time"] = execution_time

        imputer = SimpleImputer(strategy="constant", fill_value=0)
        X_fp_train = imputer.fit_transform(X_fp_train)
        X_fp_valid = imputer.transform(X_fp_valid)
        X_fp_test = imputer.transform(X_fp_test)

        for classifier, clf_kwargs in classifier_parameters:
            clf_name = classifier.__name__
            scores = []
            scores_valid = []
            try:
                for epoch in range(10):
                    clf = classifier(random_state=epoch, **clf_kwargs)
                    clf.fit(X_fp_train, y_train)
                    scores.append(
                        roc_auc_score(y_test, clf.predict_proba(X_fp_test)[:, 1])
                    )
                    scores_valid.append(
                        roc_auc_score(y_valid, clf.predict_proba(X_fp_valid)[:, 1])
                    )
            except ValueError as e:
                print(f" - Error: {e}")
                records[-1][clf_name + "_error"] = str(e)
                continue
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
