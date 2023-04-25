import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit

from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from tdc.single_pred import Tox
from rdkit.Chem import AllChem
from rdkit import Chem


def example_ogbg(dataset_name: str):
    assert dataset_name in ["ogbg-molhiv", "ogbg-molpcba"]

    # Prepare dataset
    PygGraphPropPredDataset(name=dataset_name, root="./datasets")

    dataset = pd.read_csv(
        f"./datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]

    if dataset_name == "ogbg-molhiv":
        y = dataset["HIV_active"]
    else:
        # molpcba is multilabel
        dataset.pop("smiles")
        dataset.pop("mol_id")
        y = dataset
        y[np.isnan(y)] = 0

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True
    )

    # Convert SMILES to RDKit molecules
    X_train = [Chem.MolFromSmiles(x) for x in X_train]
    X_test = [Chem.MolFromSmiles(x) for x in X_test]

    # Compute fingerprint for each molecule
    X_train = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in X_train]
    X_test = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in X_test]

    # To numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if dataset_name == "ogbg-molhiv":
        # Train a random forest classifier
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
    else:
        # Train a multi-output random forest classifier
        clf = MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1)
        clf.fit(X_train, y_train)

    if dataset_name == "ogbg-molhiv":
        # For ogbg-molhiv auroc is used
        print(
            f"AUROC: {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])}"
        )
    else:
        # For ogbg-molpcba ap is used
        print(
            f"AP: {average_precision_score(y_test, clf.predict_proba(X_test)[:, 1])}"
        )


def example_tdc():
    dataset = Tox(name="hERG")
    split = dataset.get_split()
    train, valid, test = split["train"], split["valid"], split["tests"]

    y_train = train["Y"]
    X_train = train["Drug"]

    # class distribution is about 2/3 - 1 and 1/3 - 0
    # y_train.value_counts().plot.bar()
    # plt.show()

    y_valid = valid["Y"]
    X_valid = valid["Drug"]

    # We can merge train and valid parts, because we won't need the valid one.
    y_train = pd.concat([y_train, y_valid])
    X_train = pd.concat([X_train, X_valid])

    y_test = test["Y"]
    X_test = test["Drug"]

    X_train = [Chem.MolFromSmiles(x) for x in X_train]
    X_test = [Chem.MolFromSmiles(x) for x in X_test]

    # Compute fingerprint for each molecule
    X_train = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in X_train]
    X_test = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in X_test]

    # To numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print(
        f"AP: {average_precision_score(y_test, clf.predict_proba(X_test)[:, 1])}"
    )


if __name__ == "__main__":
    """
    First example uses datasets from ogb.stanford.edu.
    Second one from tdcommons.ai.

    ogbg-molpcba (it's a bin. class., multilabel example) should work fine, but took some time on my computer,
    so I haven't finished that run.

    Uncomment below to make it work. (ogbg has 2 options for dataset: ogbg-molhiv and ogbg-molpcba)
    """

    # example_ogbg('ogbg-molhiv')
    example_tdc()
