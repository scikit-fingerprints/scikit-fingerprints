import numpy as np
import pandas as pd
import rdkit

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem, PandasTools
from rdkit import Chem


if __name__ == "__main__":
    dataset = pd.read_csv("./datasets/ogbg_molhiv/mapping/mol.csv.gz")
    X = dataset["smiles"]
    y = dataset["HIV_active"]

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

    # Train a random forest classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # For ogbg-molhiv auroc is used
    print(f"AUROC: {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])}")
