import numpy as np
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from skfp.datasets.moleculenet import load_ogb_splits, load_hiv, load_moleculenet_benchmark
from skfp.filters import LipinskiFilter, BeyondRo5Filter
from skfp.preprocessing import MolFromSmilesTransformer

datasets = load_moleculenet_benchmark()

def get_data_and_labels_at(data, labels, indexes):

    split_data = [data[i] for i in indexes]
    split_labels = [labels[i] for i in indexes]

    return split_data, split_labels

def filter_x_and_y(data, labels, filter):
    filtered_data, filtered_labels = zip(*[(x, labels[i]) for i, x in enumerate(data) if filter.transform([x])])
    return filtered_data, filtered_labels

for dataset in datasets:
    dataset_name, data, labels = dataset
    print(dataset_name)
    train_idx, valid_idx, test_idx = load_ogb_splits(dataset_name)

    X_train, y_train = get_data_and_labels_at(data, labels, train_idx)
    X_valid, y_valid = get_data_and_labels_at(data, labels, valid_idx)
    X_test, y_test = get_data_and_labels_at(data, labels, test_idx)

    curr_filter = LipinskiFilter()
    # filtered_data = curr_filter.transform(X_train)
    X_train_filtered, y_train_filtered = filter_x_and_y(X_train, y_train, curr_filter)
    X_test_filtered, y_test_filtered = filter_x_and_y(X_test, y_test, curr_filter)
    X_valid_filtered, y_valid_filtered = filter_x_and_y(X_valid, y_valid, curr_filter)

    print(len(X_train))
    print(len(X_train_filtered))
    print(len(y_train_filtered))

    mol_from_smiles = MolFromSmilesTransformer()

    mols = mol_from_smiles.transform(X_train_filtered)



    break

