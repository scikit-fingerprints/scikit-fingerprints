import os
from collections.abc import Sequence
from time import time
from typing import Callable

import numpy as np
import pandas as pd
from ogb.graphproppred import GraphPropPredDataset
from rdkit.Chem import Mol

import skfp.fingerprints.base
from skfp.fingerprints import *
from skfp.fingerprints.base import FingerprintTransformer
from skfp.preprocessing import MolFromSmilesTransformer


def compute_and_save_fingerprint(
    X_train: Sequence[Mol], fp_transformer: FingerprintTransformer, file_name: str
):
    """
    Compute a fingerprint and save it
    """

    if os.path.exists(file_name):
        return
    X_train = fp_transformer.transform(X_train)
    with open(file_name, "wb") as f:
        np.save(f, X_train)


def save_dataset_as_fingerprints(
    data: Sequence[str],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    save_dir_path: str,
    fprints: Sequence[Callable],
):
    """
    Store the data processed by all fingerprints(fprints) at save_dir_path
    the indices for scaffold split must be provided
    """

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    X = np.array(MolFromSmilesTransformer().transform(data))

    n_molecules = len(X)
    print("Number of molecules:", n_molecules)

    with open(os.path.join(save_dir_path, "labels_train.npy"), "wb") as f:
        np.save(f, y[train_idx])
    with open(os.path.join(save_dir_path, "labels_valid.npy"), "wb") as f:
        np.save(f, y[valid_idx])
    with open(os.path.join(save_dir_path, "labels_test.npy"), "wb") as f:
        np.save(f, y[test_idx])

    for fingerprint in fprints:
        fp_name = fingerprint.__name__.removesuffix("Fingerprint")
        print(fp_name, "Fingerprint")
        fp_path = os.path.join(save_dir_path, fp_name)

        start = time()

        fp_transformer = fingerprint(n_jobs=-1)
        compute_and_save_fingerprint(
            X[train_idx],
            fp_transformer,
            f"{fp_path}_train.npy",
        )
        compute_and_save_fingerprint(
            X[valid_idx],
            fp_transformer,
            f"{fp_path}_valid.npy",
        )
        compute_and_save_fingerprint(
            X[test_idx],
            fp_transformer,
            f"{fp_path}_test.npy",
        )

        end = time()
        execution_time = end - start
        print(f" - Time of fingerprint computation : {round(execution_time, 2)}s")


def load_fingerprints_from_file(file_path: str) -> np.ndarray:
    """
    Opens saved fingerprints stored at file_path
    Notice that file_path should be provided without the extension,
    so that the function can load both data and labels
    """

    with open(file_path, "rb") as f:
        X = np.load(f)
    return X


def load_dataset_as_fingerprints(dir_path: str) -> Sequence[Sequence[np.ndarray]]:
    fp_names = [file_name.split(sep="_")[0] for file_name in os.listdir(dir_path)]
    fp_names = [file for i, file in enumerate(fp_names) if file not in fp_names[:i]]
    fp_names.remove("labels")

    X_list = []
    for fp_name in fp_names:
        X_train = load_fingerprints_from_file(
            os.path.join(dir_path, f"{fp_name}_train.npy")
        )
        X_valid = load_fingerprints_from_file(
            os.path.join(dir_path, f"{fp_name}_valid.npy")
        )
        X_test = load_fingerprints_from_file(
            os.path.join(dir_path, f"{fp_name}_test.npy")
        )

        X_list.append((X_train, X_valid, X_test))

    return X_list


def load_dataset_labels(dir_path: str) -> Sequence[np.ndarray]:
    with open(os.path.join(dir_path, "labels_train.npy"), "rb") as f:
        y_train = np.load(f)
    with open(os.path.join(dir_path, "labels_valid.npy"), "rb") as f:
        y_valid = np.load(f)
    with open(os.path.join(dir_path, "labels_test.npy"), "rb") as f:
        y_test = np.load(f)

    return (y_train, y_valid, y_test)


if __name__ == "__main__":
    dataset_names = [
        "ogbg-molhiv",
        "ogbg-molbace",
        "ogbg-molbbbp",
    ]

    property_names = [
        "HIV_active",
        "Class",
        "p_np",
    ]

    fingerprints = [
        AtomPairFingerprint,
        AutocorrFingerprint,
        AvalonFingerprint,
        # E3FPFingerprint,
        ECFPFingerprint,
        ERGFingerprint,
        EStateFingerprint,
        # GETAWAYFingerprint,
        LayeredFingerprint,
        MACCSFingerprint,
        MAPFingerprint,
        MHFPFingerprint,
        # MordredFingerprint,
        # MORSEFingerprint,
        PatternFingerprint,
        # PharmacophoreFingerprint,
        PhysiochemicalPropertiesFingerprint,
        PubChemFingerprint,
        # RDFFingerprint,
        RDKitFingerprint,
        SECFPFingerprint,
        TopologicalTorsionFingerprint,
        # WHIMFingerprint,
    ]

    fp_save_dir = "./saved_fingerprints"

    for dataset_name, property_name in zip(dataset_names, property_names):
        dataset = GraphPropPredDataset(name=dataset_name, root="../dataset")
        csv_path = f"../dataset/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
        dataframe = pd.read_csv(csv_path)

        split_idx = dataset.get_idx_split()
        train_idx = np.array(split_idx["train"])
        valid_idx = np.array(split_idx["valid"])
        test_idx = np.array(split_idx["test"])

        X = dataframe["smiles"]
        y = dataframe[property_name]

        save_dataset_as_fingerprints(
            X,
            train_idx,
            valid_idx,
            test_idx,
            os.path.join(fp_save_dir, dataset_name),
            fingerprints,
        )

    # example for each dataset
    for dataset_name, property_name in zip(dataset_names, property_names):
        # load all data into a list of split tuples for each dataset
        # (X_train, X_valid, X_test)
        # also loads a tuple of labels (y_train, y_valid, y_test)
        # additionally returns a list of fingerprint names
        X_list = load_dataset_as_fingerprints(
            os.path.join(fp_save_dir, dataset_name),
        )

        y_tuple = load_dataset_labels(os.path.join(fp_save_dir, dataset_name))
