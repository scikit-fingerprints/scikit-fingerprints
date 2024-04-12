import logging
import os
from time import time

import numpy as np
import pandas as pd
from ogb.graphproppred import GraphPropPredDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from skfp.fingerprints import *
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

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

fp_names = [
    "AtomPair",
    "Autocorr",
    "Avalon",
    # "E3FP", # errors out - won't generate a conformer for one molecule
    "ECFP",
    "ERG",
    "EState",
    # "Getway", # expects conf_id
    "Layered",
    "MACCS-Keys",
    "MAP",
    # "MHFP", # overflow
    # "MORDRED", # overflow
    # "MORSE", # expects conf_id
    "Pattern",
    # "Pharmacophore", # very long
    "Phsiochemical-Properties",
    "PubChem",
    # "RDF", # expects conf_id
    "RDKit",
    "SECFP",
    "Topological-Torsion",
    "Whim",
]

fprints = [
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
    # MHFPFingerprint,
    # MordredFingerprint,
    # MORSEFingerprint,
    PatternFingerprint,
    # PharmacophoreFingerprint,
    PhysiochemicalPropertiesFingerprint,
    PubChemFingerprint,
    RDFFingerprint,
    RDKitFingerprint,
    SECFPFingerprint,
    TopologicalTorsionFingerprint,
    WHIMFingerprint,
]

fps_requiring_conformers = ["Getway"]

save_data_dir = "./saved_fingerprints"
if not os.path.exists(save_data_dir):
    os.mkdir(save_data_dir)

for dataset_name, property_name in zip(dataset_names, property_names):
    dataset = GraphPropPredDataset(name=dataset_name, root="../dataset")
    split_idx = dataset.get_idx_split()
    train_idx = np.array(split_idx["train"])
    valid_idx = np.array(split_idx["valid"])
    test_idx = np.array(split_idx["test"])

    if not os.path.exists("saved_fingerprints/" + dataset_name):
        os.mkdir(save_data_dir + "/" + dataset_name)

    dataframe = pd.read_csv(
        f"../dataset/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )

    print(dataframe.columns)

    X = dataframe["smiles"]
    y = dataframe[property_name]

    # X = MolFromSmilesTransformer().transform(X)

    n_molecules = len(X)
    print("Number of molecules:", n_molecules)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    np.random.seed(42)

    fingerprints_train = []
    fingerprints_valid = []
    fingerprints_test = []

    for fingerprint, fp_name in zip(fprints, fp_names):
        print(fp_name, "Fingerprint")

        dir_path = save_data_dir + "/" + dataset_name + "/" + fp_name

        if not os.path.exists(dir_path + "_train.npy"):
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
            print(f" - Time of fingerprint computation : {round(execution_time, 2)}s")

            with open(dir_path + "_train.npy", "wb") as f:
                np.save(f, X_fp_train)
            with open(dir_path + "_valid.npy", "wb") as f:
                np.save(f, X_fp_valid)
            with open(dir_path + "_test.npy", "wb") as f:
                np.save(f, X_fp_test)
        else:
            with open(dir_path + "_train.npy", "rb") as f:
                X_fp_train = np.load(f)
            with open(dir_path + "_valid.npy", "rb") as f:
                X_fp_valid = np.load(f)
            with open(dir_path + "_test.npy", "rb") as f:
                X_fp_test = np.load(f)

        fingerprints_train.append(X_fp_train)
        fingerprints_valid.append(X_fp_valid)
        fingerprints_test.append(X_fp_test)
