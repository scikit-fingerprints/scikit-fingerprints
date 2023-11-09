import os
from time import time
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit.Chem.rdFingerprintGenerator as fpgens
from joblib import cpu_count
from ogb.graphproppred import GraphPropPredDataset
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from scipy.sparse import csr_array

from base import FingerprintTransformer
from featurizers.fingerprints import (
    AtomPairFingerprint,
    ERGFingerprint,
    MACCSKeysFingerprint,
    MorganFingerprint,
    TopologicalTorsionFingerprint,
)

dataset_name = "ogbg-molhiv"

# N_SPLITS - number of parts in which the dataset will be divided.
# the test is performed first on 1 of them, then 2, ... then N_SPLITS
# testing different sizes of input data
N_SPLITS = 5
# N_REPEATS - number of test repetitions - for getting average time score
N_REPEATS = 5
N_CORES = [i for i in range(1, cpu_count() + 1)][:8]
COUNT_TYPES = [False, True]
SPARSE_TYPES = [False, True]
PLOT_DIR = "./benchmark_times_plotted"


def get_times_emf(
    X: pd.DataFrame, transformer_function: FingerprintTransformer, **kwargs
):
    n_molecules = X.shape[0]
    emf_transformer = transformer_function(**kwargs)

    # testing for different sizes of input datasets
    result = []
    for data_fraction in np.linspace(0, 1, N_SPLITS + 1)[1:]:
        n = int(n_molecules * data_fraction)
        subset = X[:n]
        times = [None for _ in range(N_REPEATS)]
        # testing several times to get average computation time
        for i in range(N_REPEATS):
            start = time()
            X_transformed = emf_transformer.transform(subset)
            end = time()
            times[i] = end - start
        result.append(np.mean(times))
    return np.array(result)


def get_generator_times_rdkit(
    X: pd.DataFrame, generator: object, count: Optional[bool], sparse: bool
):
    n_molecules = X.shape[0]
    if count:
        fp_function = lambda x: generator.GetCountFingerprint(
            MolFromSmiles(x)
        ).ToList()
    else:
        fp_function = lambda x: generator.GetFingerprint(MolFromSmiles(x))

    # testing for different sizes of input datasets
    result = []
    for data_fraction in np.linspace(0, 1, N_SPLITS + 1)[1:]:
        n = int(n_molecules * data_fraction)
        subset = X[:n]
        times = [None for _ in range(N_REPEATS)]
        # testing several times to get average computation time
        for i in range(N_REPEATS):
            start = time()
            if sparse:
                X_transformed = csr_array([fp_function(x) for x in subset])
            else:
                X_transformed = np.array([fp_function(x) for x in subset])
            end = time()
            times[i] = end - start
        result.append(np.mean(times))
    return np.array(result)


def get_times_rdkit(
    X: pd.DataFrame, func: Callable, sparse: bool = False, **kwargs
):
    n_molecules = X.shape[0]
    # testing for different sizes of input datasets
    result = []
    for data_fraction in np.linspace(0, 1, N_SPLITS + 1)[1:]:
        n = int(n_molecules * data_fraction)
        subset = X[:n]
        times = [None for _ in range(N_REPEATS)]
        # testing several times to get average computation time
        for i in range(N_REPEATS):
            start = time()
            if sparse:
                X_transformed = csr_array(
                    [func(MolFromSmiles(x), **kwargs) for x in subset]
                )
            else:
                X_transformed = np.array(
                    [func(MolFromSmiles(x), **kwargs) for x in subset]
                )
            end = time()
            times[i] = end - start
        result.append(np.mean(times))
    return np.array(result)


def plot_results(
    n_molecules: int,
    y_emf: List,
    y_rdkit: List,
    title: str = "",
    sparse: bool = None,
    count: bool = None,
    save: bool = True,
):
    if sparse:
        title += " sparse"

    if count:
        title += " count"
    elif count is not None:
        title += " bit"

    X = n_molecules * np.linspace(0, 1, N_SPLITS + 1)[1:]

    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot()
    ax1.set_title(title)

    for i, y in enumerate(y_emf):
        ax1.plot(X, y, label=f"emf time - {i+1}")

    ax1.plot(X, y_rdkit, label="rdkit time")

    ax1.set_ylabel("Time of computation")
    ax1.set_xlabel("Number of fingerprints")

    ax1.set_xlim(n_molecules * 0.1, n_molecules * 1.1)
    ax1.set_ylim(bottom=0)

    plt.legend(loc="upper left", fontsize="8")
    if save:
        plt.savefig(PLOT_DIR + "/" + title.replace(" ", "_") + ".png")
    else:
        plt.show()


if __name__ == "__main__":
    benchmark_time_start = time()

    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    GraphPropPredDataset(name=dataset_name)
    dataset = pd.read_csv(
        f"./dataset/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]
    y = dataset["HIV_active"]

    n_molecules = X.shape[0]

    # MORGAN FINGERPRINT
    morgan_emf_times = [
        [
            [
                get_times_emf(
                    X,
                    MorganFingerprint,
                    sparse=sparse,
                    count=count,
                    n_jobs=n_cores,
                )
                for n_cores in N_CORES
            ]
            for sparse in SPARSE_TYPES
        ]
        for count in COUNT_TYPES
    ]

    generator = fpgens.GetMorganGenerator()

    morgan_rdkit_times = [
        [
            get_generator_times_rdkit(
                X,
                generator,
                sparse=sparse,
                count=count,
            )
            for sparse in SPARSE_TYPES
        ]
        for count in COUNT_TYPES
    ]

    for i, count in enumerate(COUNT_TYPES):
        for j, sparse in enumerate(SPARSE_TYPES):
            plot_results(
                n_molecules,
                morgan_emf_times[i][j],
                morgan_rdkit_times[i][j],
                "Morgan Fingerprint",
                count,
                sparse,
            )

    # ATOM PAIR FINGERPRINT
    atom_pair_emf_times = [
        [
            [
                get_times_emf(
                    X,
                    AtomPairFingerprint,
                    sparse=sparse,
                    count=count,
                    n_jobs=n_cores,
                )
                for n_cores in N_CORES
            ]
            for sparse in SPARSE_TYPES
        ]
        for count in COUNT_TYPES
    ]

    generator = fpgens.GetAtomPairGenerator()

    atom_pair_rdkit_times = [
        [
            get_generator_times_rdkit(
                X,
                generator,
                sparse=sparse,
                count=count,
            )
            for sparse in SPARSE_TYPES
        ]
        for count in COUNT_TYPES
    ]

    for i, count in enumerate(COUNT_TYPES):
        for j, sparse in enumerate(SPARSE_TYPES):
            plot_results(
                n_molecules,
                atom_pair_emf_times[i][j],
                atom_pair_rdkit_times[i][j],
                "Atom Pair Fingerprint",
                count,
                sparse,
            )

    # TOPOLOGICAL TORSION FINGERPRINT
    topological_torsion_emf_times = [
        [
            [
                get_times_emf(
                    X,
                    TopologicalTorsionFingerprint,
                    sparse=sparse,
                    count=count,
                    n_jobs=n_cores,
                )
                for n_cores in N_CORES
            ]
            for sparse in SPARSE_TYPES
        ]
        for count in COUNT_TYPES
    ]

    generator = fpgens.GetTopologicalTorsionGenerator()

    topological_torsion_rdkit_times = [
        [
            get_generator_times_rdkit(
                X,
                generator,
                sparse=sparse,
                count=count,
            )
            for sparse in SPARSE_TYPES
        ]
        for count in COUNT_TYPES
    ]

    for i, count in enumerate(COUNT_TYPES):
        for j, sparse in enumerate(SPARSE_TYPES):
            plot_results(
                n_molecules,
                topological_torsion_emf_times[i][j],
                topological_torsion_rdkit_times[i][j],
                "Topological Torsion Fingerprint",
                count,
                sparse,
            )

    # MACCS KEYS FINGERPRINT
    MACCSKeys_emf_times = [
        [
            get_times_emf(
                X,
                MACCSKeysFingerprint,
                n_jobs=n_cores,
                sparse=sparse,
            )
            for n_cores in N_CORES
        ]
        for sparse in SPARSE_TYPES
    ]

    MACCSKeys_rdkit_times = [
        get_times_rdkit(X, GetMACCSKeysFingerprint, sparse=sparse)
        for sparse in SPARSE_TYPES
    ]

    for i, sparse in enumerate(SPARSE_TYPES):
        plot_results(
            n_molecules,
            MACCSKeys_emf_times[i],
            MACCSKeys_rdkit_times[i],
            "MACCKeys fingerprint",
            count=None,
            sparse=sparse,
        )

    # ERG FINGERPRINT
    ERG_emf_times = [
        [
            get_times_emf(X, ERGFingerprint, n_jobs=n_cores, sparse=sparse)
            for n_cores in N_CORES
        ]
        for sparse in SPARSE_TYPES
    ]

    ERG_rdkit_times = [
        get_times_rdkit(X, GetErGFingerprint, sparse=sparse)
        for sparse in SPARSE_TYPES
    ]

    for i, sparse in enumerate(SPARSE_TYPES):
        plot_results(
            n_molecules,
            ERG_emf_times[i],
            ERG_rdkit_times[i],
            "ERG fingerprint",
            count=None,
            sparse=sparse,
        )

    benchmark_time_end = time()
    print("Time of execution: ", benchmark_time_end-benchmark_time_start, "s")
