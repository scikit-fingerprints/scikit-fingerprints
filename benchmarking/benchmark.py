import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

import rdkit.Chem.rdFingerprintGenerator as fpgens
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

from time import time

from featurizers.fingerprints import (
    MorganFingerprint,
    AtomPairFingerprint,
    TopologicalTorsionFingerprint,
    MACCSKeysFingerprint,
    ERGFingerprint,
)

dataset_name = "ogbg-molhiv"
N_SPLITS = 5
N_REPEATS = 5
N_CORES = [1, 2, 4, -1]
COUNT_TYPES = [False, True]
SPARSE_TYPES = [False, True]


def get_times_emf(X, n_molecules, transformer_function, **kwargs):
    result = []
    emf_transformer = transformer_function(**kwargs)
    for data_fraction in np.linspace(0, 1, N_SPLITS + 1)[1:]:
        n = int(n_molecules * data_fraction)
        subset = X[:n]
        times = [None for _ in range(N_REPEATS)]
        for i in range(N_REPEATS):
            start = time()
            X_transformed = emf_transformer.transform(subset)
            end = time()
            times[i] = end - start
        result.append(sum(times) / N_REPEATS)
    return np.array(result)


def get_generator_times_rdkit(X, n_molecules, generator, count, sparse):
    if count:
        fp_function = lambda x: generator.GetCountFingerprint(
            MolFromSmiles(x)
        ).ToList()
    else:
        fp_function = lambda x: generator.GetFingerprint(MolFromSmiles(x))
    result = []
    for data_fraction in np.linspace(0, 1, N_SPLITS + 1)[1:]:
        n = int(n_molecules * data_fraction)
        subset = X[:n]
        times = [None for _ in range(N_REPEATS)]
        for i in range(N_REPEATS):
            start = time()
            if sparse:
                X_transformed = csr_array([fp_function(x) for x in subset])
            else:
                X_transformed = np.array([fp_function(x) for x in subset])
            end = time()
            times[i] = end - start
        result.append(sum(times) / N_REPEATS)
    return np.array(result)


def get_times_rdkit(X, n_molecules, func, sparse=False, **kwargs):
    result = []
    for data_fraction in np.linspace(0, 1, N_SPLITS + 1)[1:]:
        n = int(n_molecules * data_fraction)
        subset = X[:n]
        times = [None for _ in range(N_REPEATS)]
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
        result.append(sum(times) / N_REPEATS)
    return np.array(result)


def plot_results(
    n_molecules, y_emf, y_rdkit, title="", sparse=None, count=None
):
    if sparse is not None:
        if sparse:
            title += " sparse"

    if count is not None:
        if count:
            title += " count"
        else:
            title += " bit"

    X = n_molecules * np.linspace(0, 1, N_SPLITS + 1)[1:]

    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot()
    ax1.set_title(title)

    ax1.plot(X, y_emf[0], label="emf time - 1 job")
    ax1.plot(X, y_emf[1], label="emf time - 2 job")
    ax1.plot(X, y_emf[2], label="emf time - 4 job")
    ax1.plot(X, y_emf[3], label="emf time - all jobs")
    ax1.plot(X, y_rdkit, label="rdkit time")

    ax1.set_ylabel("Time of computiation")
    ax1.set_xlabel("Number of finberprints")

    ax1.set_xlim(n_molecules * 0.1, n_molecules * 1.1)
    ax1.set_ylim(bottom=0)

    plt.legend(loc="upper left")
    plt.savefig(title.replace(" ", "_") + ".png")
    # plt.show()


from rdkit.Chem import MolFromSmiles

if __name__ == "__main__":
    # GraphPropPredDataset(name=dataset_name)
    dataset = pd.read_csv(
        f"../dataset/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]
    y = dataset["HIV_active"]

    n_molecules = X.shape[0]

    # MORGAN FINGERPRINT
    print("Morgan")
    print("emf times")c
    morgan_emf_times = [
        [
            [
                get_times_emf(
                    X,
                    n_molecules,
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

    print("rdkit times")
    morgan_rdkit_times = [
        [
            get_generator_times_rdkit(
                X,
                n_molecules,
                generator,
                sparse=sparse,
                count=count,
            )
            for sparse in SPARSE_TYPES
        ]
        for count in COUNT_TYPES
    ]

    for count, i in enumerate(COUNT_TYPES):
        for sparse, j in enumerate(SPARSE_TYPES):
            plot_results(
                n_molecules,
                morgan_emf_times[i][j],
                morgan_rdkit_times[i][j],
                "Morgan Fingerprint",
                count,
                sparse,
            )

    # ATOM PAIR FINGERPRINT
    print("Atom Pair")
    print("emf times")
    atom_pair_emf_times = [
        [
            [
                get_times_emf(
                    X,
                    n_molecules,
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

    print("rdkit times")
    atom_pair_rdkit_times = [
        [
            get_generator_times_rdkit(
                X,
                n_molecules,
                generator,
                sparse=sparse,
                count=count,
            )
            for sparse in SPARSE_TYPES
        ]
        for count in COUNT_TYPES
    ]

    for count, i in enumerate(COUNT_TYPES):
        for sparse, j in enumerate(SPARSE_TYPES):
            plot_results(
                n_molecules,
                atom_pair_emf_times[i][j],
                atom_pair_rdkit_times[i][j],
                "Atom Pair Fingerprint",
                count,
                sparse,
            )

    # TOPOLOGICAL TORSION FINGERPRINT
    print("Topological Torsion")
    print("emf times")
    topological_torsion_emf_times = [
        [
            [
                get_times_emf(
                    X,
                    n_molecules,
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

    print("rdkit times")
    topological_torsion_rdkit_times = [
        [
            get_generator_times_rdkit(
                X,
                n_molecules,
                generator,
                sparse=sparse,
                count=count,
            )
            for sparse in SPARSE_TYPES
        ]
        for count in COUNT_TYPES
    ]

    for count, i in enumerate(COUNT_TYPES):
        for sparse, j in enumerate(SPARSE_TYPES):
            plot_results(
                n_molecules,
                topological_torsion_emf_times[i][j],
                topological_torsion_rdkit_times[i][j],
                "Topological Torsion Fingerprint",
                count,
                sparse,
            )

    # MACCS KEYS FINGERPRINT
    print("MACCS Keys")
    print("emf times")
    MACCSKeys_emf_times = [
        [
            get_times_emf(
                X,
                n_molecules,
                MACCSKeysFingerprint,
                n_jobs=n_cores,
                sparse=sparse,
            )
            for n_cores in N_CORES
        ]
        for sparse in SPARSE_TYPES
    ]

    print("rdkit times")
    MACCSKeys_rdkit_times = [
        get_times_rdkit(X, n_molecules, GetMACCSKeysFingerprint, sparse=sparse)
        for sparse in SPARSE_TYPES
    ]

    for sparse, i in enumerate(SPARSE_TYPES):
        plot_results(
            n_molecules,
            MACCSKeys_emf_times[i],
            MACCSKeys_rdkit_times[i],
            "MACCKeys fingerprint",
            count=None,
            sparse=sparse,
        )

    # ERG FINGERPRINT
    print("ERG")
    print("emf times")
    ERG_emf_times = [
        [
            get_times_emf(
                X, n_molecules, ERGFingerprint, n_jobs=n_cores, sparse=sparse
            )
            for n_cores in N_CORES
        ]
        for sparse in SPARSE_TYPES
    ]

    print("rdkit times")
    ERG_rdkit_times = [
        get_times_rdkit(X, n_molecules, GetErGFingerprint, sparse=sparse)
        for sparse in SPARSE_TYPES
    ]

    for sparse, i in enumerate(SPARSE_TYPES):
        plot_results(
            n_molecules,
            ERG_emf_times[i],
            ERG_rdkit_times[i],
            "ERG fingerprint",
            count=None,
            sparse=sparse,
        )
