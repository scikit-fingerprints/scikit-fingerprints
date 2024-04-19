import os
from time import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from ogb.graphproppred import GraphPropPredDataset

from skfp.fingerprints import *
from skfp.preprocessing import MolFromSmilesTransformer

dataset_name = "ogbg-molhiv"

# N_SPLITS - number of parts in which the dataset will be divided.
# the test is performed first on 1 of them, then 2, ... then N_SPLITS
# testing different sizes of input data
N_SPLITS = 5
# N_REPEATS - number of test repetitions - for getting average time score
N_REPEATS = 5
MAX_CORES = cpu_count()
N_CORES = [2**i for i in range(MAX_CORES.bit_length())]
if MAX_CORES > N_CORES[-1]:
    N_CORES.append(MAX_CORES)
PLOT_DIR = "benchmark_times/benchmark_times_plotted"
SCORE_DIR = "benchmark_times/benchmark_times_saved"


def get_times_skfp(X: pd.DataFrame, transformer_constructor: Callable, **kwargs):
    print(f" - fingerprint : {transformer_constructor.__name__}")
    n_molecules = X.shape[0]

    result = []
    for n_jobs in N_CORES:
        print(f" - - n_jobs : {n_jobs}")
        # testing different fractions of the dataset

        for data_fraction in np.linspace(0, 1, N_SPLITS + 1)[1:]:
            print(f" - - - data fraction : {round(data_fraction*100,2)}%")
            idx = (data_fraction * n_molecules).astype(int)

            # testing several times to get average computation time
            times = []
            for i in range(N_REPEATS):
                print(f" - - - - repeat : {i}/{N_REPEATS - 1}")
                # select random molecules - data_fraction part of the dataset
                start = time()
                transformer = transformer_constructor(n_jobs=n_jobs, **kwargs)
                _ = transformer.transform(X[:idx])
                end = time()
                times.append(end - start)
            result.append(np.mean(times))

    return np.array(result).reshape((len(N_CORES), N_SPLITS))


def save_results(
    n_molecules: int,
    times: list,
    title: str = "",
    save: bool = True,
):

    X = n_molecules * np.linspace(0, 1, N_SPLITS + 1)[1:]

    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot()
    ax1.set_title(title)

    for i, y in zip(N_CORES, times):
        ax1.plot(X, y, label=f"our time - # cores: {i}")

    ax1.set_ylabel("Time of computation [s]")
    ax1.set_xlabel("Number of fingerprints")

    ax1.set_xlim(n_molecules * 0.1, n_molecules * 1.1)
    ax1.set_ylim(bottom=0)

    plt.legend(loc="upper left", fontsize="14")

    to_save = np.object_([times])

    title = title.replace(" ", "_")
    np.save(os.path.join(SCORE_DIR, f"{title}.npy"), to_save)
    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"{title}.png"))
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    full_time_start = time()

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)

    GraphPropPredDataset(name=dataset_name, root="../dataset")
    dataset = pd.read_csv(
        f"../dataset/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]
    X = np.array(MolFromSmilesTransformer().transform(X))

    y = dataset["HIV_active"]

    n_molecules = X.shape[0]

    print(f"Number of molecules : {n_molecules}")

    fingerprint_constructors = [
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

    for fingerprint in fingerprint_constructors:
        if not os.path.exists(os.path.join(SCORE_DIR, f"{fingerprint.__name__}.npy")):
            times = get_times_skfp(X, fingerprint)
            print(times)
            save_results(n_molecules, times, fingerprint.__name__, True)

    full_time_end = time()
    print(f"Time of execution: {full_time_end - full_time_start} s")
