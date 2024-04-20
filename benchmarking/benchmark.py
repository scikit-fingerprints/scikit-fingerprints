import os
from time import time
from typing import Callable, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from ogb.graphproppred import GraphPropPredDataset

from skfp.fingerprints import *
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

dataset_name = "ogbg-molbace"

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
PLOT_DIR = os.path.join("benchmark_times", "benchmark_times_plotted")
SCORE_DIR = os.path.join("benchmark_times", "benchmark_times_saved")


def get_times_skfp(X: np.ndarray, transformer_cls: type, **kwargs) -> np.ndarray:
    print(f" - fingerprint : {transformer_cls.__name__}")
    n_molecules = X.shape[0]

    result = []
    for n_jobs in N_CORES:
        print(f" - - n_jobs : {n_jobs}")
        # testing different fractions of the dataset

        for data_fraction in np.linspace(0, 1, N_SPLITS + 1)[1:]:
            print(f" - - - data fraction : {data_fraction:.2%}")
            idx = round(data_fraction * n_molecules)

            # testing several times to get average computation time
            times = []
            for i in range(N_REPEATS):
                print(f" - - - - repeat : {i}/{N_REPEATS - 1}")
                # select random molecules - data_fraction part of the dataset
                start = time()
                transformer = transformer_cls(n_jobs=n_jobs, **kwargs)
                _ = transformer.transform(X[:idx])
                end = time()
                times.append(end - start)
            result.append(np.mean(times))

    return np.array(result).reshape((len(N_CORES), N_SPLITS))


def save_results(
    n_molecules: int,
    times: np.ndarray,
    title: str = "",
    save: bool = True,
) -> None:

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

    title = title.replace(" ", "_")
    np.save(os.path.join(SCORE_DIR, f"{title}.npy"), times)
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

    if os.path.exists("mols_with_conformers.npy"):
        X = np.load("mols_with_conformers.npy", allow_pickle=True)
    else:
        X = dataset["smiles"]
        X = MolFromSmilesTransformer().transform(X)
        X = np.array(
            ConformerGenerator(n_jobs=-1, error_on_gen_fail=False).transform(X)
        )
        np.save("mols_with_conformers.npy", X, allow_pickle=True)

    n_molecules = X.shape[0]

    print(f"Number of molecules : {n_molecules}")

    fingerprint_constructors = [
        AtomPairFingerprint,
        AutocorrFingerprint,
        AvalonFingerprint,
        E3FPFingerprint,
        ECFPFingerprint,
        ERGFingerprint,
        EStateFingerprint,
        GETAWAYFingerprint,
        LayeredFingerprint,
        MACCSFingerprint,
        MAPFingerprint,
        MHFPFingerprint,
        MordredFingerprint,
        MORSEFingerprint,
        PatternFingerprint,
        PharmacophoreFingerprint,
        PhysiochemicalPropertiesFingerprint,
        PubChemFingerprint,
        RDFFingerprint,
        RDKitFingerprint,
        SECFPFingerprint,
        TopologicalTorsionFingerprint,
        WHIMFingerprint,
    ]

    for fingerprint in fingerprint_constructors:
        if not os.path.exists(os.path.join(SCORE_DIR, f"{fingerprint.__name__}.npy")):
            times = get_times_skfp(X=X, transformer_cls=fingerprint)
            print(times)
            save_results(
                n_molecules=n_molecules,
                times=times,
                title=fingerprint.__name__,
                save=True,
            )

    full_time_end = time()
    print(f"Time of execution: {np.round(full_time_end - full_time_start,2)} s")
