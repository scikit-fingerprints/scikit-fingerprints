import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from ogb.graphproppred import GraphPropPredDataset

from skfp.fingerprints import *
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

DATASET_NAME = "ogbg-molhiv"

# N_SPLITS - number of parts in which the dataset will be divided.
# the test is performed first on 1 of them, then 2, ... then N_SPLITS
# testing different sizes of input data
N_SPLITS = 5
# N_REPEATS - number of test repetitions - for getting average time score
N_REPEATS = 5
MAX_CORES = cpu_count(only_physical_cores=True)
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


LEGAL_PLOT_TYPES = ["time", "speedup", "fprints_per_second"]


def save_plots(
    n_molecules: int,
    times: np.ndarray,
    title: str = "",
    save: bool = True,
    plot_type: str = "time",
) -> None:
    dir_name = plot_type

    X = n_molecules * np.linspace(0, 1, N_SPLITS + 1)[1:]

    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot()
    ax1.set_title(title)
    ax1.set_xlabel("Number of molecules")

    if plot_type == "time":
        ax1.set_ylabel("Time of computation")
        for i, y in zip(N_CORES, times):
            ax1.plot(X, y, marker="o", label=f"# cores: {i}")
    elif plot_type == "speedup":
        ax1.set_ylabel("Speedup")
        for i, y in zip(N_CORES[1:], times[1:]):
            ax1.plot(X, times[0] / y, marker="o", label=f"# cores: {i}")
    elif plot_type == "fprints_per_second":
        ax1.set_ylabel("Fingerprints / s")
        for i, y in zip(N_CORES, times):
            ax1.plot(X, X / y, marker="o", label=f"# cores: {i}")

    ax1.set_xlim(n_molecules * 0.1, n_molecules * 1.1)
    ax1.set_ylim(bottom=0)

    plt.legend(loc="upper left", fontsize="14")

    fig.tight_layout()

    if save:
        os.makedirs(os.path.join(PLOT_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, dir_name, f"{title}.png"))
    else:
        plt.show()

    plt.close(fig)


def save_combined_plot(
    n_molecules: int,
    fingerprints: list,
    times: list,
    save: bool = True,
    type: str = "time",
) -> None:
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot()
    ax1.set_ylabel("Fingerprints")
    fp_names = [fp.__name__ for fp in fingerprints]

    if type == "time":
        file_name = "times_of_sequential_computation"
        ax1.set_xlabel("Time of computation")
        ax1.set_title("Times of sequential computation for all fingerprints")
        ax1.barh(fp_names, [time[0, -1] for time in times], color="skyblue")
    elif type == "speedup":
        file_name = "speedup_for_all_cores"
        ax1.set_xlabel("speedup")
        ax1.set_title("Speedup for all fingerprints")
        ax1.barh(
            fp_names, [time[-1, 0] / time[-1, -1] for time in times], color="skyblue"
        )
    elif type == "fprints_per_second":
        file_name = "fingerprints_per_second_sequential"
        ax1.set_xlabel("Fingerprints / s")
        ax1.set_title("Molecules per second for all fingerprints")
        ax1.barh(
            fp_names, [n_molecules / time[0, -1] for time in times], color="skyblue"
        )
    else:
        return

    fig.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, f"{file_name}.png"))
    else:
        plt.show()


if __name__ == "__main__":
    full_time_start = time()

    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)

    GraphPropPredDataset(name=DATASET_NAME, root="../dataset")
    dataset = pd.read_csv(
        f"../dataset/{'_'.join(DATASET_NAME.split('-'))}/mapping/mol.csv.gz"
    )

    if os.path.exists("mols_with_conformers.npy"):
        X = np.load("mols_with_conformers.npy", allow_pickle=True)
    else:
        X = dataset["smiles"][:10000]
        X = MolFromSmilesTransformer().transform(X)
        X = np.array(
            ConformerGenerator(n_jobs=-1, error_on_gen_fail=False).transform(X)
        )
        np.save("mols_with_conformers.npy", X, allow_pickle=True)

    n_molecules = X.shape[0]

    print(f"Number of molecules : {n_molecules}")

    fingerprints = [
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

    times_for_all_fingerprints = []
    for fingerprint in fingerprints:
        if not os.path.exists(os.path.join(SCORE_DIR, f"{fingerprint.__name__}.npy")):
            times = get_times_skfp(X=X, transformer_cls=fingerprint)
            np.save(os.path.join(SCORE_DIR, f"{fingerprint.__name__}.npy"), times)
        else:
            times = np.load(os.path.join(SCORE_DIR, f"{fingerprint.__name__}.npy"))
        for plot_type in LEGAL_PLOT_TYPES:
            save_plots(
                n_molecules=n_molecules,
                times=times,
                title=fingerprint.__name__,
                save=True,
                plot_type=plot_type,
            )
        times_for_all_fingerprints.append(times)

    for plot_type in LEGAL_PLOT_TYPES:
        save_combined_plot(
            n_molecules=n_molecules,
            fingerprints=fingerprints,
            times=times_for_all_fingerprints,
            type=plot_type,
        )

    full_time_end = time()
    print(f"Time of execution: {full_time_end - full_time_start:.2f} s")
