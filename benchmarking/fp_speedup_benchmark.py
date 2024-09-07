import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from ogb.graphproppred import GraphPropPredDataset
from rdkit.Chem import Mol

import skfp.fingerprints as fps
from skfp.preprocessing import MolFromSmilesTransformer

N_REPEATS = 5  # number of calculation repetitions
N_CORES = cpu_count(only_physical_cores=True)
DATASET_NAME = "ogbg-molhiv"


def measure_speedups(X: list[Mol], fp_name: str, transformer_cls: type) -> list[float]:
    print(f"Fingerprint: {fp_name}")

    os.makedirs("benchmark_speedups", exist_ok=True)
    if os.path.exists(f"benchmark_speedups/{fp_name}.npy"):
        print("Loading from cached file")
        return np.load(f"benchmark_speedups/{fp_name}.npy", allow_pickle=True)

    all_times = []
    for n_jobs in list(range(1, N_CORES + 1)):
        print(f"    n_jobs: {n_jobs}")
        curr_times = []
        for i in range(N_REPEATS):
            start = time()
            transformer_cls(n_jobs=n_jobs).transform(X)
            end = time()
            curr_times.append(end - start)
        all_times.append(np.mean(curr_times))

    serial_time = all_times[0]
    speedups = [serial_time / t for t in all_times]

    np.save(f"benchmark_speedups/{fp_name}.npy", speedups, allow_pickle=True)
    return speedups


def make_plot(fp_speedups: dict[str, list[float]]) -> None:
    ax = plt.gca()

    # set titles
    ax.set_title("Speedup plot")
    ax.set_xlabel("Number of processes")
    ax.set_ylabel("Speedup")

    # set global plotting parameters, add y=x line
    ax.set_aspect("equal")
    ax.set_xlim(0, N_CORES + 0.5)
    ax.set_ylim(0, N_CORES + 0.5)
    ax.axline((0, 0), slope=1, alpha=0.75, linestyle="dashed")

    # add fingerprint speedups
    xs = list(range(1, N_CORES + 1))
    for fp_name, speedups in fp_speedups.items():
        ax.plot(xs, speedups, marker="o", label=fp_name)

    ax.legend(loc="upper left", fontsize="14")
    plt.tight_layout()
    plt.savefig("speedups.png")


if __name__ == "__main__":
    full_time_start = time()

    GraphPropPredDataset(name=DATASET_NAME, root=os.path.join("..", "dataset"))
    dataset_path = os.path.join(
        "..", "dataset", "_".join(DATASET_NAME.split("-")), "mapping", "mol.csv.gz"
    )
    dataset = pd.read_csv(dataset_path)

    X = MolFromSmilesTransformer().transform(dataset["smiles"])

    fp_classes = {
        # descriptors
        "EState": fps.EStateFingerprint,
        "Mordred": fps.MordredFingerprint,
        # substructural
        "MACCS": fps.MACCSFingerprint,
        "PubChem": fps.PubChemFingerprint,
        # hashed
        "ECFP": fps.ECFPFingerprint,
        "RDKit": fps.RDKitFingerprint,
    }
    fp_speedups = {
        fp_name: measure_speedups(X, fp_name, fp_cls)
        for fp_name, fp_cls in fp_classes.items()
    }
    make_plot(fp_speedups)
