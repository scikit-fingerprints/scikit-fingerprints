import gc
import inspect
import os
from time import sleep, time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skfp.fingerprints as fps
from joblib import cpu_count
from skfp.datasets.moleculenet import load_hiv
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

mpl.rcParams.update({"font.size": 18})

# N_SPLITS - number of parts in which the dataset will be divided.
# the test is performed first on 1 of them, then 2, ... then N_SPLITS
# testing different sizes of input data
N_SPLITS = 5
# N_REPEATS - number of test repetitions - for getting average time score
N_REPEATS = 5
MAX_CORES = cpu_count(only_physical_cores=True)
N_CORES = [2**i for i in range(MAX_CORES.bit_length())]
if max(N_CORES) < MAX_CORES:
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
                gc.collect()
                sleep(10)
                times.append(end - start)
            result.append(np.mean(times))

    return np.array(result).reshape((len(N_CORES), N_SPLITS))


PLOT_TYPES = ["time", "speedup", "fps_per_second"]


def make_plot(
    plot_type: str,
    n_molecules: int,
    times: np.ndarray,
    title: str = "",
    save: bool = True,
    file_format: str = "pdf",
) -> None:
    dir_name = plot_type

    X = n_molecules * np.linspace(0, 1, N_SPLITS + 1)[1:]

    plt.rcParams["font.size"] = 20
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot()
    ax1.set_title(title)
    ax1.set_xlabel("Number of molecules")

    if plot_type == "time":
        ax1.set_ylabel("Time of computation [s]")
        for i, y in zip(N_CORES, times, strict=False):
            ax1.plot(X, y, marker="o", label=f"# cores: {i}")
    elif plot_type == "speedup":
        ax1.set_ylabel("Speedup")
        for i, y in zip(N_CORES[1:], times[1:], strict=False):
            ax1.plot(X, times[0] / y, marker="o", label=f"# cores: {i}")
    elif plot_type == "fps_per_second":
        ax1.set_ylabel("Fingerprints per second")
        for i, y in zip(N_CORES, times, strict=False):
            ax1.plot(X, X / y, marker="o", label=f"# cores: {i}")

    ax1.set_xlim(n_molecules * 0.1, n_molecules * 1.1)
    ax1.set_ylim(bottom=0)

    plt.legend(loc="upper left", fontsize="14")

    fig.tight_layout()

    if save:
        os.makedirs(os.path.join(PLOT_DIR, dir_name), exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, dir_name, f"{title}.{file_format}"))
    else:
        plt.show()

    plt.close(fig)


def make_combined_plot(
    plot_type: str,
    n_molecules: int,
    fingerprints: list,
    times: list,
    save: bool = True,
    file_format: str = "pdf",
) -> None:
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot()
    fp_names = [fp.__name__.removesuffix("Fingerprint") for fp in fingerprints]

    if plot_type == "time":
        file_name = "times_of_sequential_computation"
        ax1.set_xlabel("Time of computation")
        ax1.set_title("Sequential computation time [s]")
        times_to_plot = [time[0, -1] for time in times]
        ax1.barh(fp_names, times_to_plot, color="skyblue")
    elif plot_type == "speedup":
        file_name = f"speedup_for_{MAX_CORES}_cores"
        plt.xticks(np.arange(0, 17))
        ax1.axvline(x=1, linestyle="dashed")
        ax1.set_xlabel("speedup")
        ax1.set_title("Speedup")
        times_to_plot = [time[0, -1] / time[-1, -1] for time in times]
        ax1.barh(fp_names, times_to_plot, color="skyblue")
    elif plot_type == "fps_per_second":
        file_name = "fingerprints_per_second_sequential"
        ax1.set_xlabel("Fingerprints per second")
        ax1.set_title("Molecules processed per second")
        times_to_plot = [n_molecules / time[0, -1] for time in times]
        ax1.barh(fp_names, times_to_plot, color="skyblue")
    else:
        return

    fig.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOT_DIR, f"{file_name}.{file_format}"))
    else:
        plt.show()


if __name__ == "__main__":
    full_time_start = time()

    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)

    dataset = load_hiv()

    if os.path.exists("mols_with_conformers.npy"):
        X = np.load("mols_with_conformers.npy", allow_pickle=True)
    else:
        X = dataset["smiles"][:10000]
        X = MolFromSmilesTransformer(valid_only=True).transform(X)
        X = ConformerGenerator(n_jobs=-1, errors="filter").transform(X)
        X = np.array(X)
        np.save("mols_with_conformers.npy", X, allow_pickle=True)

    n_molecules = X.shape[0]

    print(f"Number of molecules : {n_molecules}")

    fingerprints = [
        fps.AtomPairFingerprint,
        fps.AutocorrFingerprint,
        fps.AvalonFingerprint,
        fps.E3FPFingerprint,
        fps.ECFPFingerprint,
        fps.ERGFingerprint,
        fps.EStateFingerprint,
        fps.FunctionalGroupsFingerprint,
        fps.GETAWAYFingerprint,
        fps.GhoseCrippenFingerprint,
        fps.KlekotaRothFingerprint,
        fps.LaggnerFingerprint,
        fps.LayeredFingerprint,
        fps.LingoFingerprint,
        fps.MACCSFingerprint,
        fps.MAPFingerprint,
        fps.MHFPFingerprint,
        fps.MordredFingerprint,
        fps.MORSEFingerprint,
        fps.PatternFingerprint,
        fps.PharmacophoreFingerprint,
        fps.PhysiochemicalPropertiesFingerprint,
        fps.PubChemFingerprint,
        fps.RDFFingerprint,
        fps.RDKitFingerprint,
        fps.SECFPFingerprint,
        fps.TopologicalTorsionFingerprint,
        fps.USRFingerprint,
        fps.USRCATFingerprint,
        fps.WHIMFingerprint,
    ]

    all_times = []
    for fingerprint in fingerprints:
        fingerprint_save_path = os.path.join(SCORE_DIR, f"{fingerprint.__name__}.npy")
        if not os.path.exists(fingerprint_save_path):
            kwargs = {}
            if "errors" in inspect.signature(fingerprint).parameters:
                kwargs["errors"] = "ignore"
            times = get_times_skfp(X=X, transformer_cls=fingerprint, **kwargs)
            np.save(fingerprint_save_path, times)
        else:
            times = np.load(fingerprint_save_path)[: len(N_CORES)]
        for plot_type in PLOT_TYPES:
            make_plot(
                plot_type=plot_type,
                n_molecules=n_molecules,
                times=times,
                title=fingerprint.__name__,
                save=True,
            )
        all_times.append(times)

    for plot_type in PLOT_TYPES:
        make_combined_plot(
            plot_type=plot_type,
            n_molecules=n_molecules,
            fingerprints=fingerprints,
            times=all_times,
        )

    full_time_end = time()
    print(f"Time of execution: {full_time_end - full_time_start:.2f} s")
