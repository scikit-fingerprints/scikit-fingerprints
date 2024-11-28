import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from ogb.graphproppred import Evaluator

from skfp.datasets.moleculenet import load_ogb_splits, load_moleculenet_benchmark
from skfp.filters import LipinskiFilter, BeyondRo5Filter, BMSFilter, BrenkFilter, FAF4DruglikeFilter, \
    FAF4LeadlikeFilter, \
    GhoseFilter, HaoFilter, InpharmaticaFilter, LINTFilter, MLSMRFilter, MolecularWeightFilter, NIBRFilter, NIHFilter, \
    OpreaFilter, PAINSFilter, PfizerFilter, REOSFilter, RuleOfThreeFilter, RuleOfFourFilter, RuleOfTwoFilter, \
    RuleOfXuFilter, SureChEMBLFilter, TiceHerbicidesFilter, TiceInsecticidesFilter, ValenceDiscoveryFilter, \
    ZINCBasicFilter, ZINCDruglikeFilter
from skfp.fingerprints import AtomPairFingerprint
from skfp.preprocessing import MolFromSmilesTransformer

datasets = load_moleculenet_benchmark(subset="classification")


def get_data_and_labels_at(data, labels, indexes):
    split_data = [data[i] for i in indexes]
    split_labels = [labels[i] for i in indexes]

    return split_data, split_labels


def filter_x_and_y(data, labels, filter):
    filtered_data, filtered_labels = zip(*[(x, labels[i]) for i, x in enumerate(data) if filter.transform([x])])
    return filtered_data, filtered_labels


def smiles_to_fingerprint(smiles):
    atom_pair_fingerprint = AtomPairFingerprint()

    X = atom_pair_fingerprint.transform(smiles)
    return X


def get_model(
        random_state: int,
        hyperparams: dict,
        verbose: bool,
):
    n_jobs = -1

    model = RandomForestClassifier(
        **hyperparams,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )

    return model


def evaluate_model(
        dataset_name: str,
        task_type: str,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
) -> float:
    # use OGB evaluation for MoleculeNet
    if task_type == "classification":
        y_pred = model.predict_proba(X_test)[:, 1]
        y_test = y_test.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    elif task_type == "multioutput_classification":
        # extract positive class probability for each task
        y_pred = model.predict_proba(X_test)
        y_pred = [y_pred_i[:, 1] for y_pred_i in y_pred]
        y_pred = np.column_stack(y_pred)
    else:
        raise ValueError(f"Task type '{task_type}' not recognized")

    evaluator = Evaluator(dataset_name)
    metrics = evaluator.eval(
        {
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    # extract the AUROC
    metric = next(iter(metrics.values()))
    return metric


def activate_filter(filter, X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_train_filtered, y_train_filtered = filter_x_and_y(X_train, y_train, filter)
    X_test_filtered, y_test_filtered = filter_x_and_y(X_test, y_test, filter)
    X_valid_filtered, y_valid_filtered = filter_x_and_y(X_valid, y_valid, filter)

    return X_train_filtered, y_train_filtered, X_valid_filtered, y_valid_filtered, X_test_filtered, y_test_filtered


filter_dict = {
    "Lipinski": LipinskiFilter(),
    "BeyondRo5": BeyondRo5Filter(),
    "BMS": BMSFilter(),
    "Brenk": BrenkFilter(),
    "faf4_druglike": FAF4DruglikeFilter(),
    "faf4_leadlike": FAF4LeadlikeFilter(),
    "ghose": GhoseFilter(),
    "hao": HaoFilter(),
    "inpharmatica": InpharmaticaFilter(),
    "lint": LINTFilter(),
    "mlsmr": MLSMRFilter(),
    "mol_weight": MolecularWeightFilter(),
    "nibr": NIBRFilter(),
    "nih": NIHFilter(),
    "oprea": OpreaFilter(),
    "pains": PAINSFilter(),
    "pfizer": PfizerFilter(),
    "reos": REOSFilter(),
    "rule_of_2" : RuleOfTwoFilter(),
    "rule_of_3": RuleOfThreeFilter(),
    "rule_of_4": RuleOfFourFilter(),
    "rule_of_xu": RuleOfXuFilter(),
    "surechembl": SureChEMBLFilter(),
    "tice_herebicides": TiceHerbicidesFilter(),
    "tice_insecticides": TiceInsecticidesFilter(),
    "valence_discovery": ValenceDiscoveryFilter(),
    "zinc_basic": ZINCBasicFilter(),
    "zinc_druglike": ZINCDruglikeFilter(),
    "None": None
}


def main():
    for dataset in datasets:
        dataset_name, data, labels = dataset
        if dataset_name != "HIV":
            continue
        print(dataset_name)
        train_idx, valid_idx, test_idx = load_ogb_splits(dataset_name)

        X_train, y_train = get_data_and_labels_at(data, labels, train_idx)
        X_valid, y_valid = get_data_and_labels_at(data, labels, valid_idx)
        X_test, y_test = get_data_and_labels_at(data, labels, test_idx)

        for filter_name, filter in filter_dict.items():
            if filter != None:
                train_X, train_y, valid_x, valid_y, test_x, test_y = activate_filter(filter, X_train, y_train, X_valid,
                                                                                     y_valid, X_test, y_test)
            else:
                train_X = X_train
                train_y = y_train
                test_x = X_test
                test_y = y_test
                valid_x = X_valid
                valid_y = y_valid

            fingerprints_train = smiles_to_fingerprint(train_X)
            fingerprints_test = smiles_to_fingerprint(test_x)

            hyperparams = {
                "n_estimators": 1000,
                "criterion": "entropy",
                "min_samples_split": 10,
            }
            model = get_model(
                random_state=0,
                hyperparams=hyperparams,
                verbose=False,
            )
            model.fit(fingerprints_train, train_y)

            y_pred = model.predict(fingerprints_test)
            Mean_accuracy = np.mean(y_pred == test_y)
            if filter != None:
                print(f"{filter_name} Filtered: {Mean_accuracy}")
            else:
                print(f"Unfiltered: {Mean_accuracy}")


if __name__ == "__main__":
    main()
