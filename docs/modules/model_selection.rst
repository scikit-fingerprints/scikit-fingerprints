===============
Model selection
===============

Tools for model selection, such as train-test splitting and hyperparameter tuning.

.. automodule:: skfp.model_selection

=========================================================

.. py:currentmodule:: skfp.model_selection

Hyperparameter optimization:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    FingerprintEstimatorGridSearch
    FingerprintEstimatorRandomizedSearch

Splitters:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    butina_train_test_split
    butina_train_valid_test_split
    maxmin_train_test_split
    maxmin_train_valid_test_split
    maxmin_stratified_train_test_split
    maxmin_stratified_train_valid_test_split
    randomized_scaffold_train_test_split
    randomized_scaffold_train_valid_test_split
    scaffold_train_test_split
    scaffold_train_valid_test_split
