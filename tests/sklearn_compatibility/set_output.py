import inspect

import pandas as pd
import sklearn

import skfp.fingerprints


def test_individual_set_output(mols_conformers_list):
    mols_conformers_list = mols_conformers_list[:10]

    for name, obj in inspect.getmembers(skfp.fingerprints):
        if not inspect.isclass(obj):
            continue

        fp = obj()
        fp.set_output(transform="pandas")
        output = fp.transform(mols_conformers_list)
        if not isinstance(output, pd.DataFrame):
            raise AssertionError(
                f"For fingerprint {name} output was of type {type(output)} "
                f"instead of pd.DataFrame"
            )


def test_global_set_output(mols_conformers_list):
    mols_conformers_list = mols_conformers_list[:10]

    with sklearn.config_context(transform_output="pandas"):
        # loop through all fingerprint classes
        for name, obj in inspect.getmembers(skfp.fingerprints):
            if not inspect.isclass(obj):
                continue

            # check that class properly sets the output format
            fp = obj()
            output = fp.transform(mols_conformers_list)
            if not isinstance(output, pd.DataFrame):
                raise AssertionError(
                    f"For fingerprint {name} output was of type {type(output)} "
                    f"instead of pd.DataFrame"
                )
