User Guide
==========

Installation
------------

To use latest stable version of scikit-fingerprints, install it using pip:

.. code-block:: console

   $ pip install scikit-fingerprints

If you wish to have newest changes available on development branch install with:

.. code-block:: console

   $ pip install git+https://github.com/scikit-fingerprints/scikit-fingerprints.git

Basic usage
-----------

The most important feature of scikit-fingerprints are chemical fingerprints - functions used
for embbedding chemical molecules into latent spaces.

.. doctest::

   >>> from skfp.fingerprints import AtomPairFingerprint
   >>>
   >>> smiles_list = ['O=S(=O)(O)CCS(=O)(=O)O', 'O=C(O)c1ccccc1O']
   >>>
   >>> atom_pair_fingerprint = AtomPairFingerprint()
   >>> X_skfp = atom_pair_fingerprint.transform(smiles_list)
   >>> X_skfp # doctest: +NORMALIZE_WHITESPACE
   array([[0, 0, 0, ..., 0, 0, 0],
          [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)



Conformer Fingerprints
----------------------

Some fingerprints use conformers (3D-based), and they need to be provided by user. For them you need
to create molecules first and compute conformers. Those fingerprints have requires_conformers
attribute set to True.


.. doctest::

   >>> from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
   >>> from skfp.fingerprints import WHIMFingerprint
   >>>
   >>> smiles_list = ["O=S(=O)(O)CCS(=O)(=O)O", "O=C(O)c1ccccc1O"]
   >>>
   >>> mol_from_smiles = MolFromSmilesTransformer()
   >>> conf_gen = ConformerGenerator()
   >>> fp = WHIMFingerprint()
   >>> print(fp.requires_conformers)
   True
   >>> mols_list = mol_from_smiles.transform(smiles_list)
   >>> mols_list = conf_gen.transform(mols_list)
   >>>
   >>> X = fp.transform(mols_list)
   >>> print(X) # doctest: +SKIP
   [[4.3190e+00 1.1930e+00 ... 1.9210e+01 2.7304e+01]
    [4.0400e+00 2.1240e+00 ... 1.6414e+01 1.4818e+01]]



scikit-learn compatibility
--------------------------

You can also use scikit-learn functionalities like pipelines, feature unions etc. to build complex workflows.
Popular datasets, e.g. from MoleculeNet benchmark, can be loaded directly.

.. doctest::

   >>> from skfp.datasets.moleculenet import load_clintox
   >>> from skfp.metrics import multioutput_auroc_score, extract_pos_proba
   >>> from skfp.model_selection import scaffold_train_test_split
   >>> from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint
   >>> from skfp.preprocessing import MolFromSmilesTransformer
   >>>
   >>> from sklearn.ensemble import RandomForestClassifier
   >>> from sklearn.pipeline import make_pipeline, make_union
   >>>
   >>>
   >>> smiles, y = load_clintox()
   >>> smiles_train, smiles_test, y_train, y_test = scaffold_train_test_split(
   ...     smiles, y, test_size=0.2
   ... )
   >>>
   >>> pipeline = make_pipeline(
   ...     MolFromSmilesTransformer(),
   ...     make_union(ECFPFingerprint(count=True), MACCSFingerprint()),
   ...     RandomForestClassifier(random_state=0),
   ... )
   >>> pipeline.fit(smiles_train, y_train) # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
   Pipeline(steps=[('molfromsmilestransformer', MolFromSmilesTransformer()),
                    ('featureunion',
                     FeatureUnion(transformer_list=[('ecfpfingerprint',
                                                     ECFPFingerprint(count=True)),
                                                    ('maccsfingerprint',
                                                     MACCSFingerprint())])),
                    ('randomforestclassifier',
                     RandomForestClassifier(random_state=0))])

   >>> y_pred_proba = pipeline.predict_proba(smiles_test)
   >>> y_pred_proba = extract_pos_proba(y_pred_proba)
   >>> auroc = multioutput_auroc_score(y_test, y_pred_proba)
   >>> print(f"AUROC: {auroc:.2%}") # doctest: +SKIP
   AUROC: 84.25%

