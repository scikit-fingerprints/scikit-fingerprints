Usage
=====

Installation
------------

To use scikit-fingerprints, first install it using pip:

.. code-block:: console

   (.venv) $ pip install scikit-fingerprints

Then you can import the package and use it in your code:

.. code-block:: python

   from skfp.fingerprints import AtomPairFingerprint

   smiles_list = ['O=S(=O)(O)CCS(=O)(=O)O', 'O=C(O)c1ccccc1O']

   atom_pair_fingerprint = AtomPairFingerprint()
   X_skfp = atom_pair_fingerprint.transform(smiles_list)

   print(X_skfp)