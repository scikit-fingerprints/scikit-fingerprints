=============
Preprocessing
=============

.. automodule:: skfp.preprocessing

=========================================================

.. py:currentmodule:: skfp.preprocessing

Reading and writing molecular formats:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    MolFromAminoseqTransformer
    MolFromInchiTransformer
    MolFromSDFTransformer
    MolFromSmilesTransformer
    MolToInchiTransformer
    MolToSDFTransformer
    MolToSmilesTransformer


Preprocessing classes:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    ConformerGenerator
    MolStandardizer

Molecular filters:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    BasicZINCFilter
    BeyondRo5Filter
    BMSFilter
    BrenkFilter
    GhoseFilter
    GlaxoFilter
    HaoFilter
    InpharmaticaFilter
    LINTFilter
    LipinskiFilter
    MLSMRFilter
    MolecularWeightFilter
    NIHFilter
    PAINSFilter
    PfizerFilter
    RuleOfFour
    RuleOfReos
    RuleOfThree
    RuleOfTwo
    RuleOfVeber
    RuleOfXu
    SureChEMBLFilter
    TiceHerbicidesFilter
    TiceInsecticidesFilter
