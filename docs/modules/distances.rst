==========================
Distances and similarities
==========================

Functions for computing distance and similarity measures common in chemoinformatics.

.. automodule:: skfp.distances

=========================================================

.. py:currentmodule:: skfp.distances

Distances
---------

Single vectors:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    braun_blanquet_binary_distance
    ct4_binary_distance
    ct4_count_distance
    dice_binary_distance
    dice_count_distance
    fraggle_distance
    harris_lahey_binary_distance
    kulczynski_binary_distance
    mcconnaughey_binary_distance
    rand_binary_distance
    rogot_goldberg_binary_distance
    russell_binary_distance
    simpson_binary_distance
    sokal_sneath_2_binary_distance
    tanimoto_binary_distance
    tanimoto_count_distance

Bulk functions for matrices (pairwise distances):

.. autosummary::
    :nosignatures:
    :toctree: generated/

    bulk_rand_binary_distance
    bulk_simpson_binary_distance
    bulk_tanimoto_binary_distance
    bulk_tanimoto_count_distance

Similarities
------------

Single vectors:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    braun_blanquet_binary_similarity
    ct4_binary_similarity
    ct4_count_similarity
    dice_binary_similarity
    dice_count_similarity
    fraggle_similarity
    harris_lahey_binary_similarity
    kulczynski_binary_similarity
    mcconnaughey_binary_similarity
    rand_binary_similarity
    rogot_goldberg_binary_similarity
    russell_binary_similarity
    simpson_binary_similarity
    sokal_sneath_2_binary_similarity
    tanimoto_binary_similarity
    tanimoto_count_similarity

Bulk functions for matrices (pairwise similarities):

.. autosummary::
    :nosignatures:
    :toctree: generated/

    bulk_rand_binary_similarity
    bulk_simpson_binary_similarity
    bulk_tanimoto_binary_similarity
    bulk_tanimoto_count_similarity
