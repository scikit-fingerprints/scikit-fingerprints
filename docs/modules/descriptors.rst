===========
Descriptors
===========

Modules for computing molecular descriptors.

.. automodule:: skfp.descriptors

=========================================================

.. py:currentmodule:: skfp.descriptors

Constitutional descriptors (based on atomic composition):

.. autosummary::
    :nosignatures:
    :toctree: generated/

    average_molecular_weight
    bond_type_count
    element_atom_count
    heavy_atom_count
    molecular_weight
    number_of_rings
    number_of_rotatable_bonds
    total_atom_count

Topological descriptors (based on graph topology/structure):

.. autosummary::
    :nosignatures:
    :toctree: generated/

    average_wiener_index
    graph_distance_index
    polarity_number
    wiener_index
    zagreb_index
