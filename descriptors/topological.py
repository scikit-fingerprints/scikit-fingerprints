from collections import Counter
from typing import Optional

import numpy as np
from rdkit.Chem import GetDistanceMatrix, rdmolops


class TopologicalDescriptors:
    def __init__(self, mol, distance_matrix: Optional[np.ndarray] = None):
        if mol is None:
            raise ValueError("Mol cannot be None")
        self.mol = mol
        if distance_matrix is None:
            self.distance_matrix = GetDistanceMatrix(mol)
        else:
            self.distance_matrix = distance_matrix

    def weiner_index(self) -> int:
        """Calculates the Wiener index of a molecule.

        Args:
            mol: RDKit Mol object.
            distance_matrix: Precomputed distance matrix. If None, it will be calculated.

        Returns:
            int: Wiener index.
        """
        return np.sum(self.distance_matrix) // 2

    def average_weiner_index(self) -> float:
        """Calculates the average Wiener index of a molecule.

        Args:
            mol: RDKit Mol object.

        Returns:
            float: Average Wiener index.
        """
        wiener_index = self.weiner_index()
        num_atoms = self.mol.GetNumAtoms()
        return (2 * wiener_index) / (num_atoms * (num_atoms - 1))

    def graph_distance_index(self) -> float:
        """Calculates the Graph Distance Index (GDI) of a molecule.

        Args:
            mol: RDKit Mol object.

        Returns:
            float: Graph Distance Index.
        """
        distances = self.distance_matrix[
            np.triu_indices_from(self.distance_matrix, k=1)
        ]
        distance_counts = Counter(distances)
        return sum((k * f) ** 2 for k, f in distance_counts.items())

    def zagreb_index(self) -> float:
        """Calculates the first Zagreb index of a molecule.

        Args:
            mol: RDKit Mol object.

        Returns:
            float: First Zagreb index.
        """
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(self.mol)
        degree_sequence = adjacency_matrix.sum(axis=1)
        return sum(degree_sequence**2)

    def polarity_number(self) -> int:
        """Calculates the Polarity Number of a molecule.

        Args:
            mol: RDKit Mol object.

        Returns:
            int: Polarity Number.
        """
        return (self.distance_matrix == 3).sum() // 2
