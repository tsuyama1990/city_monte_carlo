"""Tomoyuki Tsuyama, Base MC Tool."""

import numpy as np
import pandas as pd
from ase.geometry import get_layers


class OrderParameterFCC:
    """Calculate local rotationally invariant order parameters (LROP) for FCC structures.

    Attributes:
    base_atoms (ase.Atoms): The base structure of atoms.
    ele2val (dict): Dictionary mapping element symbols to numerical values for LROP calculation.
    eps (float): Tolerance for determining proximity in LROP calculations.
    """

    def __init__(self, base_atoms, attention_element_list, eps=0.1):
        """Initialize the OrderParameterFCC class.

        Parameters:
        base_atoms (ase.Atoms): The base structure of atoms.
        attention_element_list (list): A list of two elements for calculating order parameters.
        eps (float): Tolerance for determining proximity.
        """
        self.base_atoms = base_atoms
        assert len(attention_element_list) == 2, "Select 2 elements to calculate the order parameters."
        self.ele2val = {attention_element_list[0]: -1, attention_element_list[1]: 1}
        self.eps = eps

    def make_pbc(self, atoms, miller):
        """Make periodic boundary conditions (PBC) compatible for atoms.

        Parameters:
        atoms (ase.Atoms): The structure of atoms to check.
        miller (list): Miller indices for layer calculation.

        Returns:
        int: Factor for periodicity.
        """
        layer_id, _ = get_layers(atoms, miller=miller, tolerance=0.01)
        layer_id_base_atoms, _ = get_layers(self.base_atoms, miller=miller, tolerance=0.01)

        if (layer_id.max() + 1) % (layer_id_base_atoms.max() + 1) != 0:
            raise ValueError("Supercell structures are not identical to the repeat of base atoms.")

        factor = (layer_id.max() + 1) // (layer_id_base_atoms.max() + 1)
        return factor

    def get_repeat_factors(self, atoms):
        """Get repeat factors for atoms based on Miller indices.

        Parameters:
        atoms (ase.Atoms): The structure of atoms.

        Returns:
        np.array: Repeat factors for each Miller index.
        """
        millers = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        repeat_factors = [self.make_pbc(atoms=atoms, miller=miller) for miller in millers]
        return np.array(repeat_factors)

    def get_lrop(self, atoms):
        """Get local rotationally invariant order parameters (LROP) for atoms.

        Parameters:
        atoms (ase.Atoms): The structure of atoms.

        Returns:
        np.array: LROP values.
        """
        atoms.cell = self.base_atoms.cell.copy()
        repeat_factors = self.get_repeat_factors(atoms)
        atoms.cell = self.base_atoms.repeat(repeat_factors).cell.copy()
        scaled_positions = atoms.get_scaled_positions() * repeat_factors
        pos_cell = np.round(scaled_positions - 0.45, 0)  # round 0.95 - 1.95 to one
        pos_internal = np.round(scaled_positions - pos_cell, 3)
        masks = [
            np.linalg.norm(pos_internal - ori_sublat, axis=1) < self.eps
            for ori_sublat in self.base_atoms.get_scaled_positions()
        ]
        m = np.zeros(4)

        for i, mask in enumerate(masks):
            m[i] = np.array([self.ele2val[atom] for atom in atoms[mask].symbols]).sum() / len(atoms)

        coef = [[1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]]
        xi = np.dot(coef, np.array([m[0], m[1], m[2], m[3]]).reshape(4, 1))
        return xi.reshape(1, 3)[0]

    def lro_generator(self, traj):
        """Generate LROP values for each structure in a trajectory.

        Parameters:
        traj (iterable): An iterable of atomic structures.

        Yields:
        np.array: LROP values for each structure.
        """
        for atoms in traj:
            yield self.get_lrop(atoms)

    def build_mini_dataframe(self, traj, temperature):
        """Build a DataFrame of LROP values for a trajectory.

        Parameters:
        traj (iterable): An iterable of atomic structures.
        temperature (float): The temperature of the simulation.

        Returns:
        pd.DataFrame: DataFrame containing LROP values and temperature.
        """
        lros = list(self.lro_generator(traj))
        df = pd.DataFrame(lros, columns=["x", "y", "z"])
        df["temperature_K"] = temperature
        return df
