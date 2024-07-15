"""Tomoyuki Tsuyama, Base MC Tool."""

import logging

import numpy as np
from ase import Atoms
from numba import jit

from src.metropolis.base_atom_swap import BaseAtomSwap

# setting for logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ForceFieldAtomSwap(BaseAtomSwap):
    """Class for performing atom swaps using a force field.

    Parameters
    ----------
    original_atoms : ase.Atoms
        The original atomic configuration.
    calc : object
        Calculator for potential energy calculations.
    cutoff_cluster_A : float
        Cutoff distance for cluster determination in angstroms.
    boltzman_const_eV_per_K : float, optional
        Boltzmann constant in eV/K. Default is None.
    eps_distance_A : float, optional
        Epsilon distance in angstroms for distance calculations. Default is 0.5.
    empty_int : int, optional
        Integer representing empty space. Default is 0.
    pbc : bool, optional
        Periodic boundary conditions. Default is True.
    op_elements_list : list, optional
        List of elements for order parameter calculation.
    """

    def __init__(
        self,
        original_atoms,
        calc,
        cutoff_cluster_A,
        boltzman_const_eV_per_K=None,
        eps_distance_A=0.5,
        empty_int=0,
        pbc=True,
        op_elements_list=None,
    ):
        """Class for performing atom swaps using a force field.

        Parameters
        ----------
        original_atoms : ase.Atoms
            The original atomic configuration.
        calc : object
            Calculator for potential energy calculations.
        cutoff_cluster_A : float
            Cutoff distance for cluster determination in angstroms.
        boltzman_const_eV_per_K : float, optional
            Boltzmann constant in eV/K. Default is None.
        eps_distance_A : float, optional
            Epsilon distance in angstroms for distance calculations. Default is 0.5.
        empty_int : int, optional
            Integer representing empty space. Default is 0.
        pbc : bool, optional
            Periodic boundary conditions. Default is True.
        op_elements_list : list, optional
            List of elements for order parameter calculation.
        """
        logger.info("Force-Field mode is chosen.")
        logger.info(f"Force Field : {calc}")
        super().__init__(
            original_atoms,
            empty_int=empty_int,
            boltzman_const_eV_per_K=boltzman_const_eV_per_K,
            pbc=pbc,
            op_elements_list=op_elements_list,
            eps_distance_A=eps_distance_A,
        )
        self.calc = calc
        self.cutoff_cluster = float(cutoff_cluster_A)

    @staticmethod
    @jit(nopython=True, cache=True)
    def cut_cluster(
        ele_arr,
        swap_init_id,
        swap_term_id,
        grid_points,
        cutoff_cluster,
        empty_int,
        cell,
        inv_cell,
        pbc,
    ):
        """Cut a cluster of atoms around the swap center and target atoms.

        Parameters
        ----------
        ele_arr : np.ndarray
            Array of atomic numbers.
        swap_init_id : int
            Index of the initial atom to be swapped.
        swap_term_id : int
            Index of the target atom to be swapped.
        grid_points : np.ndarray
            Array of grid points.
        cutoff_cluster : float
            Cutoff distance for cluster determination in angstroms.
        empty_int : int
            Integer representing empty space.
        cell : np.ndarray
            Cell dimensions.
        inv_cell : np.ndarray
            Inverse cell dimensions.
        pbc : bool
            Periodic boundary conditions.

        Returns:
        -------
        cluster_ele_arr : np.ndarray
            Array of atomic numbers in the cluster.
        cluster_pos : np.ndarray
            Array of positions in the cluster.
        cluster_indices : np.ndarray
            Array of indices in the cluster.
        """

        def get_neighbors(grid_points, centre, cell, inv_cell, pbc, cutoff):
            delta = grid_points - centre

            if pbc:
                delta -= np.round(delta @ inv_cell) @ cell

            distance = np.sqrt(np.sum((delta) ** 2, axis=1))

            return distance <= cutoff

        def make_continuous(grid_points, centre, cell, inv_cell, pbc):
            if pbc:
                delta = grid_points - centre
                nn_points = grid_points - np.round(delta @ inv_cell) @ cell

            else:
                nn_points = grid_points

            return nn_points

        dist_centre_bool = get_neighbors(grid_points, grid_points[swap_init_id], cell, inv_cell, pbc, cutoff_cluster)
        dist_surround_bool = get_neighbors(grid_points, grid_points[swap_term_id], cell, inv_cell, pbc, cutoff_cluster)
        cluster_bool = dist_centre_bool | dist_surround_bool

        # Filter out empty sites
        filtered_ele_arr = ele_arr != empty_int
        cluster_indices = np.where(cluster_bool & filtered_ele_arr)[0]

        cluster_ele_arr = ele_arr[cluster_indices]

        swap_centre = (grid_points[swap_init_id] + grid_points[swap_term_id]) / 2

        cluster_pos = make_continuous(grid_points[cluster_indices], swap_centre, cell, inv_cell, pbc)

        min_vals = np.zeros(cluster_pos.shape[1])
        for i in range(cluster_pos.shape[1]):
            min_vals[i] = np.min(cluster_pos[:, i])
        cluster_pos -= min_vals

        return cluster_ele_arr, cluster_pos, cluster_indices

    def single_swap(
        self,
        ele_arr,
        swap_centre_ele,
        swap_surround_ele,
        temp,
    ):
        """Perform a single swap of atoms.

        Parameters
        ----------
        ele_arr : np.ndarray
            Array of atomic numbers.
        swap_centre_ele : int
            Atomic number of the center element to be swapped.
        swap_surround_ele : int
            Atomic number of the surrounding element to be swapped.
        temp : float
            Temperature in Kelvin.
        """
        swap_init_id, swap_term_ids = self.get_swap_indices(
            ele_arr,
            swap_centre_ele,
            swap_surround_ele,
            self.grid_points,
        )
        if len(swap_term_ids) != 0:
            ele_arr_near_centre, grid_points_near_centre, id_converter = self.extract_nn(
                ele_arr,
                self.grid_points,
                self.grid_points[swap_init_id],
                self.original_cell,
                self.original_cell_inv,
                self.pbc,
                self.cutoff_cluster * 2.0,
                self._id,
            )
            self.rng.shuffle(swap_term_ids)
            for swap_term_id in swap_term_ids:
                delta_e = self.get_delta_e(
                    ele_arr_near_centre,
                    grid_points_near_centre,
                    id_converter[swap_init_id],
                    id_converter[swap_term_id],
                    swap_centre_ele,
                    swap_surround_ele,
                )

                accepted = self.evaluate_metropolis(
                    ele_arr,
                    temp,
                    delta_e,
                    swap_init_id,
                    swap_term_id,
                    self.k_b,
                    swap_centre_ele,
                    swap_surround_ele,
                    self.rng,
                )

                self.steps_done += 1

                if accepted:
                    break

    @staticmethod
    @jit(nopython=True, cache=True)
    def extract_nn(ele_arr, grid_points, centre, cell, inv_cell, pbc, cutoff, _id):
        """Extract nearest neighbors for the swap.

        Parameters
        ----------
        ele_arr : np.ndarray
            Array of atomic numbers.
        grid_points : np.ndarray
            Array of grid points.
        centre : np.ndarray
            Center point coordinates.
        cell : np.ndarray
            Cell dimensions.
        inv_cell : np.ndarray
            Inverse cell dimensions.
        pbc : bool
            Periodic boundary conditions.
        cutoff : float
            Cutoff distance for nearest neighbor determination in angstroms.
        _id : np.ndarray
            Array of indices.

        Returns:
        -------
        new_ele_arr : np.ndarray
            Array of atomic numbers in the nearest neighbor region.
        new_grid_points : np.ndarray
            Array of positions in the nearest neighbor region.
        id_converter : dict
            Dictionary mapping original indices to new indices.
        """
        delta = grid_points - centre
        if pbc:
            delta -= np.round(delta @ inv_cell) @ cell

        distance = np.sqrt(np.sum(delta**2, axis=1))

        mask = distance <= cutoff
        new_ele_arr = ele_arr[mask]
        new_grid_points = grid_points[mask]
        cluster_ids = _id[mask]

        id_converter = {original_id: new_id for new_id, original_id in enumerate(cluster_ids)}
        return new_ele_arr, new_grid_points, id_converter

    def get_delta_e(
        self,
        ele_arr,
        grid_points,
        swap_init_id,
        swap_term_id,
        swap_centre_ele,
        swap_surround_ele,
    ):
        """Calculate the energy difference for the swap.

        Parameters
        ----------
        ele_arr : np.ndarray
            Array of atomic numbers.
        grid_points : np.ndarray
            Array of grid points.
        swap_init_id : int
            Index of the initial atom to be swapped.
        swap_term_id : int
            Index of the target atom to be swapped.
        swap_centre_ele : int
            Atomic number of the center element to be swapped.
        swap_surround_ele : int
            Atomic number of the surrounding element to be swapped.

        Returns:
        -------
        delta_e : float
            Energy difference for the swap.
        """
        cluster_ele_arr, cluster_pos, cluster_indices = self.cut_cluster(
            ele_arr,
            swap_init_id,
            swap_term_id,
            grid_points=grid_points,
            cutoff_cluster=self.cutoff_cluster,
            empty_int=self.empty_int,
            cell=self.original_cell,
            inv_cell=self.original_cell_inv,
            pbc=self.pbc,
        )
        cluster_before = Atoms(numbers=cluster_ele_arr, positions=cluster_pos)
        cluster_after = cluster_before.copy()
        cluster_after.numbers[cluster_indices == swap_init_id] = swap_surround_ele
        cluster_after.numbers[cluster_indices == swap_term_id] = swap_centre_ele
        cluster_before.calc = self.calc
        cluster_after.calc = self.calc
        cluster_before.cell = np.diag(cluster_before.positions.max(axis=0) + 10)
        cluster_after.cell = np.diag(cluster_after.positions.max(axis=0) + 10)

        delta_e = cluster_after.get_potential_energy() - cluster_before.get_potential_energy()

        return delta_e
