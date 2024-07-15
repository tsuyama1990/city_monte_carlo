"""Tomoyuki Tsuyama, Base MC Tool."""

import csv
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from ase import Atom, Atoms
from ase.constraints import FixAtoms
from ase.geometry import get_layers
from ase.io import write
from numba import jit

from src.utils.analysis import OrderParameterFCC

# setting for logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class BaseAtomSwap(ABC):
    """Base class for atom swapping operations in molecular dynamics simulations.

    Parameters
    ----------
    original_atoms : ase.Atoms
        The original atomic configuration.
    boltzman_const_eV_per_K : float, optional
        Boltzmann constant in eV/K. Default is 8.61733e-5 eV/K.
    eps_distance_A : float, optional
        Epsilon distance in angstroms for distance calculations. Default is 0.1.
    pbc : bool, optional
        Periodic boundary conditions. Default is True.
    op_elements_list : list, optional
        List of elements for order parameter calculation.
    empty_int : int, optional
        Integer representing empty space. Default is 0.
    nn_distance_A : float, optional
        Nearest neighbor distance in angstroms. Default is 3.
    steps_done : int, optional
        The number of steps done in the simulation. Default is 0.
    """

    def __init__(
        self,
        original_atoms,
        boltzman_const_eV_per_K=None,
        eps_distance_A=0.1,
        pbc=True,
        op_elements_list=None,
        empty_int=0,
        nn_distance_A=3,
        steps_done=0,
    ):
        """Base class for atom swapping operations in molecular dynamics simulations.

        Parameters
        ----------
        original_atoms : ase.Atoms
            The original atomic configuration.
        boltzman_const_eV_per_K : float, optional
            Boltzmann constant in eV/K. Default is 8.61733e-5 eV/K.
        eps_distance_A : float, optional
            Epsilon distance in angstroms for distance calculations. Default is 0.1.
        pbc : bool, optional
            Periodic boundary conditions. Default is True.
        op_elements_list : list, optional
            List of elements for order parameter calculation.
        empty_int : int, optional
            Integer representing empty space. Default is 0.
        nn_distance_A : float, optional
            Nearest neighbor distance in angstroms. Default is 3.
        steps_done : int, optional
            The number of steps done in the simulation. Default is 0.
        """
        self.original_atoms = original_atoms
        self.original_atoms.positions -= self.original_atoms.positions.min()
        self.empty_int = empty_int
        self.nn_distance_A = nn_distance_A

        if boltzman_const_eV_per_K is not None:
            self.k_b = boltzman_const_eV_per_K
        else:
            self.k_b = 8.61733e-5

        self.eps_distance_A = eps_distance_A
        self.rng = np.random.default_rng()
        self.pbc = pbc
        self.op_elements_list = op_elements_list
        if self.op_elements_list is not None:
            self.op = OrderParameterFCC(
                base_atoms=self.original_atoms,
                attention_element_list=self.op_elements_list,
            )
        self.steps_done = steps_done

    def phase_shift_fcc(self, supercell, rpts, num_samplings=100):
        """Shift the phase of FCC atoms arrangement if it does not match with grid points.

        Parameters
        ----------
        supercell : ase.Atoms
            The supercell atomic configuration.
        rpts : array_like
            Repetition pattern for the supercell.
        num_samplings : int, optional
            Number of samplings to check. Default is 100.
        """
        supercell_pos = supercell.positions
        grid_pos = self.original_atoms.repeat(rpts).positions

        id_rnd_sample_from_supercell = self.rng.choice(range(supercell_pos.shape[0]), num_samplings)
        supercell_pos_sampled = supercell_pos[id_rnd_sample_from_supercell]

        def queries_exist_generator(supercell_pos_sampled, grid_pos, eps_distance_A):
            for point in supercell_pos_sampled:
                query_exist = np.any(np.linalg.norm(grid_pos - point, axis=1) < eps_distance_A) is np.True_
                yield query_exist

        queries_exist = list(queries_exist_generator(supercell_pos_sampled, grid_pos, self.eps_distance_A))
        query_all_exist = np.all(queries_exist)

        if not query_all_exist:
            logging.info("No match grid points and supercell positions. Try to shift 1 layer for grid points.")
            original_atoms_1st_layer_x = get_layers(self.original_atoms, miller=[1, 0, 0])[1][1]
            self.original_atoms.positions[:, 0] -= original_atoms_1st_layer_x
            self.original_atoms.wrap()

    def get_initial_coordinate(self, repeat, supercell):
        """Get initial coordinates for the atomic configuration.

        Parameters
        ----------
        repeat : int or array_like
            Repetition pattern for the atomic configuration.
        supercell : ase.Atoms
            The supercell atomic configuration.

        Returns:
        -------
        ele_arr : np.ndarray
            Array of atomic numbers representing the initial coordinates.
        """
        if isinstance(repeat, int):
            repeat = np.array([repeat, repeat, repeat]).astype(int)

        supercell.positions -= supercell.positions.min()

        # atoms_number array
        atoms_numbers = supercell.get_atomic_numbers()
        fix_atoms_indices_arr = self.get_constraint_indices(supercell, get_bool=True)

        # if the phase of FCC atoms arrangement does not match with gridpoints, shift them by 1 layer
        self.phase_shift_fcc(supercell, repeat)
        grid_points = self.original_atoms.repeat(repeat).positions

        grid_atoms_list = []
        fix_atoms_list = []

        for _id, point in enumerate(grid_points):
            bool_grid_match = np.linalg.norm(supercell.positions - point, axis=1) < self.eps_distance_A
            if np.any(bool_grid_match) is np.True_:
                grid_atoms_list.append(atoms_numbers[bool_grid_match][0])
                if np.any(np.all([bool_grid_match, fix_atoms_indices_arr], axis=0)) is np.True_:
                    fix_atoms_list.append(_id)

            else:
                grid_atoms_list.append(self.empty_int)

        ele_arr = np.array(grid_atoms_list)
        self.grid_points = grid_points.astype(float)

        self.fix_atoms_indices_arr = np.array(fix_atoms_list)

        self.ele_set = list(OrderedDict.fromkeys(grid_atoms_list))

        if self.empty_int in self.ele_set:
            self.ele_set.remove(self.empty_int)

        self.original_cell = self.original_atoms.repeat(repeat).cell.array.copy()
        self.original_cell_inv = np.linalg.inv(self.original_cell)
        self._id = np.array(range(len(ele_arr)))

        return ele_arr

    def get_constraint_indices(self, supercell, get_bool=False):
        """Get indices of fixed atoms.

        Parameters
        ----------
        supercell : ase.Atoms
            The supercell atomic configuration.
        get_bool : bool, optional
            Whether to return boolean array. Default is False.

        Returns:
        -------
        fix_atoms : np.ndarray
            Indices of fixed atoms.
        fix_atoms_bool : np.ndarray
            Boolean array of fixed atoms.
        """
        const_list = supercell.constraints
        if const_list:
            # fix_atoms_instance = [i for i in const_list if isinstance(i, FixAtoms)][0]
            fix_atoms_instance = next(i for i in const_list if isinstance(i, FixAtoms))
            fix_atoms = fix_atoms_instance.index
            fix_atoms_bool = np.isin(range(len(supercell)), fix_atoms)

        else:
            fix_atoms = np.array([])
            fix_atoms_bool = np.array([False for _ in range(len(supercell))])

        if get_bool is False:
            return fix_atoms
        else:
            return fix_atoms_bool

    @abstractmethod
    def single_swap(self, ele_arr, swap_centre_ele, swap_surround_ele, temp):
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
        pass

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_neighbors(grid_points, centre, cell, inv_cell, pbc):
        """Get the distances to neighboring atoms.

        Parameters
        ----------
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

        Returns:
        -------
        distances : np.ndarray
            Distances to neighboring atoms.
        """
        delta = grid_points - centre

        # update the diffs in displacements considering the PBC across boundary
        if pbc:
            delta -= np.round(delta @ inv_cell) @ cell

        distances = np.sqrt(np.sum((delta**2), axis=1))

        return distances

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_swap_init_id(ele_arr, swap_centre_ele, fix_atoms_indices_arr):
        """Get the initial index for swapping.

        Parameters
        ----------
        ele_arr : np.ndarray
            Array of atomic numbers.
        swap_centre_ele : int
            Atomic number of the center element to be swapped.
        fix_atoms_indices_arr : np.ndarray
            Indices of fixed atoms.

        Returns:
        -------
        swap_init_id : int
            Initial index for swapping.
        """
        indices_swap_centre_ele = np.where(ele_arr == swap_centre_ele)[0]
        # indices_swap_centre_ele_const = np.setdiff1d(indices_swap_centre_ele, fix_atoms_indices_arr)
        indices_swap_centre_ele_const = indices_swap_centre_ele.copy()
        swap_init_id = np.random.choice(indices_swap_centre_ele_const)
        return swap_init_id

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_swap_term_id(
        ele_arr,
        swap_surround_ele,
        distance,
        nn_distance_criteria_A,
        fix_atoms_indices_arr,
    ):
        """Get the indices of atoms eligible for swapping based on distance and element criteria.

        Parameters
        ----------
        ele_arr : np.ndarray
            Array of atomic numbers.
        swap_surround_ele : int
            Atomic number of the surrounding element to be swapped.
        distance : np.ndarray
            Array of distances to neighboring atoms.
        nn_distance_criteria_A : float
            Nearest neighbor distance criteria in angstroms.
        fix_atoms_indices_arr : np.ndarray
            Indices of fixed atoms.

        Returns:
        -------
        swap_term_ids : np.ndarray
            Indices of atoms eligible for swapping.
        """
        bool_ele = ele_arr == swap_surround_ele
        bool_nn = distance < nn_distance_criteria_A
        swap_term_ids_bool = bool_nn & bool_ele
        swap_term_ids = np.where(swap_term_ids_bool)[0]
        return swap_term_ids

    def get_swap_indices(self, ele_arr, swap_centre_ele, swap_surround_ele, grid_points):
        """Get the initial and target indices for atom swapping.

        Parameters
        ----------
        ele_arr : np.ndarray
            Array of atomic numbers.
        swap_centre_ele : int
            Atomic number of the center element to be swapped.
        swap_surround_ele : int
            Atomic number of the surrounding element to be swapped.
        grid_points : np.ndarray
            Array of grid points representing atomic positions.

        Returns:
        -------
        swap_init_id : int
            Initial index for swapping.
        swap_term_ids : np.ndarray
            Indices of atoms eligible for swapping.
        """
        swap_init_id = self.get_swap_init_id(ele_arr, swap_centre_ele, self.fix_atoms_indices_arr)

        distance = self.get_neighbors(
            grid_points=grid_points,
            centre=grid_points[swap_init_id],
            cell=self.original_cell,
            inv_cell=self.original_cell_inv,
            pbc=self.pbc,
        )

        swap_term_ids = self.get_swap_term_id(
            ele_arr,
            swap_surround_ele,
            distance,
            self.nn_distance_A,
            self.fix_atoms_indices_arr,
        )

        return swap_init_id, swap_term_ids

    @abstractmethod
    def get_delta_e(self) -> None:
        """Calculate the energy difference for the swap.

        Returns:
        -------
        delta_e : float
            Energy difference for the swap.
        """
        pass

    @staticmethod
    @jit(nopython=True, cache=True)
    def evaluate_metropolis(
        ele_arr,
        temp,
        delta_e,
        swap_init_id,
        swap_term_id,
        k_b,
        swap_centre_ele,
        swap_surround_ele,
        rng,
    ):
        """Evaluate the Metropolis criterion for acceptance of the swap.

        Parameters
        ----------
        ele_arr : np.ndarray
            Array of atomic numbers.
        temp : float
            Temperature in Kelvin.
        delta_e : float
            Energy difference for the swap.
        swap_init_id : int
            Index of the initial atom to be swapped.
        swap_term_id : int
            Index of the target atom to be swapped.
        k_b : float
            Boltzmann constant in eV/K.
        swap_centre_ele : int
            Atomic number of the center element to be swapped.
        swap_surround_ele : int
            Atomic number of the surrounding element to be swapped.
        rng : np.random.Generator
            Random number generator.

        Returns:
        -------
        accept : bool
            Whether the swap is accepted.
        """
        if rng.random() < np.exp(-delta_e / (temp * k_b)):
            ele_arr[swap_init_id] = swap_surround_ele
            ele_arr[swap_term_id] = swap_centre_ele
            accept = True

        else:
            accept = False

        return accept

    def write_atoms_obj(self, atoms, output_file_path, append=False):
        """Write atomic configuration to a file.

        Parameters
        ----------
        output_file_path : pathlib.Path
            Path to the output file.
        ele_arr : np.ndarray
            Array of atomic numbers.
        freq : int
            Frequency of writing the output.
        temperature : float
            Temperature in Kelvin.
        append : bool, optional
            Whether to append to the file. Default is False.
        """
        if (self.steps_done == 0) & (append is False):
            _append = False
        else:
            _append = True
        write(output_file_path, atoms, append=_append)

    def write_lro(self, atoms, output_file_path, temperature, append=False):
        """Write atomic configuration to a file.

        Parameters
        ----------
        output_file_path : pathlib.Path
            Path to the output file.
        ele_arr : np.ndarray
            Array of atomic numbers.
        freq : int
            Frequency of writing the output.
        temperature : float
            Temperature in Kelvin.
        append : bool, optional
            Whether to append to the file. Default is False.
        """
        if hasattr(self, "op"):
            if (self.steps_done == 0) & (append is False):
                _append = False
            else:
                _append = True
            lro = self.op.get_lrop(atoms).tolist()
            # write LRO to a csv file
            mode = "a" if _append else "w"
            with open(output_file_path, mode=mode, newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not _append:
                    writer.writerow(["step", "temperature", "x", "y", "z"])
                # data = [self.steps_done, temperature] + lro
                data = [self.steps_done, temperature, *lro]
                writer.writerow(data)

    def write_files(
        self,
        xyz_file_path,
        append,
        ele_arr,
        xyz_freq,
        lro_freq,
        lro_file_path,
        temperature,
    ):
        """Write atomic configurations and order parameters to files based on specified frequencies.

        Parameters
        ----------
        xyz_file_path : pathlib.Path
            Path to the XYZ output file for atomic configurations.
        append : bool
            Whether to append to the XYZ file if it exists.
        ele_arr : np.ndarray
            Array of atomic numbers.
        xyz_freq : int
            Frequency of writing the XYZ output.
        lro_freq : int
            Frequency of writing the LRO output.
        lro_file_path : pathlib.Path
            Path to the LRO output file.
        temperature : float
            Temperature in Kelvin.
        """
        trigger_xyz_write = (self.steps_done // xyz_freq) - (self._last_write_xyz // xyz_freq) >= 1
        trigger_lro_write = (self.steps_done // lro_freq) - (self._last_write_lro // xyz_freq) >= 1
        if (trigger_xyz_write) | (trigger_lro_write):
            atoms = Atoms(
                numbers=ele_arr[ele_arr != self.empty_int],
                positions=self.grid_points[ele_arr != self.empty_int],
                cell=self.original_cell,
            )
        if trigger_xyz_write:
            self.write_atoms_obj(atoms, xyz_file_path, append=append)
            self._last_write_xyz = self.steps_done

        if trigger_lro_write:
            self.write_lro(atoms, lro_file_path, temperature, append=append)
            self._last_write_lro = self.steps_done

    def run_mc_simulation(
        self,
        ele_arr,
        temp,
        steps,
        xyz_file_path,
        append,
        save_freq,
        swap_centre_ele,
        swap_surround_ele,
    ):
        """Run Monte Carlo simulation.

        Parameters
        ----------
        ele_arr : np.ndarray
            Array of atomic numbers.
        temp : float
            Temperature in Kelvin.
        steps : int
            Number of simulation steps.
        output_file_path : pathlib.Path
            Path to the output file.
        append : bool
            Whether to append to the file.
        save_freq : int
            Frequency of saving the output.
        swap_centre_ele : str
            Symbol of the center element to be swapped.
        swap_surround_ele : str
            Symbol of the surrounding element to be swapped.
        """
        self._last_write_xyz = self.steps_done - 1
        self._last_write_lro = self.steps_done - 1

        swap_centre_ele = Atom(swap_centre_ele).number
        swap_surround_ele = Atom(swap_surround_ele).number
        start_time = time.time()
        steps = int(steps)
        _init_step = self.steps_done

        while self.steps_done < _init_step + steps:
            lro_file_path = xyz_file_path.parent / xyz_file_path.name.replace(".xyz", "_lro.csv")
            self.write_files(
                xyz_file_path,
                append,
                ele_arr,
                xyz_freq=save_freq,
                lro_freq=save_freq,
                lro_file_path=lro_file_path,
                temperature=temp,
            )

            self.single_swap(
                ele_arr,
                swap_centre_ele=swap_centre_ele,
                swap_surround_ele=swap_surround_ele,
                temp=temp,
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Simulation in {steps} steps at {temp} K took {elapsed_time:.2f} seconds.")
        logger.info(f"{int(steps/elapsed_time)} steps per second.")
