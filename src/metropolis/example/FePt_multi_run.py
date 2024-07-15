"""Tomoyuki Tsuyama, Base MC Tool."""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
from ase import Atom
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import read

from src.metropolis.forcefield_atom_swap import ForceFieldAtomSwap


def multi_mc_run(atoms, supercell, repeat, xyz_path, temps, calc, cutoff_cluster):
    """Perform a multi-temperature Monte Carlo simulation run.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    supercell : ase.Atoms
        Supercell configuration.
    repeat : int or array_like
        Repetition pattern for the supercell.
    xyz_path : pathlib.Path
        Path to write XYZ output files.
    temps : np.ndarray
        Array of temperatures in Kelvin.
    calc : ase.calculators.calculator.Calculator
        ASE calculator for simulations.
    cutoff_cluster : float
        Cutoff cluster distance in angstroms.
    """
    mc = ForceFieldAtomSwap(
        original_atoms=atoms,
        calc=calc,
        cutoff_cluster_A=cutoff_cluster,
        op_elements_list=["Fe", "Pt"],
        eps_distance_A=0.1,
        pbc=True,
    )

    ele_arr = mc.get_initial_coordinate(repeat=repeat, supercell=supercell)

    i = 0
    for temp in temps:
        append = False if i == 0 else True
        mc.run_mc_simulation(
            ele_arr,
            temp=temp,
            steps=1e4,
            xyz_file_path=xyz_path,
            append=append,
            save_freq=1e4,
            swap_centre_ele="Fe",
            swap_surround_ele="Pt",
        )
        i += 1


def build_supercell(repeat, cif_path, cutoffs):
    """Generate supercells based on CIF files and repetition patterns.

    Parameters
    ----------
    repeat : int or array_like
        Repetition pattern for the supercell.
    cif_path : pathlib.Path
        Path to CIF files.
    cutoffs : list
        List of cutoff distances.

    Yields:
    ------
    tuple
        Tuple containing supercell type, ASE atoms object, repeated supercell,
        XYZ file path, and cutoff distance.
    """
    supercell_types = ["FePt", "Fe3Pt", "FePt3"]
    for supercell_type in supercell_types:
        for cutoff in cutoffs:
            atoms = read(cif_path / f"{supercell_type}.cif")
            supercell = atoms.repeat(repeat)
            xyz_file_path = (
                Path(__file__).resolve().parent / "001_first_test" / f"{cutoff}" / f"{supercell_type}_cube.xyz"
            )
            xyz_file_path.parent.mkdir(exist_ok=True, parents=True)
            yield supercell_type, atoms, supercell, xyz_file_path, cutoff


def build_calculator(md_file_path):
    """Build a LAMMPS calculator for molecular dynamics simulations.

    Parameters
    ----------
    md_file_path : pathlib.Path
        Path to the directory containing LAMMPS potential files.

    Returns:
    -------
    ase.calculators.lammpslib.LAMMPSlib
        LAMMPS calculator instance.
    """
    print(md_file_path)
    lammps_command = [
        "pair_style meam/c",
        f"pair_coeff * * {(md_file_path / 'PtFelibrary.meam')} Pt Fe {(md_file_path / 'PtFe.meam')} Pt Fe",
        f"mass 1 {Atom('Pt').mass}",
        f"mass 2 {Atom('Fe').mass}",
    ]

    calc = LAMMPSlib(lmpcmds=lammps_command, atoms_types={"Fe": 1, "Pt": 2}, keep_alive=True)
    return calc


if __name__ == "__main__":
    temps_low = np.arange(500, 1400, 50)
    temps_middle = np.arange(1400, 1800, 5)
    temps_high = np.arange(1800, 3000, 50)
    temps = np.concatenate([temps_low, temps_middle, temps_high])

    rpts = np.array([18, 18, 18])
    calc = build_calculator(Path().home() / "phys_basefiles" / "md")

    cif_path = Path().home() / "phys_basefiles" / "cifs"

    p = [
        (atoms, supercell, rpts, xyz_file_path, temps, calc, cutoff)
        for _, atoms, supercell, xyz_file_path, cutoff in build_supercell(rpts, cif_path, [4])
    ]

    with Pool() as pool:
        pool.starmap(multi_mc_run, p)
