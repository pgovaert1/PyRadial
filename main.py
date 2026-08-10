#!/usr/bin/env python


import json
import argparse

from algorithms.dirac_numerical_radial import Visualize, Generate_Fermi_Data
from algorithms.radial_data_handler import calc_double_beta_decay_spectrum, compare_G_factors_across_potentials
from algorithms.grid_creation import plot_grid_stepsize
from configurations.isotopes import ISOTOPES


def positive_float(value):
    """
    argparse type-check: parse `value` as a float and reject negative numbers.

    Parameters
    ----------
    value : str
        Raw CLI argument string.

    Returns
    -------
    float
        Parsed non-negative value.
    """
    val = float(value)
    if val < 0:
        raise argparse.ArgumentTypeError("Kinetic energy must be >= 0")
    return val

def positive_int(value):
    """
    argparse type-check: parse `value` as an int and reject negative numbers.

    Parameters
    ----------
    value : str
        Raw CLI argument string.

    Returns
    -------
    int
        Parsed non-negative value.
    """
    val = int(value)
    if val < 0:
        raise argparse.ArgumentTypeError("Number must be positive")
    return val

def clean_name(name):
    """
    argparse type-check: sanitize a user-supplied name for use as a directory name.

    Strips leading/trailing whitespace and replaces internal spaces with underscores.

    Parameters
    ----------
    name : str
        Raw CLI argument string.

    Returns
    -------
    str
        Sanitized name safe to use as a directory component.
    """
    name = name.strip()
    name = name.replace(" ", "_")
    return name


def main():
    """
    CLI entry point. Reads `configurations/config.json` for defaults, applies
    CLI argument overrides, builds the per-isotope `cnf` dict from
    `configurations/isotopes.py`, and dispatches to the selected mode
    (`generate`, `plot`, `data`, `all`, or the standalone `--G_comparison` action).
    """
    with open("configurations/config.json") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="Run double beta decay code")

    #Generator parses
    parser.add_argument("--mode", type = str, choices=["generate", "plot", "data", "all"],  help = f"Run mode: 'generate', 'plot', 'data', or 'all' (generate + data in sequence), (default from config: { config["generator"]["mode"]})")
    parser.add_argument("--num_samples", type = positive_int, help = f"Number of energy samples generated, (default from config: {config["generator"]["num_samples"]})")
    parser.add_argument("--plot_energy", type = positive_float, help = f"Kinetic energy in MeV for single plot, (default from config: {config["generator"]["T_plot_energy-MeV"]})")
    parser.add_argument("--potential", type = int, choices = [0,1,2,3], help = f"Select potential function, 0: Z/r , 1: Z/r + V0 exp(-Ar) 2: (for r<R); Z/2R (3-(r/R)^2) (for r >= R); Z/r , 3: Thomas-Fermi  , (default from config: {config["generator"]["potential_index"]})")
    parser.add_argument("--Q" , type = positive_float, help = f"Energy differnce (Q), (default from config: {config["generator"]["T_end-MeV"]})")

    #Paths parses
    parser.add_argument("--output_dir", type = clean_name, help = f"Output directory name, (default from config: Dirac_[Isotope name][Atomic number]) ")

    #Parameter parses
    parser.add_argument("--Z", type = int, help = f"Atomic number (Z) given as posivite integer value, (default from config: {config["parameters"]["atomic_number"]})")
    parser.add_argument("--A", type = positive_int, help = f"Mass number (A), (default from config: {config["parameters"]["mass_number"]})")
    parser.add_argument("--angular_momentum", type = int, help = f"Angular momentum (l), (default from config: {config["parameters"]["angular_momentum_l"]})")
    parser.add_argument("--kappa", type = int, help = f"Kappa, (default from config: { config["parameters"]["kappa"]:+d} )")


    # Mesh creation
    parser.add_argument("--resolution", type = positive_int, help = f"Number of steps mesh is made of, (default the code will roughly calculate minimum resolution needed)")
    parser.add_argument("--DRN", type = positive_float, help = f"Upper limit step size grid, (default from config: {config["mesh_grid"]["upper_limit_step_size"]})")

    # Isotope selection
    parser.add_argument("--isotope", type = str, choices = list(ISOTOPES.keys()), help = f"select an isotope from the isotopes.py file for which to run the scripts, (default from config: { config["isotope"]})")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true",
        help="Enable verbose output: show/save all plots and print progress messages. "
             "Without this flag only data files and the results .txt are written.")
    parser.add_argument("--no-z0", action="store_true", dest="no_z0",
        help="Skip Z=0 free-particle wavefunction generation even when --verbose is set. "
             "Speeds up generate mode; Fermi division plot will be unavailable in data mode.")
    parser.add_argument("--G_comparison", action="store_true",
        help="Standalone action: write a G0/G2/G4/G22 comparison across potential "
             "schemes 0/2/3 to a .txt file, directly from existing NPZ files (no "
             "regeneration, no plots, no other analysis). Runs instead of --mode. "
             "Requires pre-generated NPZ data for each potential (run 'generate' "
             "with --potential 0/2/3 first).")

    # Plot selection (data mode only, only meaningful with --verbose)
    _ALL_PLOTS = ["gf", "fermi", "epsilon", "double_diff", "electron", "energy_diff", "grid"]
    parser.add_argument("--plots", nargs="*", choices=_ALL_PLOTS, metavar="PLOT",
        help=f"Which plots to show in data mode. "
             f"Choices: {_ALL_PLOTS}. "
             f"Pass '-v' with no '--plots' to run all; pass '--plots' with no args to run none.")


    args = parser.parse_args()



    # Generator overrides
    if args.mode:
        config["generator"]["mode"] = args.mode

    if args.num_samples:
        config["generator"]["num_samples"] = args.num_samples

    if args.plot_energy is not None:
        config["generator"]["T_plot_energy-MeV"] = args.plot_energy

    if args.potential is not None:
        config["generator"]["potential_index"] = args.potential

    if args.Q is not None:
        config["generator"]["T_end-MeV"] = args.Q

    # Paths overrides
    if args.output_dir:
        config["paths"]["output_directory"] = args.output_dir

    # Parameter overrides
    if args.Z is not None:
        config["parameters"]["atomic_number"] = args.Z

    if args.A is not None:
        config["parameters"]["mass_number"] = args.A

    if args.angular_momentum is not None:
        config["parameters"]["angular_momentum_l"] = args.angular_momentum

    if args.kappa is not None:
        config["parameters"]["kappa"] = args.kappa

    # Mesh creation overrides
    if args.resolution is not None:
        config["mesh_grid"]["num_mesh_steps"] = args.resolution

    if args.DRN is not None:
        config["mesh_grid"]["upper_limit_step_size"] = args.DRN


    # Isotope override
    if args.isotope is not None:
        config["isotope"] = args.isotope

    config["verbose"] = args.verbose
    config["generate_z0"] = args.verbose and not args.no_z0
    config["G_comparison"] = args.G_comparison

    # --plots given → use that selection; -v alone → all plots; neither → no plots
    if args.plots is not None:
        config["plots"] = args.plots
    elif args.verbose:
        config["plots"] = _ALL_PLOTS
    else:
        config["plots"] = []





    mode = config["generator"]["mode"]
    potential = config["generator"]["potential_index"]


    def print_potential(potential_index):
        """Print a one-line summary of which potential model and mode are running."""
        if potential_index == 0:
            print(f"calling '{mode}' function for potential 0: Z/r")
        elif potential_index == 1:
            print(f"calling '{mode}' function for potential 1: Z/r + V0 exp(-Ar)")
        elif potential_index == 2:
            print(f"calling '{mode}' function for potential 2: (for r<R); Z/2R (3-(r/R)^2) , (for r >= R); Z/r")
        elif potential_index == 3:
            print(f"calling '{mode}' function for potential 3: Thomas Fermi")




    iso = config["isotope"]
    isotope_data = ISOTOPES[iso]

    # Validate that nuclear matrix elements are available for this isotope
    if "nuclear_matrix_elements" not in isotope_data:
        raise ValueError(
            f"Nuclear matrix elements are not defined for isotope '{iso}'. "
            f"Please add nuclear_matrix_elements to isotopes.py for this isotope."
        )

    # Map potential_index to scheme letter for G/H reference value lookup
    _SCHEME_MAP = {0: "B", 2: "A", 3: "C"}
    scheme = _SCHEME_MAP.get(potential)

    # write a smaller config file needed for Dira_numer_radial.py
    cnf ={
        "Z": isotope_data["Z"],
        "A": isotope_data["A"],
        "Q": isotope_data["Q"],
        "Simkovic_Gs": isotope_data["G_values"].get(scheme) if scheme else None,
        "Simkovic_Hs": isotope_data["H_values"].get(scheme) if scheme else None,
        "G_values": isotope_data["G_values"],
        "nuclear_matrix_elements": isotope_data["nuclear_matrix_elements"],

        "angular_momentum_l": config["parameters"]["angular_momentum_l"],
        "kappa": config["parameters"]["kappa"]
        }

    
    if config["G_comparison"]:
        compare_G_factors_across_potentials(config, cnf)
        return

    if args.verbose:
        print_potential(potential)

    if mode == "data" and config["plots"] == ["grid"]:
        plot_grid_stepsize(config["paths"]["output_directory"])
        return

    if mode == "generate":
        if args.verbose:
            print(f"Saving output in {config['paths']['output_directory']} directory")
        Generate_Fermi_Data(config, cnf)
    elif mode == "plot":
        Visualize(config, cnf)
    elif mode == "data":
        calc_double_beta_decay_spectrum(config, cnf)
    elif mode == "all":
        if args.verbose:
            print(f"Saving output in {config['paths']['output_directory']} directory")
        Generate_Fermi_Data(config, cnf)
        calc_double_beta_decay_spectrum(config, cnf)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
