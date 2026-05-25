#!/usr/bin/env python


import json
import argparse

from scipy.optimize import bracket
from algorithms.dirac_numerical_radial import Visualize, Generate_Fermi_Data
from algorithms.radial_data_generator import Calc_double_beta_decay_spectrum
from configurations.isotopes import ISOTOPES




def positive_float(value):
    val = float(value)
    if val < 0:
        raise argparse.ArgumentTypeError("Kinetic energy must be >= 0")
    return val

def positive_int(value):
    val = int(value)
    if val < 0:
        raise argparse.ArgumentTypeError("Number must be positive")
    return val

def clean_name(name):
    name = name.strip()
    name = name.replace(" ", "_")
    return name


def main():

    with open("configurations/config.json") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="Run double beta decay code")

    #Generator parses
    parser.add_argument("--mode", type = str, choices=["generate", "plot", "data"],  help = f"Run mode: 'generate' or 'plot' or 'data', (default from config: { config["generator"]["mode"]})")
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
    parser.add_argument("--isotope", type = str, choices = ["Xe136", "Mo100", "Nd150"], help = f"select an isotope from the isotopes.py file for which to run the scripts, (default from config: { config["isotope"]})")


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


    # Istopte override
    if args.isotope is not None:
        config["isotope"] = args.isotope





    mode = config["generator"]["mode"]
    potential = config["generator"]["potential_index"]


    if potential == 0:
        potential_name = "pure Coulomb potential V(r) = Z/r"
    elif potential == 1:
        potential_name = "Z/r + V0 exp(-Ar)"
    elif potential== 2:
       potential_name = "finite size Coulomb potential. (for r<R); V(r) = Z/2R * (3-(r/R)^2), (for r >= R); V(r) = Z/r"
    elif potential == 3:
        potential_name =  "Thomas Fermi potential"
    else:
        raise RuntimeError("Invaled potential index given")


    def print_potential(potential_index):
        if potential_index == 0:
            print(f"calling '{mode}' function for potential 0: Z/r")
        if potential_index == 1:
            print(f"calling '{mode}' function for potential 1: Z/r + V0 exp(-Ar)")
        if potential_index == 2:
            print(f"calling '{mode}' function for potential 2: (for r<R); Z/2R (3-(r/R)^2) (for r >= R); Z/r")
        if potential_index == 3:
            print(f"calling '{mode}' function for potential 3: Thomas Fermi")




    iso = config["isotope"]
    isotope_data = ISOTOPES[iso]


    # write a smaller config file needed for Dira_numer_radial.py
    cnf ={
        "Z": isotope_data["Z"],
        "A": isotope_data["A"],
        "Q": isotope_data["Q"],
        "Simkovic_Gs": isotope_data["G_values"],
        "Simkovic_Hs": isotope_data["H_values"],

        "angular_momentum_l": config["parameters"]["angular_momentum_l"],
        "kappa": config["parameters"]["kappa"]
        }


    if mode == "generate":
        print_potential(potential)
        print(f"Saving output in \\{config["paths"]["output_directory"]} directory")
        Generate_Fermi_Data(config,cnf)
    elif mode == "plot":
        print_potential(potential)
        Visualize(config,cnf)
    elif mode == "data":
        print_potential(potential)
        Calc_double_beta_decay_spectrum(config,cnf)
    elif mode == None:
        iso = config["isotope"]
        model = config["nuclear_model"]
        iso_data = ISOTOPES[iso]



        print(f"Generating data for {iso} using {model} model with {potential_name}")
        if config["paths"]["output_directory"] is None:
            print(f"Output is sent to Dirac_{config["isotope"]}")
        else:
            print(f"Output is sent to {config["paths"]["output_directory"]}")

        Generate_Fermi_Data(config,cnf)

        print(f"Generating data complete ")
        print(f"Calculating phase space results and plotting Fermi function and 2νββ Spectrum\n")
        Calc_double_beta_decay_spectrum(config,cnf)

        print("Done")




    else:
        raise ValueError("Unkown mode: {mode}")


if __name__ == "__main__":
    main()
