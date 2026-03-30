#!/usr/bin/env python


import json
import argparse
from Dirac_numerical_radial import Visualize, Generate_Fermi_Data
from Simkovic import Calc_double_beta_decay_spectrum

def positive_float(value):
    val = float(value)
    if val < 0:
        raise argparse.ArgumentTypeError("Kinetic energy must be >= 0")
    return val

def positive_int(value):
    val = int(value)
    if val < 0:
        raise argparse.ArgumentTypeError("Number of energy samples must be more than 0")
    return val

def clean_name(name):
    name = name.strip()
    name = name.replace(" ", "_")
    return name


def main():
    parser = argparse.ArgumentParser()

    #Generator parses
    parser.add_argument("--mode", type = str, choices=["generate", "plot", "data"],  help = "'generate' or 'plot' or 'data'")
    parser.add_argument("--num_samples", type = positive_int, help = "number of energy samples generated")
    parser.add_argument("--plot_energy", type = positive_float, help = "Kinetic energy in MeV for single plot")
    parser.add_argument("--potential", type = int, choices = [0,1,2], help = "Select potential function, 0: Z/r , 1: Z/r + V0 exp(-Ar) 2: (for r<R); Z/2R (3-(r/R)^2) (for r >= R); Z/r ")

    #Paths parses
    parser.add_argument("--output_dir", type = str, help = "Output directory name with standard: /out")

    #Parameter parses
    parser.add_argument("--atomic_num", type = positive_int, help = "Atomic number (Z) given as posivite integer value")
    parser.add_argument("--mass_number", type = positive_int, help = "Mass number (A)")
    parser.add_argument("--angular_momentum", type = int, help = "Angular momentum (l)")
    parser.add_argument("--kappa", type = int, help = "Kappa")


    args = parser.parse_args()


    with open("config.json") as f:
        config = json.load(f)

    #Generator overrides
    if args.mode:
        config["generator"]["mode"] = args.mode

    if args.num_samples:
        config["generator"]["num_samples"] = args.num_samples

    if args.plot_energy is not None:
        config["generator"]["T_plot_energy-MeV"] = args.plot_energy

    if args.potential is not None:
        config["generator"]["potential_index"] = args.potential

   #Paths overrides
    if args.output_dir is not None:
        config["paths"]["output_directory"] = clean_name(args.output_dir)

   #Parameter overrides
    if args.atomic_num is not None:
        config["parameters"]["atomic_number"] = args.atomic_number

    if args.mass_number is not None:
        config["parameters"]["mass_number"] = args.mass_number

    if args.angular_momentum is not None:
        config["parameters"]["angular_momentum_l"] = args.angular_momentum

    if args.kappa is not None:
        config["parameters"]["kappa"] = args.kappa






    mode = config["generator"]["mode"]
    potential = config["generator"]["potential_index"]
    if potential == 0:
        print(f"calling '{mode}' function for potential 0: Z/r")
    if potential == 1:
        print(f"calling '{mode}' function for potential 1: Z/r + V0 exp(-Ar)")
    if potential == 2:
        print(f"calling '{mode}' function for potential 2: (for r<R); Z/2R (3-(r/R)^2) (for r >= R); Z/r")


    if mode == "generate":
        print(f"Saving output in {clean_name(args.output_dir)} directory")
        Generate_Fermi_Data(config)
    elif mode == "plot":
        Visualize(config)
    elif mode == "data":
        Calc_double_beta_decay_spectrum(config)
    else:
        raise ValueError("Unkown mode: {mode}")


if __name__ == "__main__":
    main()
