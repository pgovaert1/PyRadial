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

    with open("config.json") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="Run double beta decay code")

    #Generator parses
    parser.add_argument("--mode", type = str, choices=["generate", "plot", "data"],  help = f"Run mode: 'generate' or 'plot' or 'data', (default from config: { config["generator"]["mode"]})")
    parser.add_argument("--num_samples", type = positive_int, help = f"Number of energy samples generated, (default from config: {config["generator"]["num_samples"]})")
    parser.add_argument("--plot_energy", type = positive_float, help = f"Kinetic energy in MeV for single plot, (default from config: {config["generator"]["T_plot_energy-MeV"]})")
    parser.add_argument("--potential", type = int, choices = [0,1,2,3], help = f"Select potential function, 0: Z/r , 1: Z/r + V0 exp(-Ar) 2: (for r<R); Z/2R (3-(r/R)^2) (for r >= R); Z/r , 3: Thomas-Fermi  , (default from config: {config["generator"]["potential_index"]})")

    #Paths parses
    parser.add_argument("--output_dir", type = clean_name, help = f"Output directory name, (default from config: {config["paths"]["output_directory"]}) ")

    #Parameter parses
    parser.add_argument("--atomic_num", type = positive_int, help = f"Atomic number (Z) given as posivite integer value, (default from config: {config["parameters"]["atomic_number"]})")
    parser.add_argument("--mass_number", type = positive_int, help = f"Mass number (A), (default from config: {config["parameters"]["mass_number"]})")
    parser.add_argument("--angular_momentum", type = int, help = f"Angular momentum (l), (default from config: {config["parameters"]["angular_momentum_l"]})")
    parser.add_argument("--kappa", type = int, help = f"Kappa, (default from config: { config["parameters"]["kappa"]:+d} )")


    # Mesh creation
    parser.add_argument("--distance", type = positive_float, help = f"Max distance mesh grid is built up to, (default from config: {config["mesh_grid"]["end_point"]})")
    parser.add_argument("--num_steps", type = positive_int, help = f"Number of steps mesh is made of, (default from config: {config["mesh_grid"]["num_mesh_steps"]})")
    parser.add_argument("--DRN", type = positive_float, help = f"Upper limit step size grid, (default from config: {config["mesh_grid"]["upper_limit_step_size"]})")


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

    # Paths overrides
    if args.output_dir:
        config["paths"]["output_directory"] = args.output_dir

    # Parameter overrides
    if args.atomic_num is not None:
        config["parameters"]["atomic_number"] = args.atomic_number

    if args.mass_number is not None:
        config["parameters"]["mass_number"] = args.mass_number

    if args.angular_momentum is not None:
        config["parameters"]["angular_momentum_l"] = args.angular_momentum

    if args.kappa is not None:
        config["parameters"]["kappa"] = args.kappa

    # Mesh creation overrides
    if args.distance is not None:
        config["mesh_grid"]["end_point"] = args.distance

    if args.num_steps is not None:
        config["mesh_grid"]["num_mesh_steps"] = args.num_steps

    if args.DRN is not None:
        config["mesh_grid"]["upper_limit_step_size"] = args.DRN






    mode = config["generator"]["mode"]
    potential = config["generator"]["potential_index"]
    if potential == 0:
        print(f"calling '{mode}' function for potential 0: Z/r")
    if potential == 1:
        print(f"calling '{mode}' function for potential 1: Z/r + V0 exp(-Ar)")
    if potential == 2:
        print(f"calling '{mode}' function for potential 2: (for r<R); Z/2R (3-(r/R)^2) (for r >= R); Z/r")
    if potential == 3:
        print(f"calling '{mode}' function for potential 3: Thomas Fermi")

    if mode == "generate":
        print(f"Saving output in \\{config["paths"]["output_directory"]} directory")
        Generate_Fermi_Data(config)
    elif mode == "plot":
        Visualize(config)
    elif mode == "data":
        Calc_double_beta_decay_spectrum(config)
    else:
        raise ValueError("Unkown mode: {mode}")


if __name__ == "__main__":
    main()
