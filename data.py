#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np


def remove_data_from_npz(filename, keys_to_remove, output_filename = None):
    data = np.load(filename)
    keep = {}

    for name in data.files:
        if name in keys_to_remove:
            print(f"Removing entry: {name}")
        else:
            keep[name] = data[name]

    if output_filename is None:
        output_filename = filename

    np.savez(output_filename, **keep)
    print(f"Saved cleaned npz to: {output_filename}")

#
# to_delete = ['Coulomb_E46954.08895429713_Z-1_l0_P', 'Coulomb_E47529.09351989325_Z-1_l0_r']
# # #
# remove_data_from_npz("Coulomb_Wavefunctions_beta_decay.npz" , to_delete)
#



##########################
##### Data comparisons
#######################



data = np.load("Coulomb_Wavefunctions_beta_decay.npz")
#
print(data.files)
# #
# # r1 = data["Spline_exp_E0.5_Z-6_0_N80000_r"]
# # P1 = data["Spline_exp_E0.5_Z-6_0_N80000_P"]
# # # #
# # # r2 = data["Spline_exp_E0.5_Z-6_0_N70000_r"]
# # # P2 = data["Spline_exp_E0.5_Z-6_0_N70000_P"]
# # #
# # #
# # r3 = data["Spline_exp_E0.5_Z-6_0_N100000_r"]
# # P3 = data["Spline_exp_E0.5_Z-6_0_N100000_P"]
# #
# # # r4 = data["Spline_exp_E0.5_Z-6_0_N55000_r"]
# # # P4 = data["Spline_exp_E0.5_Z-6_0_N55000_P"]
# #
# #
# # # r5 = data["Spline_exp_E0.5_Z-6_0_N55000_1_r"]
# # # P5 = data["Spline_exp_E0.5_Z-6_0_N55000_1_P"]
# # # #
# #
# # r6 = data["Spline_exp_E0.5_Z-6_0_N55000_2_r"]
# # P6 = data["Spline_exp_E0.5_Z-6_0_N55000_2_P"]
# #
# # r7 = data["Spline_exp_E0.5_Z-6_0_N80000_1_r"]
# # P7 = data["Spline_exp_E0.5_Z-6_0_N80000_1_P"]
# #
# r8 = data["Spline_exp_E0.5_Z-6_0_N100000_1_r"]
# P8 = data["Spline_exp_E0.5_Z-6_0_N100000_1_P"]
# #
# plt.figure(figsize=(12,8))
#
# # plt.plot(r1,P1/max(P1) , label = "N=80k")
# # # plt.plot(r2,P2/max(P2), label = "N=70k")
# # plt.plot(r3,P3/max(P3), label = "N=100K")
# # plt.plot(r4,P4/max(P4) , label = "N=55K, n=100")
# # # plt.plot(r5,P5/max(P5) , ls = "--" , label = "N=55K, n=10")
# # plt.plot(r6, P6/max(P6), label = "N=55k, global V")
# # plt.plot(r7, P7/max(P7), ls = "-" , label = "N=80k, global V")
# plt.plot(r8, P8/max(P8), ls = "--" , label = "N=100k, global V ")
# plt.xlabel("r")
# plt.ylabel("P(r)")
# plt.legend()
# plt.show()
#




















