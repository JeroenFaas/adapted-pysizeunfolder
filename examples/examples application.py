import pysizeunfolder as pu
import numpy as np
import matplotlib.pyplot as plt
import pickle
from statsmodels.distributions import ECDF


########################################## Example 1: True size distribution ###########################################

# True volume distribution:
sample_3D = np.loadtxt("examples/sample_data_3d.txt", skiprows=1)
H_true = ECDF(sample_3D)
x_vol_true = sample_3D
y_true = H_true(x_vol_true)
# True size (= cubic root of volume) distribution:
x_size_true = sample_3D**(1/3)

# True biased distribution:
idx_sort = x_size_true.argsort()
diff_y_true = np.append(y_true[idx_sort][0], np.diff(y_true[idx_sort]))
y_true_biased = np.cumsum(x_size_true[idx_sort] * diff_y_true)
y_true_biased /= y_true_biased[-1]

# For plotting, we need to add some additional points:
x_vol_true = np.append(np.append(0, x_vol_true[idx_sort]), 1.05*x_vol_true[idx_sort][-1])
x_size_true = np.append(np.append(0, x_size_true[idx_sort]), 1.05*x_size_true[idx_sort][-1])
y_true = np.append(np.append(0, y_true[idx_sort]), 1)
y_true_biased = np.append(np.append(0, y_true_biased), 1)

# Histogram true empirical volume density
plt.figure(figsize=(4, 3))
plt.hist(sample_3D, bins=500, ec="black", linewidth=0.2, density=True)
plt.xlabel(r"volume ($\mu m^3$)")
plt.xlim(0, 15000)
plt.ylabel("density")
plt.tight_layout()
plt.savefig("sample_data_true_vol_hist.png", dpi=600)
plt.close()
# Histogram true empirical size density
plt.figure(figsize=(4, 3))
plt.hist(sample_3D**(1/3), bins=70, ec="black", linewidth=0.2, density=True)
plt.xlabel(r"$\lambda$ ($\mu m$)")
plt.xlim(0, 40)
plt.ylabel("density")
plt.tight_layout()
plt.savefig("sample_data_true_size_hist.png", dpi=600)
plt.close()

# Plot true empirical volume distribution
plt.figure(figsize=(4, 3))
plt.plot(x_vol_true, y_true, c="tab:red", label="truth")
plt.xlim(left=0)
plt.xlabel(r"volume ($\mu m^3$)")
plt.xlim(0, 15000)
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("sample_data_true_vol.png", dpi=600)
plt.close()
# Plot true empirical size distribution
plt.figure(figsize=(4, 3))
plt.plot(x_size_true, y_true, c="tab:red", label="truth")
plt.xlim(0, 30)
plt.xlabel(r"$\lambda$ ($\mu m$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("sample_data_true_size.png", dpi=600)
plt.close()

# Plot true empirical biased volume distribution
plt.figure(figsize=(4, 3))
plt.plot(x_vol_true, y_true_biased, c="tab:red", label="truth")
plt.xlim(0, 30000)
plt.xlabel(r"volume ($\mu m^3$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("sample_data_true_biased_vol.png", dpi=600)
plt.close()
# Plot true empirical biased size distribution
plt.figure(figsize=(4, 3))
plt.plot(x_size_true, y_true_biased, c="tab:red", label="truth")
plt.xlim(0, 35)
plt.xlabel(r"$\lambda$ ($\mu m$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("sample_data_true_biased_size.png", dpi=600)
plt.close()


###################################### Example 2: Sample Interstitial Free Steel #######################################

# Application to real data set:
sample_2D = np.loadtxt("examples/sample_data_2d.txt", delimiter='\t', skiprows=1)
observed_sample = sample_2D[sample_2D[:, 2] == 0]
# Columns:  area	vertices	type	avg_x	avg_y

############### Interpretation 1: Treat V > Vmax like Vmax #######################
shape = ["tetrahedron", "cube", "dodecahedron"]
reference_sample = []
ref_v_density = []
ref_v_distr = []
V_K = []
obs_sample_K = []
sample_v_K_density = []
sample_v_K_distr = []
deviation1 = []

for K in range(len(shape)):
    reference_sample.append(pickle.load(open(f"examples/{shape[K]}_sample(7).pkl", "rb")))

    Vmax = reference_sample[K][1].max()
    obs_sample_K.append(observed_sample.copy())
    obs_sample_K[K][obs_sample_K[K][:, 1] > Vmax, 1] = Vmax  # Set all entries with V > Vmax to have V = Vmax.
    V_K.append(np.arange(obs_sample_K[K][:, 1].min(), obs_sample_K[K][:, 1].max() + 1, dtype=int))

    sample_v_K_density.append(np.array([sum(obs_sample_K[K][:, 1] == v) / len(obs_sample_K[K]) for v in V_K[K]]))
    sample_v_K_distr.append(np.append(np.append(0, np.cumsum(sample_v_K_density[K])), 1))

    ref_v_density.append(np.array([sum(reference_sample[K][1] == v) / len(reference_sample[K][1]) for v in V_K[K]]))
    ref_v_distr.append(np.append(np.append(0, np.cumsum(ref_v_density[K])), 1))

    deviation1.append(0)
    for v in V_K[K] - V_K[K][0] + 1:
        diff = abs(sample_v_K_distr[K][v] - ref_v_distr[K][v])  # Absolute difference in CDF at step v.
        # Update max deviation if necessary:
        if diff > deviation1[K]:
            deviation1[K] = diff

colors = ["blue", "orange", "green"]
for K in range(len(shape)):
    # Bar plot density of number of vertices
    plt.figure(figsize=(4, 3))
    plt.bar(V_K[K], sample_v_K_density[K])
    plt.xlabel("number of vertices")
    plt.xticks(V_K[K])
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(f"sample_data_num_verts_bar_1{shape[K]}.png", dpi=600)
    plt.close()

    # Step func plot distr of number of vertices
    v_axis = np.append(np.append(V_K[K][0] - 1, V_K[K]), V_K[K][-1] + 1)
    plt.figure(figsize=(4, 3))
    plt.step(v_axis, ref_v_distr[K], where="post", label=shape[K], c=f"tab:{colors[K]}")
    plt.step(v_axis, sample_v_K_distr[K], where="post", label="observations", c="tab:red")
    plt.xlabel("number of vertices")
    plt.xlim(v_axis[0], v_axis[-1])
    plt.xticks(V_K[K])
    plt.ylabel("CDF")
    plt.legend(title="Distribution of:")
    plt.tight_layout()
    plt.savefig(f"sample_data_num_verts_step_1{shape[K]}.png", dpi=600)
    plt.close()


################# Interpretation 2: translate num verts ##################
# (3) Translate range of number of vertices:
V = np.arange(observed_sample[:, 1].min(), observed_sample[:, 1].max() + 1, dtype=int)
sample_v_density = np.array([sum(observed_sample[:, 1] == v) / len(observed_sample) for v in V])
sample_v_distr = np.append(np.append(0, np.cumsum(sample_v_density)), 1)
obs_sample_translated = []
sample_v_transl_density = []
sample_v_transl_distr = []
deviation2 = []

# Convert sampled num verts to translated num verts matching the best fit shape K:
# Distr V step func shape K: (v_axis, ref_v_distr[K])
# Distr V step func sample: (v_axis, sample_v_distr)
for K in range(len(shape)):
    v_ref = 3
    obs_sample_translated.append(observed_sample.copy())
    for v_sample in V:
        while (abs(ref_v_distr[K][v_ref-2] - sample_v_distr[v_sample-2]) >
               abs(ref_v_distr[K][v_ref-1] - sample_v_distr[v_sample-2])):
            v_ref += 1
        # Distr V of K at v_ref is closest to distr V of sample at v_sample.
        obs_sample_translated[K][obs_sample_translated[K][:, 1] == v_sample, 1] = v_ref

    sample_v_transl_density.append(np.array([sum(obs_sample_translated[K][:, 1] == v) / len(obs_sample_K)
                                             for v in V_K[K]]))
    sample_v_transl_distr.append(np.append(np.append(0, np.cumsum(sample_v_transl_density[K])), 1))

    deviation2.append(0)
    for v in V_K[K] - V_K[K][0] + 1:
        diff = abs(sample_v_transl_distr[K][v] - ref_v_distr[K][v])  # Absolute difference in CDF at step v.
        # Update max deviation if necessary:
        if diff > deviation2[K]:
            deviation2[K] = diff

    # Step func plot distr of number of vertices
    v_axis = np.append(np.append(V_K[K][0] - 1, V_K[K]), V_K[K][-1] + 1)
    plt.figure(figsize=(4, 3))
    plt.step(v_axis, ref_v_distr[K], where="post", label=shape[K], c=f"tab:{colors[K]}")
    plt.step(v_axis, sample_v_transl_distr[K], where="post", label="observations", c="tab:red")
    plt.xlabel("number of vertices")
    plt.xlim(v_axis[0], v_axis[-1])
    plt.xticks(V_K[K])
    plt.ylabel("CDF")
    plt.legend(title="Distribution of:")
    plt.tight_layout()
    plt.savefig(f"sample_data_num_verts_step_2{shape[K]}.png", dpi=600)
    plt.close()


# # Step func plot distr of number of vertices
# v_axis = np.append(np.append(V[0] - 1, V), V[-1] + 1)
# plt.figure(figsize=(6, 3))
# for i in range(len(shape)):
#     a = 0.3
#     if i == np.argmin(deviation):
#         a = 1
#     plt.step(v_axis, ref_v_distr[i], where="post", label=shape[i], alpha=a)
# plt.step(v_axis, sample_v_distr, where="post", label="sample")
# plt.xlabel("number of vertices")
# plt.xlim(v_axis[0], v_axis[-1])
# plt.xticks(V)
# plt.ylabel("CDF")
# plt.legend(title="Distribution of:")
# plt.tight_layout()
# plt.savefig("sample_data_num_verts_step.png", dpi=600)
# plt.close()


########################################### Example 3: Size distr estimation ###########################################
K = 0

################# Interpretation 1 ################
x_pts_size, v_pts1, y_pts_biased_AV1 = pu.estimate_size_AV([obs_sample_K[K][:, 0], obs_sample_K[K][:, 1]],
                                                            reference_sample[K], debias=False)
y_pts_AV1 = pu.de_bias_AV(x_pts_size, v_pts1, y_pts_biased_AV1, reference_sample[K])

################# Interpretation 2 ################
_, v_pts2, y_pts_biased_AV2 = pu.estimate_size_AV([obs_sample_translated[K][:, 0],
                                                   obs_sample_translated[K][:, 1]], reference_sample[K], debias=False)
y_pts_AV2 = pu.de_bias_AV(x_pts_size, v_pts2, y_pts_biased_AV2, reference_sample[K])

#################### Method A #####################
_, y_pts_biased_A = pu.estimate_size(obs_sample_K[K][:, 0], reference_sample[K][0], debias=False)
y_pts_A = pu.de_bias(x_pts_size, y_pts_biased_A, reference_sample[K][0])

# For plotting with the matplotlib step function, we need to add some additional points:
x_pts_size = np.append(np.append(0, x_pts_size), 1.05*x_pts_size[-1])
x_pts_vol = x_pts_size**3  # Volume = size^3
y_pts_biased_A = np.append(np.append(0, y_pts_biased_A), 1)
y_pts_biased_AV1 = np.append(np.append(0, y_pts_biased_AV1), 1)
y_pts_biased_AV2 = np.append(np.append(0, y_pts_biased_AV2), 1)
y_pts_A = np.append(np.append(0, y_pts_A), 1)
y_pts_AV1 = np.append(np.append(0, y_pts_AV1), 1)
y_pts_AV2 = np.append(np.append(0, y_pts_AV2), 1)


# Step funcs plot estimates of biased volume distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_vol, y_pts_biased_A, where="post", label="estimate A")
plt.step(x_pts_vol, y_pts_biased_AV1, where="post", label="estimate A, V (1)")
plt.step(x_pts_vol, y_pts_biased_AV2, where="post", label="estimate A, V (2)")
plt.plot(x_vol_true, y_true_biased, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 30000)
plt.xlabel(r"volume ($\mu m^3$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"sample_data_estimates_biased_vol_{shape[K]}.png", dpi=600)
plt.close()
# Step funcs plot estimates of biased size distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_size, y_pts_biased_A, where="post", label="estimate A")
plt.step(x_pts_size, y_pts_biased_AV1, where="post", label="estimate A, V (1)")
plt.step(x_pts_size, y_pts_biased_AV2, where="post", label="estimate A, V (2)")
plt.plot(x_size_true, y_true_biased, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 35)
plt.xlabel(r"$\lambda$ ($\mu m$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"sample_data_estimates_biased_size_{shape[K]}.png", dpi=600)
plt.close()

# Step funcs plot estimates of true volume distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_vol, y_pts_A, where="post", label="estimate A")
plt.step(x_pts_vol, y_pts_AV1, where="post", label="estimate A, V (1)")
plt.step(x_pts_vol, y_pts_AV2, where="post", label="estimate A, V (2)")
plt.plot(x_vol_true, y_true, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 30000)
plt.xlabel(r"volume ($\mu m^3$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"sample_data_estimates_vol_{shape[K]}.png", dpi=600)
plt.close()
# Step funcs plot estimates of true size distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_size, y_pts_A, where="post", label="estimate A")
plt.step(x_pts_size, y_pts_AV1, where="post", label="estimate A, V (1)")
plt.step(x_pts_size, y_pts_AV2, where="post", label="estimate A, V (2)")
plt.plot(x_size_true, y_true, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 35)
plt.xlabel(r"$\lambda$ ($\mu m$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"sample_data_estimates_size_{shape[K]}.png", dpi=600)
plt.close()


################# Interpretation 2 ################
x_pts_size, v_pts, y_pts_biased_AV = pu.estimate_size_AV([obs_sample_translated[K][:, 0],
                                                          obs_sample_translated[K][:, 1]], reference_sample[K],
                                                         debias=False)
y_pts_AV = pu.de_bias_AV(x_pts_size, v_pts, y_pts_biased_AV, reference_sample[K])

_, y_pts_biased_A = pu.estimate_size(obs_sample_translated[K][:, 0], reference_sample[K][0], debias=False)
y_pts_A = pu.de_bias(x_pts_size, y_pts_biased_A, reference_sample[K][0])

# For plotting with the matplotlib step function, we need to add some additional points:
x_pts_size = np.append(np.append(0, x_pts_size), 1.05*x_pts_size[-1])
x_pts_vol = x_pts_size**3  # Volume = size^3
y_pts_biased_A = np.append(np.append(0, y_pts_biased_A), 1)
y_pts_biased_AV = np.append(np.append(0, y_pts_biased_AV), 1)
y_pts_A = np.append(np.append(0, y_pts_A), 1)
y_pts_AV = np.append(np.append(0, y_pts_AV), 1)

# Step funcs plot estimates of biased volume distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_vol, y_pts_biased_A, where="post", label="estimate (A)")
plt.step(x_pts_vol, y_pts_biased_AV, where="post", label="estimate (A, V)")
plt.plot(x_vol_true, y_true_biased, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 30000)
plt.xlabel(r"volume ($\mu m^3$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"sample_data_estimates_biased_vol_2{shape[K]}.png", dpi=600)
plt.close()
# Step funcs plot estimates of biased size distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_size, y_pts_biased_A, where="post", label="estimate (A)")
plt.step(x_pts_size, y_pts_biased_AV, where="post", label="estimate (A, V)")
plt.plot(x_size_true, y_true_biased, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 35)
plt.xlabel(r"$\lambda$ ($\mu m$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"sample_data_estimates_biased_size_2{shape[K]}.png", dpi=600)
plt.close()

# Step funcs plot estimates of true volume distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_vol, y_pts_A, where="post", label="estimate (A)")
plt.step(x_pts_vol, y_pts_AV, where="post", label="estimate (A, V)")
plt.plot(x_vol_true, y_true, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 30000)
plt.xlabel(r"volume ($\mu m^3$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"sample_data_estimates_vol_2{shape[K]}.png", dpi=600)
plt.close()
# Step funcs plot estimates of true size distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_size, y_pts_A, where="post", label="estimate (A)")
plt.step(x_pts_size, y_pts_AV, where="post", label="estimate (A, V)")
plt.plot(x_size_true, y_true, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 35)
plt.xlabel(r"$\lambda$ ($\mu m$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"sample_data_estimates_size_2{shape[K]}.png", dpi=600)
plt.close()


# # Size distr estimation based on best fit shape:
# K = np.argmin(deviation2)  # Shape = shape[K]; Ref sample for G_K = ref_sample[K]
# Vmax = reference_sample[K][1].max()
# # Limit to valid data:
# observed_sample = sample_2D[sample_2D[:, 1] <= Vmax]  # Discard higher num verts data
# observed_sample = observed_sample[observed_sample[:, 2] == 0]
#
# x_pts_size, v_pts, y_pts_biased_AV = pu.estimate_size_AV([observed_sample[:, 0], observed_sample[:, 1]],
#                                                          reference_sample[K], debias=False)
# y_pts_AV = pu.de_bias_AV(x_pts_size, v_pts, y_pts_biased_AV, reference_sample[K])
#
# _, y_pts_biased_A = pu.estimate_size(observed_sample[:, 0], reference_sample[K][0], debias=False)
# y_pts_A = pu.de_bias(x_pts_size, y_pts_biased_A, reference_sample[K][0])
#
# # For plotting with the matplotlib step function, we need to add some additional points:
# x_pts_size = np.append(np.append(0, x_pts_size), 1.05*x_pts_size[-1])
# x_pts_vol = x_pts_size**3  # Volume = size^3
# y_pts_biased_A = np.append(np.append(0, y_pts_biased_A), 1)
# y_pts_biased_AV = np.append(np.append(0, y_pts_biased_AV), 1)
# y_pts_A = np.append(np.append(0, y_pts_A), 1)
# y_pts_AV = np.append(np.append(0, y_pts_AV), 1)
#
# # Step funcs plot estimates of biased volume distr for both methods
# plt.figure(figsize=(4, 3))
# plt.step(x_pts_vol, y_pts_biased_A, where="post", label="estimate (A)")
# plt.step(x_pts_vol, y_pts_biased_AV, where="post", label="estimate (A, V)")
# plt.plot(x_vol_true, y_true_biased, c="red", linestyle="dashed", label="truth")
# plt.xlim(0, 30000)
# plt.xlabel(r"volume ($\mu m^3$)")
# plt.ylabel("CDF")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig("sample_data_estimates_biased_vol.png", dpi=600)
# plt.close()
# # Step funcs plot estimates of biased size distr for both methods
# plt.figure(figsize=(4, 3))
# plt.step(x_pts_size, y_pts_biased_A, where="post", label="estimate (A)")
# plt.step(x_pts_size, y_pts_biased_AV, where="post", label="estimate (A, V)")
# plt.plot(x_size_true, y_true_biased, c="red", linestyle="dashed", label="truth")
# plt.xlim(0, x_pts_size[-1])
# plt.xlabel(r"$\lambda$ ($\mu m$)")
# plt.ylabel("CDF")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig("sample_data_estimates_biased_size.png", dpi=600)
# plt.close()
#
# # Step funcs plot estimates of true volume distr for both methods
# plt.figure(figsize=(4, 3))
# plt.step(x_pts_vol, y_pts_A, where="post", label="estimate (A)")
# plt.step(x_pts_vol, y_pts_AV, where="post", label="estimate (A, V)")
# plt.plot(x_vol_true, y_true, c="red", linestyle="dashed", label="truth")
# plt.xlim(0, 30000)
# plt.xlabel(r"volume ($\mu m^3$)")
# plt.ylabel("CDF")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig("sample_data_estimates_vol.png", dpi=600)
# plt.close()
# # Step funcs plot estimates of true size distr for both methods
# plt.figure(figsize=(4, 3))
# plt.step(x_pts_size, y_pts_A, where="post", label="estimate (A)")
# plt.step(x_pts_size, y_pts_AV, where="post", label="estimate (A, V)")
# plt.plot(x_size_true, y_true, c="red", linestyle="dashed", label="truth")
# plt.xlim(0, x_pts_size[-1])
# plt.xlabel(r"$\lambda$ ($\mu m$)")
# plt.ylabel("CDF")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig("sample_data_estimates_size.png", dpi=600)
# plt.close()


####################################### Example 4: Estimation incl invalid data ########################################

# Size distr estimation based on best fit shape:
K = np.argmin(deviation)  # Shape = shape[K]; Ref sample for G_K = ref_sample[K]
Vmax = reference_sample[K][1].max()

# Limit to valid data:
observed_sample = sample_2D[sample_2D[:, 2] == 0]

# # (2) Treat like 10 vertices:
# observed_sample[observed_sample[:, 1] > Vmax, 1] = Vmax  # Set all entries with V > Vmax to have V = Vmax.

# (3) Interpolate range of number of vertices (version 1):
# Convert sampled num verts to interpolated num verts matching the best fit shape K:
# Distr V step func shape K: (v_axis, ref_v_distr[K])
# Distr V step func sample: (v_axis, sample_v_distr)
v_ref = 3
for v_sample in V:
    while (abs(ref_v_distr[K][v_ref-2] - sample_v_distr[v_sample-2]) >
           abs(ref_v_distr[K][v_ref-1] - sample_v_distr[v_sample-2])):
        v_ref += 1
    # Distr V of K at v_ref is closest to distr V of sample at v_sample.
    observed_sample[observed_sample[:, 1] == v_sample, 1] = v_ref  # Set all entries with V=v_sample to have V=v_ref.

# # (4) Interpolate range of number of vertices (version 2):
# # Convert sampled num verts to interpolated num verts matching the best fit shape K:
# # Distr V step func shape K: (v_axis, ref_v_distr[K])
# # Distr V step func sample: (v_axis, sample_v_distr)
# v_ref = 3
# for v_sample in V:
#     while ref_v_distr[K][v_ref-2] < sample_v_distr[v_sample-2]:
#         v_ref += 1
#     # Distr V of K at v_ref >= Distr V of sample at v_sample.
#     observed_sample[observed_sample[:, 1] == v_sample, 1] = v_ref  # Set all entries with V=v_sample to have V=v_ref.


x_pts_size, v_pts, y_pts_biased_AV = pu.estimate_size_AV([observed_sample[:, 0], observed_sample[:, 1]],
                                                         reference_sample[K], debias=False)
y_pts_AV = pu.de_bias_AV(x_pts_size, v_pts, y_pts_biased_AV, reference_sample[K])

_, y_pts_biased_A = pu.estimate_size(observed_sample[:, 0], reference_sample[K][0], debias=False)
y_pts_A = pu.de_bias(x_pts_size, y_pts_biased_A, reference_sample[K][0])

# For plotting with the matplotlib step function, we need to add some additional points:
x_pts_size = np.append(np.append(0, x_pts_size), 1.05*x_pts_size[-1])
x_pts_vol = x_pts_size**3  # Volume = size^3
y_pts_biased_A = np.append(np.append(0, y_pts_biased_A), 1)
y_pts_biased_AV = np.append(np.append(0, y_pts_biased_AV), 1)
y_pts_A = np.append(np.append(0, y_pts_A), 1)
y_pts_AV = np.append(np.append(0, y_pts_AV), 1)

# Step funcs plot estimates of biased volume distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_vol, y_pts_biased_A, where="post", label="estimate (A)")
plt.step(x_pts_vol, y_pts_biased_AV, where="post", label="estimate (A, V)")
plt.plot(x_vol_true, y_true_biased, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 30000)
plt.xlabel(r"volume ($\mu m^3$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("sample_data_estimates_biased_vol.png", dpi=600)
plt.close()
# Step funcs plot estimates of biased size distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_size, y_pts_biased_A, where="post", label="estimate (A)")
plt.step(x_pts_size, y_pts_biased_AV, where="post", label="estimate (A, V)")
plt.plot(x_size_true, y_true_biased, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 35)
plt.xlabel(r"$\lambda$ ($\mu m$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("sample_data_estimates_biased_size.png", dpi=600)
plt.close()

# Step funcs plot estimates of true volume distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_vol, y_pts_A, where="post", label="estimate (A)")
plt.step(x_pts_vol, y_pts_AV, where="post", label="estimate (A, V)")
plt.plot(x_vol_true, y_true, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 30000)
plt.xlabel(r"volume ($\mu m^3$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("sample_data_estimates_vol.png", dpi=600)
plt.close()
# Step funcs plot estimates of true size distr for both methods
plt.figure(figsize=(4, 3))
plt.step(x_pts_size, y_pts_A, where="post", label="estimate (A)")
plt.step(x_pts_size, y_pts_AV, where="post", label="estimate (A, V)")
plt.plot(x_size_true, y_true, c="red", linestyle="dashed", label="truth")
plt.xlim(0, 35)
plt.xlabel(r"$\lambda$ ($\mu m$)")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("sample_data_estimates_size.png", dpi=600)
plt.close()


############################################ Example 5: Estimation errors ##############################################

# True volume/size distribution: (x_vol/size_true, y_true(_biased))
# Estimate volume/size distribution based on A/AV: (x_pts_vol/size, y_pts(_biased)_A/AV)
errors_biased = pu.error_estimate(x_size_true, y_true_biased, np.array([x_pts_size, x_pts_size]),
                                  np.array([y_pts_biased_A, y_pts_biased_AV]))
errors = pu.error_estimate(x_size_true, y_true, np.array([x_pts_size, x_pts_size]), np.array([y_pts_A, y_pts_AV]))

print(f"""Errors:
Biased A\t{errors_biased[0]:.4f}
True A\t{errors[0]:.4f}
Biased AV\t{errors_biased[1]:.4f}
True AV\t{errors[1]:.4f}""")
