import pysizeunfolder as pu
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import expon, gamma, lognorm


# Authors: Thomas van der Jagt (1), Jeroen Faas (2-6) (modified 1)


####################################### Example 1: st exp-distr based on A #############################################
rng = np.random.default_rng(0)
reference_sample = pickle.load(open("examples/cube_sample.pkl", "rb"))[0]  # CUBE
# reference_sample = pickle.load(open("examples/dodecahedron_sample.pkl", "rb"))[0]  # DODECAHEDRON
n = 1000

# Generate a sample of observed section areas (Lemma 2), given that particles are cubes and the
# underlying size distribution is a standard exponential distribution.
sizes = rng.gamma(shape=2, scale=1, size=n)
areas = pu.iur_3d_shape("cube", n, rng=rng)  # CUBE
# areas = pu.iur_3d_shape("dodecahedron", n, rng=rng)  # DODECAHEDRON
sample = np.square(sizes)*areas

# Estimate the underlying size distribution CDF using the sample of observed section areas
x_pts, y_pts = pu.estimate_size(sample, reference_sample)

# For plotting with the matplotlib step function, we need to add some additional points
x_pts = np.append(np.append(0, x_pts), 1.05*x_pts[-1])
y_pts = np.append(np.append(0, y_pts), 1)

# The true size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = expon.cdf(x)

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 7])
# plt.title(r"Estimate vs truth $(H(\lambda))$")
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("size_estimate.png", dpi=600)
#plt.show()

# Alternatively: skip the de-biasing step
# Estimate the underlying size distribution using the sample of observed section areas
x_pts, y_pts = pu.estimate_size(sample, reference_sample, debias=False)

# For plotting with the matplotlib step function, we need to add some additional points
x_pts = np.append(np.append(0, x_pts), 1.05*x_pts[-1])
y_pts = np.append(np.append(0, y_pts), 1)

# The true biased size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = gamma.cdf(x, a=2, scale=1)

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 9])
# plt.title(r"Estimate vs truth ($H^b(\lambda)$)")
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("biased_size_estimate.png", dpi=600)
#plt.show()

# Various code examples

# The estimated (biased) volume distribution function may be plotted via
x_pts = np.append(np.append(0, x_pts**3), 1.05*x_pts[-1]**3)
y_pts = np.append(np.append(0, y_pts), 1)
# Estimate both biased and debiased in a more efficient way:
x_pts_biased, y_pts_biased = pu.estimate_size(sample, reference_sample, debias=False)
y_pts = pu.de_bias(x_pts_biased, y_pts_biased, reference_sample)  # x_pts=x_pts_biased


##################################### Example 2: lognormal-distr based on A ############################################
rng = np.random.default_rng(0)
# reference_sample = pickle.load(open("examples/cube_sample.pkl", "rb"))[0]  # CUBE
reference_sample = pickle.load(open("examples/dodecahedron_sample.pkl", "rb"))[0]  # DODECAHEDRON
n = 1000

# Generate a sample of observed section areas (Lemma 2), given that particles are cubes/dodecahedra
# and the underlying size distribution is a lognormal(mu, sigma) distribution.
mu = 2
sigma = 0.5
# The corresponding biased distribution is a lognormal(mu + sigma^2, sigma) distribution.
sizes = rng.lognormal(mean=mu + sigma**2, sigma=sigma, size=n)
# areas = pu.iur_3d_shape("cube", n, rng=rng)  # CUBE
areas = pu.iur_3d_shape("dodecahedron", n, rng=rng)  # DODECAHEDRON
sample = np.square(sizes)*areas

# Estimate the underlying size distribution CDF using the sample of observed section areas
x_pts, y_pts_biased = pu.estimate_size(sample, reference_sample, debias=False)
y_pts = pu.de_bias(x_pts, y_pts_biased, reference_sample)

# For plotting with the matplotlib step function, we need to add some additional points
x_pts = np.append(np.append(0, x_pts), 1.05*x_pts[-1])
y_pts = np.append(np.append(0, y_pts), 1)

# The true size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = lognorm.cdf(x, sigma, scale=np.exp(mu))  # The lognormal(mu, sigma) distribution.

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 25])
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("estimate_size.png", dpi=600)
plt.close()

# Alternatively: skip the de-biasing step
y_pts_biased = np.append(np.append(0, y_pts_biased), 1)

# The true biased size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = lognorm.cdf(x, sigma, scale=np.exp(mu + sigma**2))  # The lognormal(mu + sigma^2, sigma) distribution.

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts_biased, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 30])
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("estimate_biased_size.png", dpi=600)
plt.close()


####################################### Example 3: st exp-distr based on A & V #########################################
rng = np.random.default_rng(0)
# reference_sample = pickle.load(open("examples/cube_sample.pkl", "rb"))  # CUBE
reference_sample = pickle.load(open("examples/dodecahedron_sample.pkl", "rb"))  # DODECAHEDRON
n = 1000

# Generate a sample of observed section areas (Lemma 2), given that particles are cubes and the
# underlying size distribution is a standard exponential distribution.
sizes = rng.gamma(shape=2, scale=1, size=n)
# areas, sections = pu.iur_3d_shape("cube", n, return_vertices=True, rng=rng)  # CUBE
areas, sections = pu.iur_3d_shape("dodecahedron", n, return_vertices=True, rng=rng)  # DODECAHEDRON
sample_areas = np.square(sizes)*areas

# Number of vertices for each section profile.
num_verts = pu.number_vertices(sections)
sample = [sample_areas, num_verts]

# Estimate the underlying size distribution CDF using the sample of observed pairs of section area and num of verts.
x_pts, y_pts_biased = pu.estimate_size_AV(sample, reference_sample, debias=False)
y_pts = pu.de_bias(x_pts, y_pts_biased, reference_sample[0])

# For plotting with the matplotlib step function, we need to add some additional points
x_pts = np.append(np.append(0, x_pts), 1.05*x_pts[-1])
y_pts = np.append(np.append(0, y_pts), 1)

# The true size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = expon.cdf(x)

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 7])
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("estimate_size_AV.png", dpi=600)
plt.close()

# Alternatively: skip the de-biasing step
y_pts_biased = np.append(np.append(0, y_pts_biased), 1)

# The true biased size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = gamma.cdf(x, a=2, scale=1)

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts_biased, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 9])
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("estimate_biased_size_AV.png", dpi=600)
plt.close()


###################################### Example 4: lognormal-distr based on A & V #######################################
rng = np.random.default_rng(0)
# reference_sample = pickle.load(open("examples/cube_sample.pkl", "rb"))  # CUBE
reference_sample = pickle.load(open("examples/dodecahedron_sample.pkl", "rb"))  # DODECAHEDRON
n = 1000

# Generate a sample of observed section areas (Lemma 2), given that particles are cubes/dodecahedra
# and the underlying size distribution is a lognormal(mu, sigma) distribution.
mu = 2
sigma = 0.5
# The corresponding biased distribution is a lognormal(mu + sigma^2, sigma) distribution.
sizes = rng.lognormal(mean=mu + sigma**2, sigma=sigma, size=n)
# areas, sections = pu.iur_3d_shape("cube", n, return_vertices=True, rng=rng)  # CUBE
areas, sections = pu.iur_3d_shape("dodecahedron", n, return_vertices=True, rng=rng)  # DODECAHEDRON
sample_areas = np.square(sizes)*areas

# Number of vertices for each section profile.
num_verts = pu.number_vertices(sections)
sample = [sample_areas, num_verts]

# Estimate the underlying size distribution CDF using the sample of observed pairs of section area and num of verts.
x_pts, y_pts_biased = pu.estimate_size_AV(sample, reference_sample, debias=False)
y_pts = pu.de_bias(x_pts, y_pts_biased, reference_sample[0])

# For plotting with the matplotlib step function, we need to add some additional points
x_pts = np.append(np.append(0, x_pts), 1.05*x_pts[-1])
y_pts = np.append(np.append(0, y_pts), 1)

# The true size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = lognorm.cdf(x, sigma, scale=np.exp(mu))  # The lognormal(mu, sigma) distribution.

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 25])
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("estimate_size_AV.png", dpi=600)
plt.close()

# Alternatively: skip the de-biasing step
y_pts_biased = np.append(np.append(0, y_pts_biased), 1)

# The true biased size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
y = lognorm.cdf(x, sigma, scale=np.exp(mu + sigma**2))  # The lognormal(mu + sigma^2, sigma) distribution.

plt.figure(figsize=(4, 3))
plt.step(x_pts, y_pts_biased, where="post", c="tab:blue", label="estimate")
plt.plot(x, y, c="red", linestyle="dashed", label="truth")
plt.xlim([0, 30])
plt.xlabel(r"$\lambda$")
plt.ylabel("CDF")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("estimate_biased_size_AV.png", dpi=600)
plt.close()


############################################ Example 5: many est runs based on A & V ###################################
shape = "dodecahedron"  # "tetrahedron" / "cube" / "dodecahedron"
distr = "lognorm"  # "exp" / "lognorm"
n = 1000  # 1000 / 2000 / 5000 / 10000
repeat = 100  # 1 / 100
############ Do not alter below ############
rng = np.random.default_rng(0)
reference_sample = pickle.load(open(f"examples/{shape}_sample.pkl", "rb"))

# Generate a sample of observed section areas (Lemma 3.1):
sizes = np.zeros(repeat*n, dtype=float)
if distr == "exp":
    # and the underlying size distribution is a standard exponential distribution,
    # The corresponding biased distribution is a gamma(2, 1) distribution.
    sizes = rng.gamma(shape=2, scale=1, size=repeat*n)
elif distr == "lognorm":
    # or the underlying size distribution is a lognormal(mu, sigma) distribution.
    mu = 2
    sigma = 0.5
    # The corresponding biased distribution is a lognormal(mu + sigma^2, sigma) distribution.
    sizes = rng.lognormal(mean=mu + sigma**2, sigma=sigma, size=repeat*n)
else:
    print("Invalid distribution given.")

areas, sections = pu.iur_3d_shape(shape, repeat*n, return_vertices=True, rng=rng)

sample_areas = np.square(sizes)*areas
# Number of vertices for each section profile.
num_verts = pu.number_vertices(sections)
# Reshape:
sample_areas = np.reshape(sample_areas, (repeat, n))
num_verts = np.reshape(num_verts, (repeat, n))

v_pts = np.zeros((repeat, n), dtype=int)
x_pts = np.zeros((repeat, n+2), dtype=float)
y_pts_A = np.zeros((repeat, n+2), dtype=float)
y_pts_AV = np.zeros((repeat, n+2), dtype=float)
y_pts_biased_A = np.zeros((repeat, n+2), dtype=float)
y_pts_biased_AV = np.zeros((repeat, n+2), dtype=float)

for i in range(repeat):
    # Estimate the underlying size distribution CDF using the sample of observed pairs of A & V.
    (x_pts[i, 1:n+1], v_pts[i],
     y_pts_biased_AV[i, 1:n+1]) = pu.estimate_size_AV([sample_areas[i], num_verts[i]],
                                                      reference_sample, debias=False)
    y_pts_AV[i, 1:n + 1] = pu.de_bias_AV(x_pts[i, 1:n + 1], v_pts[i], y_pts_biased_AV[i, 1:n + 1], reference_sample)

    # Estimate the underlying size distribution CDF using the sample of observed A.
    _, y_pts_biased_A[i, 1:n + 1] = pu.estimate_size(sample_areas[i], reference_sample[0], debias=False)
    y_pts_A[i, 1:n+1] = pu.de_bias(x_pts[i, 1:n+1], y_pts_biased_A[i, 1:n+1], reference_sample[0])

    # For plotting with the matplotlib step function, we need to add some additional points
    x_pts[i, n+1] = 1.05*x_pts[i, n]
    y_pts_biased_A[i, n+1], y_pts_biased_AV[i, n+1] = 1, 1
    y_pts_A[i, n+1], y_pts_AV[i, n+1] = 1, 1

if repeat > 1:
    # The average of all estimates:
    order = np.argsort(x_pts[:, 1:].flatten())
    diffs_y_A = y_pts_A[:, 1:] - y_pts_A[:, :-1]
    diffs_y_AV = y_pts_AV[:, 1:] - y_pts_AV[:, :-1]

    x_pts_avg = np.append(0, x_pts[:, 1:].flatten()[order])
    y_pts_A_avg = np.append(0, diffs_y_A.flatten()[order])
    y_pts_AV_avg = np.append(0, diffs_y_AV.flatten()[order])
    for i in range(1, len(x_pts_avg)):
        y_pts_A_avg[i] = y_pts_A_avg[i-1] + y_pts_A_avg[i]
        y_pts_AV_avg[i] = y_pts_AV_avg[i-1] + y_pts_AV_avg[i]
    y_pts_A_avg /= repeat
    y_pts_AV_avg /= repeat

# The true size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
if distr == "exp":
    y = expon.cdf(x)  # The exp(1) distribution.
elif distr == "lognorm":
    y = lognorm.cdf(x, sigma, scale=np.exp(mu))  # The lognormal(mu, sigma) distribution.

if n == 1000:
    for params in ("A", "AV"):
        plt.figure(figsize=(4, 3))
        if repeat > 1:
            if params == "A":
                estimates = plt.step(x_pts[0], y_pts_A[0], where="post", c="tab:blue", label="estimate", alpha=0.02)
                for i in range(1, repeat):
                    plt.step(x_pts[i], y_pts_A[i], where="post", c="tab:blue", alpha=0.02)
            else:
                estimates = plt.step(x_pts[0], y_pts_AV[0], where="post", c="tab:blue", label="estimate", alpha=0.02)
                for i in range(1, repeat):
                    plt.step(x_pts[i], y_pts_AV[i], where="post", c="tab:blue", alpha=0.02)
            estimates[0].set_alpha(alpha=0.5)
            if params == "A":
                plt.step(x_pts_avg, y_pts_A_avg, where="post", c="black", label="mean estimate")
            else:
                plt.step(x_pts_avg, y_pts_AV_avg, where="post", c="black", label="mean estimate")
        else:
            if params == "A":
                plt.step(x_pts[0], y_pts_A[0], where="post", c="tab:blue", label="estimate")
            else:
                plt.step(x_pts[0], y_pts_AV[0], where="post", c="tab:blue", label="estimate")
        plt.plot(x, y, c="red", linestyle="dashed", label="truth")
        if distr == "exp":
            plt.xlim([0, 7])
        elif distr == "lognorm":
            plt.xlim([0, 25])
        plt.xlabel(r"$\lambda$")
        plt.ylabel("CDF")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{shape}_estimates_size_{distr}_{params}({n}x{repeat}).png", dpi=600)
        plt.close()

# Alternatively: skip the de-biasing step
if repeat > 1:
    # The average of all estimates:
    diffs_y_biased_A = y_pts_biased_A[:, 1:] - y_pts_biased_A[:, :-1]
    diffs_y_biased_AV = y_pts_biased_AV[:, 1:] - y_pts_biased_AV[:, :-1]

    y_pts_biased_A_avg = np.append(0, diffs_y_biased_A.flatten()[order])
    y_pts_biased_AV_avg = np.append(0, diffs_y_biased_AV.flatten()[order])
    for i in range(1, len(x_pts_avg)):
        y_pts_biased_A_avg[i] = y_pts_biased_A_avg[i-1] + y_pts_biased_A_avg[i]
        y_pts_biased_AV_avg[i] = y_pts_biased_AV_avg[i - 1] + y_pts_biased_AV_avg[i]
    y_pts_biased_A_avg /= repeat
    y_pts_biased_AV_avg /= repeat

# The true biased size distribution CDF
x = np.linspace(0, np.max(x_pts), 2000)
if distr == "exp":
    y = gamma.cdf(x, a=2, scale=1)  # The gamma(2, 1) distribution.
elif distr == "lognorm":
    y = lognorm.cdf(x, sigma, scale=np.exp(mu + sigma**2))  # The lognormal(mu + sigma^2, sigma) distribution.

if n == 1000:
    for params in ("A", "AV"):
        plt.figure(figsize=(4, 3))
        if repeat > 1:
            if params == "A":
                estimates = plt.step(x_pts[0], y_pts_biased_A[0], where="post", c="tab:blue", label="estimate",
                                     alpha=0.02)
                for i in range(1, repeat):
                    plt.step(x_pts[i], y_pts_biased_A[i], where="post", c="tab:blue", alpha=0.02)
            else:
                estimates = plt.step(x_pts[0], y_pts_biased_AV[0], where="post", c="tab:blue", label="estimate",
                                     alpha=0.02)
                for i in range(1, repeat):
                    plt.step(x_pts[i], y_pts_biased_AV[i], where="post", c="tab:blue", alpha=0.02)
            estimates[0].set_alpha(alpha=0.5)
            if params == "A":
                plt.step(x_pts_avg, y_pts_biased_A_avg, where="post", c="black", label="mean estimate")
            else:
                plt.step(x_pts_avg, y_pts_biased_AV_avg, where="post", c="black", label="mean estimate")
        else:
            if params == "A":
                plt.step(x_pts[0], y_pts_biased_A[0], where="post", c="tab:blue", label="estimate")
            else:
                plt.step(x_pts[0], y_pts_biased_AV[0], where="post", c="tab:blue", label="estimate")
        plt.plot(x, y, c="red", linestyle="dashed", label="truth")
        if distr == "exp":
            plt.xlim([0, 9])
        elif distr == "lognorm":
            plt.xlim([0, 30])
        plt.xlabel(r"$\lambda$")
        plt.ylabel("CDF")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{shape}_estimates_biased_size_{distr}_{params}({n}x{repeat}).png", dpi=600)
        plt.close()

# Save data for later analysis.
f = open(f"{shape}_estimates_{distr}({n}).pkl", "wb")
if repeat > 1:
    pickle.dump([sizes, areas, v_pts, x_pts, y_pts_A, y_pts_AV, y_pts_biased_A, y_pts_biased_AV,
                 x_pts_avg, y_pts_A_avg, y_pts_AV_avg, y_pts_biased_A_avg, y_pts_biased_AV_avg], f)
else:
    pickle.dump([sizes, areas, v_pts, x_pts, y_pts_A, y_pts_AV, y_pts_biased_A, y_pts_biased_AV], f)
f.close()

# Plots from file data:
# shape = "cube"  # "tetrahedron" / "cube" / "dodecahedron"
# distr = "exp"  # "exp" / "lognorm"
# repeat = 100  # 1 / 100
# n = 1000
# a = 0.2
#
# data = pickle.load(open(f"examples/{shape}_estimates_{distr}({n}).pkl", "rb"))
# x_pts = data[3]
# y_pts_A = data[4]
# y_pts_AV = data[5]
# y_pts_biased_A = data[6]
# y_pts_biased_AV = data[7]
#
# # The average of all estimates:
# order = np.argsort(x_pts[:, 1:].flatten())
# diffs_y_A = y_pts_A[:, 1:] - y_pts_A[:, :-1]
# diffs_y_AV = y_pts_AV[:, 1:] - y_pts_AV[:, :-1]
# x_pts_avg = np.append(0, x_pts[:, 1:].flatten()[order])
# y_pts_A_avg = np.append(0, diffs_y_A.flatten()[order])
# y_pts_AV_avg = np.append(0, diffs_y_AV.flatten()[order])
# diffs_y_biased_A = y_pts_biased_A[:, 1:] - y_pts_biased_A[:, :-1]
# diffs_y_biased_AV = y_pts_biased_AV[:, 1:] - y_pts_biased_AV[:, :-1]
# y_pts_biased_A_avg = np.append(0, diffs_y_biased_A.flatten()[order])
# y_pts_biased_AV_avg = np.append(0, diffs_y_biased_AV.flatten()[order])
#
# for i in range(1, len(x_pts_avg)):
#     y_pts_A_avg[i] = y_pts_A_avg[i-1] + y_pts_A_avg[i]
#     y_pts_AV_avg[i] = y_pts_AV_avg[i-1] + y_pts_AV_avg[i]
#     y_pts_biased_A_avg[i] = y_pts_biased_A_avg[i - 1] + y_pts_biased_A_avg[i]
#     y_pts_biased_AV_avg[i] = y_pts_biased_AV_avg[i - 1] + y_pts_biased_AV_avg[i]
# y_pts_A_avg /= repeat
# y_pts_AV_avg /= repeat
# y_pts_biased_A_avg /= repeat
# y_pts_biased_AV_avg /= repeat
#
# # The true size distribution CDF
# x = np.linspace(0, np.max(x_pts), 2000)
# if distr == "exp":
#     y = expon.cdf(x)  # The exp(1) distribution.
# elif distr == "lognorm":
#     y = lognorm.cdf(x, sigma, scale=np.exp(mu))  # The lognormal(mu, sigma) distribution.
#
# for params in ("A", "AV"):
#     plt.figure(figsize=(4, 3))
#     plt.step([-2, -1], [0, 0], c="tab:blue", label="estimate", alpha=0.5)
#     if params == "A":
#         for i in range(repeat):
#             plt.step(x_pts[i], y_pts_A[i], where="post", c="tab:blue", alpha=a)
#     else:
#         for i in range(repeat):
#             plt.step(x_pts[i], y_pts_AV[i], where="post", c="tab:blue", alpha=a)
#     if params == "A":
#         plt.step(x_pts_avg, y_pts_A_avg, where="post", c="black", label="mean estimate")
#     else:
#         plt.step(x_pts_avg, y_pts_AV_avg, where="post", c="black", label="mean estimate")
#     plt.plot(x, y, c="red", linestyle="dashed", label="truth")
#     if distr == "exp":
#         plt.xlim([0, 7])
#     elif distr == "lognorm":
#         plt.xlim([0, 25])
#     plt.xlabel(r"$\lambda$")
#     plt.ylabel("CDF")
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.savefig(f"{shape}_estimates_size_{distr}_{params}({n}).png", dpi=600)
#     plt.close()
#
# # The true biased size distribution CDF
# x = np.linspace(0, np.max(x_pts), 2000)
# if distr == "exp":
#     y = gamma.cdf(x, a=2, scale=1)  # The gamma(2, 1) distribution.
# elif distr == "lognorm":
#     y = lognorm.cdf(x, sigma, scale=np.exp(mu + sigma ** 2))  # The lognormal(mu + sigma^2, sigma) distribution.
#
# for params in ("A", "AV"):
#     plt.figure(figsize=(4, 3))
#     plt.step([-2, -1], [0, 0], c="tab:blue", label="estimate", alpha=0.5)
#     if params == "A":
#         for i in range(repeat):
#             plt.step(x_pts[i], y_pts_biased_A[i], where="post", c="tab:blue", alpha=a)
#     else:
#         for i in range(repeat):
#             plt.step(x_pts[i], y_pts_biased_AV[i], where="post", c="tab:blue", alpha=a)
#     if params == "A":
#         plt.step(x_pts_avg, y_pts_biased_A_avg, where="post", c="black", label="mean estimate")
#     else:
#         plt.step(x_pts_avg, y_pts_biased_AV_avg, where="post", c="black", label="mean estimate")
#     plt.plot(x, y, c="red", linestyle="dashed", label="truth")
#     if distr == "exp":
#         plt.xlim([0, 9])
#     elif distr == "lognorm":
#         plt.xlim([0, 30])
#     plt.xlabel(r"$\lambda$")
#     plt.ylabel("CDF")
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.savefig(f"{shape}_estimates_biased_size_{distr}_{params}({n}).png", dpi=600)
#     plt.close()


############################################ Example 6: errors of estimates above ######################################
shape = "cube"  # "tetrahedron" / "cube" / "dodecahedron"
N = [1000, 2000, 5000, 10000]  # [1000 / 2000 / 5000 / 10000]
############ Do not alter below ############
repeat = 100
i = 0
mu, sigma = 2, 0.5
errors = np.zeros((4*len(N), repeat), dtype=float)
errors_biased = np.zeros((4*len(N), repeat), dtype=float)

for n in N:
    for distr in ("exp", "lognorm"):
        data = pickle.load(open(f"examples/{shape}_estimates_{distr}({n}).pkl", "rb"))
        # Saved data from Example 5:
        # [0] = sizes, [1] = areas, [2] = v_pts, [3] = x_pts, [4] = y_pts_A, [5] = y_pts_AV,
        # [6] = y_pts_biased_A, [7] = y_pts_biased_AV, [8] = x_pts_avg, [9] = y_pts_A_avg, [10] = y_pts_AV_avg,
        # [11] = y_pts_biased_A_avg, [12] = y_pts_biased_AV_avg

        # True regular and biased distributions:
        x_true = np.linspace(0, np.max(data[3]), 20000)
        if distr == "exp":
            y_true = expon.cdf(x_true)  # The exp(1) distribution.
            y_true_biased = gamma.cdf(x_true, a=2, scale=1)  # The gamma(2, 1) distribution.
        elif distr == "lognorm":
            y_true = lognorm.cdf(x_true, sigma, scale=np.exp(mu))  # The lognormal(mu, sigma) distribution.
            y_true_biased = lognorm.cdf(x_true, sigma, scale=np.exp(mu + sigma ** 2))  # ln(mu + sigma^2, sigma).

        # Errors: supremum norms of estimates and average estimates.
        errors[i] = pu.error_estimate(x_true, y_true, data[3], data[4])  # A
        errors[i+1] = pu.error_estimate(x_true, y_true, data[3], data[5])  # AV
        errors_biased[i] = pu.error_estimate(x_true, y_true_biased, data[3], data[6])  # A
        errors_biased[i+1] = pu.error_estimate(x_true, y_true_biased, data[3], data[7])  # AV
        i += 2  # Next entries for errors(_biased) in next iteration.

# Entries in errors(_biased):
# [4n+0] = A & exp,     [4n+1] = AV & exp,    [4n+2] = A & lognorm,    [4n+3] = AV & lognorm

# Latex format results to be used in tabular. Header biased:
print("""\\rowcolor{gray!10}
        \multicolumn{2}{c|}{\\rule{0pt}{10pt}Estimates of $H^b$}
        &\multicolumn{2}{c|}{$\left\| \hat{H}_{n, A}^b - H^b \\right\|_\infty$}
        &\multicolumn{2}{c}{$\left\| \hat{H}_{n, A, V}^b - H^b \\right\|_\infty$} \\\\
        \\rowcolor{gray!10}
        $n$ &$H$ &mean error &$(2.5\%, 97.5\%)$ &mean error &$(2.5\%, 97.5\%)$ \\\\
        \hline""")
for n in range(len(N)):
    i = 4 * n  # Reset.
    for distr in ("Exponential", "Log-normal"):
        # Content rows biased:
        print(f"        ${N[n]}$ &\\textit{{{distr}}} &${errors_biased[i].mean():.4f}$ "
          f"&$({np.quantile(errors_biased[i], 0.025):.4f}, {np.quantile(errors_biased[i], 0.975):.4f})$ "
          f"&${errors_biased[i+1].mean():.4f}$ "
          f"&$({np.quantile(errors_biased[i+1], 0.025):.4f}, {np.quantile(errors_biased[i+1], 0.975):.4f})$ \\\\")
        i += 2
# Header regular:
print("""        \hline \hline
        \\rowcolor{gray!10}
        \multicolumn{2}{c|}{\\rule{0pt}{10pt}Estimates of $H$}
        &\multicolumn{2}{c|}{$\left\| \hat{H}_{n, A} - H \\right\|_\infty$}
        &\multicolumn{2}{c}{$\left\| \hat{H}_{n, A, V} - H \\right\|_\infty$} \\\\
        \\rowcolor{gray!10}
        $n$ &$H$ &mean error &$(2.5\%, 97.5\%)$ &mean error &$(2.5\%, 97.5\%)$ \\\\
        \hline""")
for n in range(len(N)):
    i = 4 * n  # Reset.
    for distr in ("Exponential", "Log-normal"):
        # Content rows regular:
        print(f"        ${N[n]}$ &\\textit{{{distr}}} &${errors[i].mean():.4f}$ "
          f"&$({np.quantile(errors[i], 0.025):.4f}, {np.quantile(errors[i], 0.975):.4f})$ "
          f"&${errors[i+1].mean():.4f}$ "
          f"&$({np.quantile(errors[i+1], 0.025):.4f}, {np.quantile(errors[i+1], 0.975):.4f})$ \\\\")
        i += 2
print(f"        \hline % {shape.capitalize()}")

################### pair-wise difference in error #########################
# [4n+0] = A & exp,     [4n+1] = AV & exp,    [4n+2] = A & lognorm,    [4n+3] = AV & lognorm
# Latex format results to be used in tabular. Header difference:
print("""\\rowcolor{gray!10}
        \multicolumn{2}{c|}{\\rule{0pt}{10pt}Difference in estimates}
        &\multicolumn{2}{c|}{$\left\| \hat{H}_{n, A, V}^b - H^b \\right\|_\infty - 
        \left\| \hat{H}_{n, A}^b - H^b \\right\|_\infty$}
        &\multicolumn{2}{c}{$\left\| \hat{H}_{n, A, V} - H \\right\|_\infty - 
        \left\| \hat{H}_{n, A} - H \\right\|_\infty$} \\\\
        \\rowcolor{gray!10}
        $n$ &$H$ &mean diff. &$(2.5\%, 97.5\%)$ &mean diff. &$(2.5\%, 97.5\%)$ \\\\
        \hline""")
for n in range(len(N)):
    i = 4 * n  # Reset.
    for distr in ("Exponential", "Log-normal"):
        # Content rows:
        print(f"        ${N[n]}$ &\\textit{{{distr}}} "
              f"&${(errors_biased[i+1] - errors_biased[i]).mean():.4f}$ "
              f"&$({np.quantile(errors_biased[i+1] - errors_biased[i], 0.025):.4f}, "
              f"{np.quantile(errors_biased[i+1] - errors_biased[i], 0.975):.4f})$ "
              f"&${(errors[i+1] - errors[i]).mean():.4f}$ "
              f"&$({np.quantile(errors[i+1] - errors[i], 0.025):.4f}, "
              f"{np.quantile(errors[i+1] - errors[i], 0.975):.4f})$ \\\\")
        i += 2
print(f"        \hline % {shape.capitalize()}")
