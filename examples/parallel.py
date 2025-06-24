import os
import numpy as np
import pysizeunfolder as pu
from joblib import Parallel, delayed
# import matplotlib.pyplot as plt
import pickle


# Author: Thomas van der Jagt

shape = "tetrahedron"  # "tetrahedron" / "cube" / "dodecahedron"
n = 10000000

if shape == "cube":
    # Reference particle: cube of unit volume
    points = np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                       [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]])
elif shape == "dodecahedron":
    # Reference particle: dodecahedron (not yet) of unit volume
    phi = (1 + np.sqrt(5)) * 0.5
    points = np.array(
            [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [1, -1, 1],
             [0, phi, 1. / phi], [0, -phi, 1. / phi], [0, phi, -1. / phi], [0, -phi, -1. / phi],
             [1. / phi, 0, phi], [-1. / phi, 0, phi], [1. / phi, 0, -phi], [-1. / phi, 0, -phi],
             [phi, 1. / phi, 0], [-phi, 1. / phi, 0], [phi, -1. / phi, 0], [-phi, -1. / phi, 0]])
    # ^ Dodecahedron with edge length 2/phi => volume = 4 phi sqrt(5)
    # Normalised to areas of dodecahedron with unit volume by method
elif shape == "tetrahedron":
    # Reference particle: tetrahedron (not yet) of unit volume
    points = np.array([[1, 0, -1./np.sqrt(2)], [-1, 0, -1./np.sqrt(2)], [0, 1, 1./np.sqrt(2)], [0, -1, 1./np.sqrt(2)]])
    # Normalised to areas of tetrahedron with unit volume by method
else:
    print("invalid shape provided.")

ss = np.random.SeedSequence(0)
num_cpus = os.cpu_count()
child_seeds = ss.spawn(num_cpus)
streams = [np.random.default_rng(s) for s in child_seeds]

block_size = n // num_cpus
remainder = n - num_cpus*block_size
sizes = [block_size]*num_cpus
sizes[-1] += remainder

res = Parallel(n_jobs=num_cpus)(delayed(pu.iur_3d_hull)(points, sizes[i], True, streams[i], True)
                                for i in range(num_cpus))
# res = np.concatenate(res)

pos = 0  # Counter for position in resulting arrays.
areas = np.zeros(n, dtype=float)
secs = np.empty(n, dtype=list)

for i in range(num_cpus):
    # Part of res done by cpu i.
    li = len(res[i][0])
    areas[pos:pos + li] = res[i][0]
    secs[pos:pos + li] = res[i][1]
    pos += li

# Obtain the number of vertices for each section.
num_verts = pu.number_vertices(secs)

# Obtain the aspect ratio for each section.
# ratios = np.zeros(n, dtype=float)
# for i in range(n):
#     # Aspect ratio for each section:
#     ratios[i] = pu.aspect_ratio(secs[i])

f = open(f"{shape}_sample({int(np.log10(n))}).pkl", "wb")
pickle.dump([areas, num_verts], f)  # Include/Exclude aspect ratios.
f.close()

# x, y = pu.approx_area_density(areas)

# plt.figure()
# plt.hist(areas, bins=80, ec='black', linewidth=0.2, density=True)
# plt.plot(x, y)
# plt.xlim(0, 1.5)
# plt.ylim(0, 3)
# plt.show()
