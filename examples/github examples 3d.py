import pysizeunfolder as pu
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pyvista as pv  # Install via: pip install vtk pyvista


# Authors: Thomas van der Jagt (1-2), Jeroen Faas (3-5)


# Example 1

rng = np.random.default_rng(0)
points = np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5],
                   [-0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]])
areas = pu.iur_3d_hull(points, n=1000000, rng=rng)
x, y = pu.approx_area_density(areas)

plt.figure(figsize=(4, 3))
plt.hist(areas, bins=80, ec='black', linewidth=0.2, density=True)
plt.plot(x, y)
plt.xlim(0, 1.5)
plt.ylim(0, 3)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.14)
plt.xlabel("Area")
plt.ylabel("Density")
plt.savefig("cube estimate.png", dpi=600)
#plt.show()

# Various code examples

areas = pu.iur_3d_shape("dodecahedron", n=10000)
x, y = pu.approx_area_density(np.sqrt(areas), sqrt_data=True)
# This is a halfspace representation of the centered unit cube
halfspaces = np.array([[0, 0, 1, -0.5], [0, 0, -1, -0.5], [1, 0, 0, -0.5],
                       [-1, 0, 0, -0.5], [0, 1, 0, -0.5], [0, -1, 0, -0.5]], dtype=np.double)
origin = np.array([0, 0, 0], dtype=np.double)
areas, sections = pu.iur_3d_half(halfspaces, origin, n=10, return_vertices=True, rng=rng)


# Example 2

rng = np.random.default_rng(2)
points = rng.uniform(low=-0.5, high=0.5, size=(15, 3))
area, section = pu.iur_3d_hull(points, 1, return_vertices=True, rng=rng)
faces = pu.faces_3d_hull(points)

faces = np.hstack([[len(face)] + face for face in faces])
polygon_mesh = pv.PolyData(points, faces)
section_mesh = pv.PolyData(section[0], [len(section[0])] + list(range(len(section[0]))))

pv.set_plot_theme('document')
p = pv.Plotter(window_size=(1000, 1000))
p.add_mesh(polygon_mesh, style="surface", show_scalar_bar=False, lighting=False, show_edges=True, color="tab:blue",
           opacity=0.35, line_width=2)
p.add_mesh(section_mesh, style="surface", show_scalar_bar=False, lighting=False, show_edges=True, color="tab:red",
           line_width=2)
p.camera.elevation = -15
p.camera.azimuth = 5
p.camera.zoom(1.3)
p.enable_anti_aliasing()
p.show()
#p.show(screenshot="random polyhedron.png")

# Example 3: cube particle
rng = np.random.default_rng(26)
points = np.array([[0,0,1], [0,1,1], [0,1,0], [0,0,0], [1,0,0], [1,0,1], [1,1,1], [1,1,0]], dtype=np.float64)
area, section = pu.iur_3d_hull(points, 1, return_vertices=True, rng=rng)
faces = pu.faces_3d_hull(points)

faces = np.hstack([[len(face)] + face for face in faces])
polygon_mesh = pv.PolyData(points, faces)
section_mesh = pv.PolyData(section[0], [len(section[0])] + list(range(len(section[0]))))

pv.set_plot_theme('document')
p = pv.Plotter(window_size=(1000, 1000))
p.add_mesh(polygon_mesh, style="surface", show_scalar_bar=False, lighting=False, show_edges=True, color="tab:blue",
           opacity=0.35, line_width=2)
p.add_mesh(section_mesh, style="surface", show_scalar_bar=False, lighting=False, show_edges=True, color="tab:red",
           line_width=2)
p.camera.elevation = -15
p.camera.azimuth = 5
p.camera.zoom(1.3)
p.enable_anti_aliasing()
p.show()

# Example 4: dodecahedron particle
rng = np.random.default_rng(26)
phi = (1 + np.sqrt(5)) * 0.5
points = np.array(
        [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [1, -1, 1],
        [0, phi, 1. / phi], [0, -phi, 1. / phi], [0, phi, -1. / phi], [0, -phi, -1. / phi],
        [1. / phi, 0, phi], [-1. / phi, 0, phi], [1. / phi, 0, -phi], [-1. / phi, 0, -phi],
        [phi, 1. / phi, 0], [-phi, 1. / phi, 0], [phi, -1. / phi, 0], [-phi, -1. / phi, 0]])
area, section = pu.iur_3d_hull(points, 1, return_vertices=True, rng=rng)
faces = pu.faces_3d_hull(points)

faces = np.hstack([[len(face)] + face for face in faces])
polygon_mesh = pv.PolyData(points, faces)
section_mesh = pv.PolyData(section[0], [len(section[0])] + list(range(len(section[0]))))

pv.set_plot_theme('document')
p = pv.Plotter(window_size=(1000, 1000))
p.add_mesh(polygon_mesh, style="surface", show_scalar_bar=False, lighting=False, show_edges=True, color="tab:blue",
           opacity=0.35, line_width=2)
p.add_mesh(section_mesh, style="surface", show_scalar_bar=False, lighting=False, show_edges=True, color="tab:red",
           line_width=2)
p.camera.elevation = -15
p.camera.azimuth = 5
p.camera.zoom(1.3)
p.enable_anti_aliasing()
p.show()


######################################## Example 5: numbers of vertices ################################################
# n = int(1e6)
# rng = np.random.default_rng(2)
# Random particle, generated as convex hull of 15 random points in the unit cube centered about the origin.
# points = rng.uniform(low=-0.5, high=0.5, size=(15, 3))
# areas, sections = pu.iur_3d_hull(points, n, return_vertices=True, rng=rng)
# Alternatively: Cube particle, the unit cube centered about the origin.
# areas, sections = pu.iur_3d_shape("cube", n, return_vertices=True, rng=rng)
# Alternatively: Dodecahedron particle, centered about the origin.
# areas, sections = pu.iur_3d_shape("dodecahedron", n, return_vertices=True, rng=rng)
# Number of vertices for each section profile.
# num_verts = pu.number_vertices(sections)

# Alternatively: Load from file
shape = "dodecahedron"
reference_sample = pickle.load(open(f"examples/{shape}_sample.pkl", "rb"))
# Area and number of vertices for each section profile.
areas, num_verts = reference_sample[0], reference_sample[1]
Smax = np.sqrt(areas.max())  # Max value of x-axis

# Fractions of each number of vertices that occurs.
fractions = pu.count_pos_numbers(num_verts, fractions=True)

# x_axis: List of all numbers of vertices.
x_axis = list(range(num_verts.min(), num_verts.max() + 1))
# Extract sqrt areas with each number of vertices for box plot.
boxes = []
for i in x_axis:
    boxes.append(pu.data_n_vertices(np.sqrt(areas), num_verts, i))

# Figure of number of vertices density bar plot.
plt.figure()
plt.bar(x_axis, fractions, edgecolor='black')
plt.xticks(x_axis)
plt.xlabel("Number of vertices")
plt.ylabel("Density")
plt.savefig("num_verts_bars.png", dpi=600)
plt.close()

# Figure of number of vertices vs sqrt area box plot.
plt.figure()
plt.boxplot(boxes, tick_labels=x_axis, flierprops=dict(marker='.', markerfacecolor="blue", markeredgecolor="blue",
                                                       markersize=5, alpha=0.01))
plt.xlabel("Number of vertices")
plt.ylabel("Square root of area")
plt.savefig("num_verts_vs_area_sqrt.png", dpi=1200)
plt.close()

# Figure of number of vertices vs sqrt area density histogram.
plt.figure()
plt.hist(boxes, bins=100, histtype="barstacked", label=x_axis, ec="black", linewidth=0.2, density=True)
plt.xlim(0, Smax)
plt.xlabel("Square root of area")
plt.ylabel("Density")
plt.legend(title="Number of vertices")
plt.savefig("area_sqrt_per_num_verts_hist.png", dpi=600)
plt.close()

# Figure of sqrt area density histogram with specific number of v vertices.
v = 4
rest = np.array([])
for i in range(len(boxes)):
    if i != v-3:
        rest = np.append(rest, boxes[i])

plt.figure()
plt.hist(np.append(-1e-2-rest, boxes[v-3]), bins=200, label=x_axis[v-3], ec="black", color="tab:orange", linewidth=0.2,
         density=True)
plt.xlim(0, Smax)
plt.ylim(0, 0.5)
plt.xlabel("Square root of area")
plt.ylabel("Density")
# plt.legend(title="Number of vertices")
plt.savefig(f"{shape}_area_sqrt_num_verts={v}_hist.png", dpi=600)
plt.close()
