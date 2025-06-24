from math import isclose
import numpy as np
from numpy.typing import NDArray as ndarray

# Author: Jeroen Faas


def aspect_ratio(vertices: ndarray[float], return_points: bool = False) -> float | tuple[float, ndarray[float]]:
    """
    A complete procedure to calculate the aspect ratio in a section profile.
    :param vertices: A numpy.ndarray of shape (N, 3), containing the 3D vertices that make up the section profile.
    :param return_points: A boolean, indicating whether or not to return end points of the major and minor axes of the
            aspect ratio.
    :return: If "return_points" is False: the value of the aspect ratio, calculated as the length of major divided by
            the length of minor axis. If "return_points" is True: the value of the aspect ratio, along with an
            numpy.ndarray of shape (4, 3), containing 3D end points of first the major and second the minor axis.
    """
    # Find maximal vertex distance: the major axis length.
    d_max, indices_max = max_distance(vertices)
    # The vertices that attain maximal distance, Vertex_1, Vertex_2: The end points of the major axis.
    Vertex_1, Vertex_2 = vertices[indices_max[0]], vertices[indices_max[1]]
    # V_middle: The middle point of the major axis.
    V_middle = (Vertex_1 + Vertex_2) / 2

    # Find two normal basis vectors for planar section in 3D: first along, second perpendicular to major axis.
    V_basis = section_basis_vectors(vertices, indices_max)
    # Project 3D vertices of section profile onto 2D. V_middle is projected to the origin. V_basis[0] is the x-axis,
    #   V_basis[1] is the y-axis.
    vertices_2D = projection_2D_section(vertices, V_middle, V_basis)
    # Find 2D vertices that share an edge that crosses the y-axis.
    indices_orth = sides_split_projected(vertices_2D)

    # y_axis: Definition of the y-axis in 2D by two points that it runs through.
    y_axis = np.stack(([0, 0], [0, 1]))
    # P1_2D, P2_2D: 2D points corresponding to the intersections of the y-axis and edges that cross the y-axis.
    P1_2D = intersect_edge_2D(vertices_2D, indices_orth[:2], ref_point=y_axis[0], ref_vector=y_axis[1])
    P2_2D = intersect_edge_2D(vertices_2D, indices_orth[2:], ref_point=y_axis[0], ref_vector=y_axis[1])
    # Project the 2D points back to 3D points, Point_1, Point_2: The end points of the minor axis.
    Point_1 = V_middle + P1_2D[0] * V_basis[0] + P1_2D[1] * V_basis[1]
    Point_2 = V_middle + P2_2D[0] * V_basis[0] + P2_2D[1] * V_basis[1]

    # Find orthogonal distance to the maximal vertex distance: the minor axis length.
    d_orth = dist(Point_1, Point_2)

    # Return desired values.
    if not return_points:
        return d_max / d_orth
    return d_max / d_orth, np.stack((Vertex_1, Vertex_2, Point_1, Point_2), dtype=float)


def dist(point_1: ndarray[float], point_2: ndarray[float]) -> float | None:
    """
    Calculates the distance between two points, defined as Euclidian norm of their difference, if they have the same
    dimension.
    :param point_1: A numpy.ndarray of shape (M,), containing the coordinates of the first point. Here, M is the
            dimension of the point.
    :param point_2: A numpy.ndarray of shape (N,), containing the coordinates of the second point. Here, N is the
            dimension of the point.
    :return: The distance between the points, if their dimensions are equal (M = N). Otherwise, returns None.
    """
    # Check whether dimensions correspond.
    if len(point_1) != len(point_2):
        raise Exception("Expected dimensions to be equal, but got dimensions", len(point_1), "and", len(point_2))
        return None

    D = 0
    for k in range(len(point_1)):
        D += (point_2[k] - point_1[k]) ** 2

    return np.sqrt(D)


def intersect_edge_2D(vertices: ndarray[float], indices: list[int], ref_point: ndarray[float] = None,
                      ref_vector: ndarray[float] = None) -> ndarray[float] | bool:
    """
    A method for finding the intersection of an edge between given vertices and the line passing through a given
    reference point along direction of a given reference vector.
    :param vertices: A numpy.ndarray of shape (N, 2), containing the 2-dimensional vertex coordinates. Here, N is the
            number of vertices.
    :param indices: A list of length 2, containing indices in "vertices" of the pair of vertices that share the edge to
            be considered.
    :param ref_point: A numpy.ndarray of shape (2,), containing the coordinates of the reference point for the
            intersection line. Defaults to the origin (0, 0).
    :param ref_vector: A numpy.ndarray of shape (2,), being the reference vertex with direction for the intersection
            line. Defaults to the first unit vector (1, 0).
    :return: A numpy.ndarray of shape (2,), containing the coordinates of the intersection, if it exists. Returns
            False otherwise.
    """
    # Default values if applicable.
    if ref_point is None:
        ref_point = np.zeros(2, dtype=float)
    if ref_vector is None:
        ref_vector = np.array((1, 0), dtype=float)

    # Line through M along v.
    line_Mv = np.stack((ref_point, ref_point + ref_vector))
    # Line along the edge in question.
    line_edge = np.stack((vertices[indices[0]], vertices[indices[1]]))

    # P: The intersection point if it exists, False otherwise.
    P = intersect_line_2D(line_Mv, line_edge)
    if P is None:
        raise Exception("Intersection does not exist.")
        return None

    return P


def intersect_line_2D(line_1: ndarray[float], line_2: ndarray[float]) -> ndarray[float] | bool:
    """
    A method to find the intersection between two lines in 2D. The lines are defined by two points to intersect with.
    :param line_1: A numpy.ndarray of shape (2, 2), containing two points for the first line to intersect with.
    :param line_2: A numpy.ndarray of shape (2, 2), containing two points for the second line to intersect with.
    :return: A numpy.ndarray of shape (2,), containing the coordinates of the intersection, if it exists. Returns
            False otherwise.
    """
    # diff: Differences in x- and y-coordinates between points.
    diff = np.transpose(np.array((line_1[0] - line_1[1], line_2[0] - line_2[1]), dtype=float))

    def det(a: ndarray[float], b: ndarray[float]) -> float:
        return a[0] * b[1] - a[1] * b[0]

    # denom: Expression for the determinant of the system.
    denom = det(diff[0], diff[1])
    # If zero, then no intersection exists, so return False.
    if denom == 0:
        raise Exception("Expected determinant to be nonzero, but got", denom, "instead.")
        return None

    d = np.array((det(line_1[0], line_1[1]), det(line_2[0], line_2[1])), dtype=float)

    return np.array((det(d, diff[0]) / denom, det(d, diff[1]) / denom), dtype=float)


def max_distance(vertices: ndarray[float]) -> tuple[float, list[int]]:
    """
    Computes the maximal distance among a given set of vertices in any dimension.
    :param vertices: A numpy.ndarray of shape (N, K), containing the coordinates of all vertices to be considered.
            Here, N is the number of vertices, K is the dimension of the vertices.
    :return: The maximal distance among vertices, and a list of length 2 containing indices for "vertices" of the
            vertices that attain this maximal distance. In the case that multiple pairs of vertices attain this exact
            same maximal distance, the last such pair that occurs in "vertices" in indicial order is returned.
    """
    dist_max: float = 0
    indices = [0, 0]

    for i in range(len(vertices) - 1):
        for j in range(i + 1, len(vertices)):
            # d_ij: The distance between vertex i and vertex j.
            d_ij = dist(vertices[i], vertices[j])

            # Update maximum and indices if applicable.
            if d_ij and d_ij >= dist_max:
                dist_max = d_ij
                indices = [i, j]

    return dist_max, indices


def projection_2D_section(vertices_3D: ndarray[float], M: ndarray[float], V: ndarray[float]) -> ndarray[float]:
    """
    A method that projects a section profile in 3D to 2D. The projection is at the same scale as the original, given
    that the input parameters are correct.
    :param vertices_3D: A numpy.ndarray of shape (N, 3), containing the 3-dimensional vertices that make up the section
            profile. Here, N is the number of vertices.
    :param M: A numpy.ndarray of shape (3,), containing the 3-dimensional coordinates of the point that will be
            projected onto the origin in 2D. This point must be in the planar section.
    :param V: A numpy.ndarray of shape (2, 3), containing the normal basis vectors that will determine the 2D projection
            coordinates. These vectors must be parallel to the planar section.
    :return: A numpy.ndarray of shape (N, 2), containing the 2-dimensional vertices that make up the section profile.
            The vertex order remains the same. Here, N is the number of vertices.
    """
    vertices_2D = np.empty((len(vertices_3D), 2), dtype=float)

    # For each vertex, perform the projection to the new 2D coordinates.
    for i in range(len(vertices_3D)):
        vertices_2D[i] = [np.dot(vertices_3D[i] - M, V[0]), np.dot(vertices_3D[i] - M, V[1])]

    return vertices_2D


def projection_3D_point(point_2D: ndarray[float], M: ndarray[float], V: ndarray[float]) -> ndarray[float]:
    """
    Projects a 2-dimensional point to a point in the given planar section in 3D. The planar section is defined with "M"
    as reference middle point and linear combinations of reference vectors in "V" as coordinate system.
    :param point_2D: A numpy.ndarray of shape (2,), containing coordinates of the point in 2D to consider.
    :param M: A numpy.ndarray of shape (3,), containing coordinates of the reference middle point in 3D. The
            2-dimensional origin is projected onto this point in 3D.
    :param V: A numpy.ndarray of shape (2, 3), containing the normal basis vectors that will determine the 3D projection
            coordinates. The first coordinate of the point is the magnitude along "V[0]" and the second coordinate of
            the point is the magnitude along "V[1]" in the linear combination.
    :return: A numpy.ndarray of shape (3,), containing coordinates of the projected point in 3D. This is the given
            linear combination of vectors, shifted by reference point "M".
    """
    return M + point_2D[0] * V[0] + point_2D[1] * V[1]


def section_basis_vectors(vertices: ndarray[float], indices: list[int]) -> ndarray[float]:
    """
    Computes two vectors that form an orthonormal basis for the planar section. The first vector follows the direction
    of the maximal distance among vertices, the second vector being a perpendicular direction within the section plane.
    :param vertices: A numpy.ndarray of shape (N, K), containing the coordinates of all vertices to be considered.
            Here, N is the number of vertices, K is the dimension of the vertices.
    :param indices: A list of length 2, containing indices for "vertices" of the vertices that attain the maximal
            distance.
    :return: A numpy.ndarry of shape (2, K), containing the normal basis vectors. Here, K is the dimension of the
            vectors.
    """
    # A, B: Vertices that attain the maximal distance.
    A, B = vertices[indices[0]], vertices[indices[1]]
    # M: Their middle point.
    M = (A + B) / 2

    # v_unit: The normal vector with direction parallel to A - B.
    v_unit = A - B
    v_unit = v_unit / np.linalg.norm(v_unit)

    # v_orth: A normal vector with direction orthogonal to v_unit, still parallel to the planar section.
    j = 0
    while j in indices:
        j += 1
    v_orth = vertices[j] - M - np.dot(vertices[j] - M, v_unit) * v_unit
    v_orth = v_orth / np.linalg.norm(v_orth)

    return np.stack((v_unit, v_orth))


def sides_split(vertices: ndarray[float], M: ndarray[float] = None, v: ndarray[float] = None) -> list[int]:
    """
    A method to identify the edges of a section profile that cross between two sides of the section. The sides are
    defined as projection of positive against negative length along given vector "v", starting from point "M". This is
    equivalent to splitting the section profile by sides of the line perpendicular to "v" that passes through "M".
    :param vertices: A numpy.ndarray of shape (N, K), containing the coordinates of all vertices of the section profile,
            in a clockwise- or counterclockwise-order. Here, N is the number of vertices, K is the dimension of the
            vertices.
    :param M: A numpy.ndarray of shape (K,), containing the coordinates of the reference point for splitting sides.
            Here, K is the dimension of the vertices. Defaults to the K-dimensional origin.
    :param v: A numpy.ndarray of shape (K,), being the vertex with reference direction for splitting sides. Here, K is
            the dimension of the vertices. Defaults to the first K-dimensional unit vector.
    :return: A list of length 2L, containing the indices in "vertices" of the L pairs of vertices that share a crossing
            edge. If the first and last vertex in the order share a crossing edge, the pair is listed first, with -1 as
            index for the last vertex.
    """
    indices: list[int] = []
    K = vertices.shape[1]
    # Default values if applicable.
    if M is None:
        M = np.zeros((1, K), dtype=float)
    if v is None:
        v = np.zeros((1, K), dtype=float)
        v[0] = 1

    # signs: The signs of the vertices when projected along v.
    signs = np.empty((len(vertices), ), dtype=int)

    # Project each vertex along v.
    for i in range(len(vertices)):
        signs[i] = np.sign(np.dot(vertices[i] - M, v))

    for i in range(len(signs)):
        # Track down sign changes between previous and current entry and save their indices.
        if signs[i - 1] != signs[i]:
            indices.append(i - 1)
            indices.append(i)

    return indices

def sides_split_projected(vertices_2D: ndarray[float]) -> list[int]:
    """
    A simplified method to identify the edges of a projected section profile that cross between two sides of the
    section. The sides are defined as positive against negative value of the first coordinate of a vertex. If the
    projection is done using the vectors resulting from "section_basis_vectors", this is equivalent to splitting the
    section profile by sides of the line perpendicular to the maximal vertex distance line (the maximal vertex distance
    goes along the first vector direction, the perpendicular direction being the second vector resulting from
    "section_basis_vectors"), that passes through the middle point between those vertices.
    :param vertices_2D: A numpy.ndarray of shape (N, K), containing the coordinates of all vertices of the section
            profile, in a clockwise- or counterclockwise-order. Here, N is the number of vertices, K is the dimension of
            the vertices.
    :return: A list of length 2L, containing the indices in "vertices" of the pair of vertices that share a crossing
            edge. If the first and last vertex in the order share a crossing edge, the pair is listed first, with -1 as
            index for the last vertex.
    """
    indices: list[int] = []
    # signs: The signs of the vertices when projected along the first direction.
    signs = np.empty((len(vertices_2D),), dtype=int)

    # If the projection is done properly, we can simply check the signs of the first coordinate.
    signs = np.sign(vertices_2D)[:, 0]

    for i in range(len(signs)):
        # Track down sign changes between previous and current entry and save their indices.
        if signs[i - 1] != signs[i]:
            indices.append(i - 1)
            indices.append(i)

    return indices
