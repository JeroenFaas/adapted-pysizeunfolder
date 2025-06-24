from .algorithms import hybrid_icm_em, icm, em
from .iur_3d import approx_area_density
import numpy as np
from scipy.interpolate import CubicSpline
from statsmodels.distributions import ECDF
from math import pi, sqrt


def gs_sphere(x):
    c = (9*pi/16)**(1./3)
    y = np.zeros(len(x))
    indices = x*x < c
    y[indices] = x[indices]/(c*np.sqrt(1-x[indices]*x[indices]/c))
    return y


def sphere_G(x):
    R = (3./(4*pi))**(1./3)
    y = np.zeros(len(x))
    y[x >= sqrt(pi)*R] = 1
    x_ = x[x < sqrt(pi)*R]
    y[x < sqrt(pi)*R] = 1 - (1./R)*np.sqrt(R*R - x_*x_/pi)
    return y


def de_bias_sphere(x_pts, estimate, G_cdf):
    n = len(x_pts)
    observed_x = np.linspace(0, 1, n)
    distances = np.zeros(n)
    Hb_probabilities = np.append(estimate[0], np.diff(estimate))

    FS_estimates = np.zeros((n, n))
    for i in range(n):
        kernel = G_cdf(x_pts[i] / x_pts)
        divisor = np.cumsum(np.flip(Hb_probabilities))
        divisor[divisor == 0.0] = 1.0
        FS_estimates[:, i] = np.flip(np.cumsum(np.flip(kernel * Hb_probabilities)) / divisor)

    for i in range(n):
        term1 = np.abs(observed_x - FS_estimates[i, :])[:(n - 1)]
        term2 = np.abs(observed_x[1:] - FS_estimates[i, :(n - 1)])
        distances[i] = 0.5 * np.dot(term1 + term2, np.diff(x_pts))

    trunc_ix = np.argmin(distances)
    temp_hb = np.copy(Hb_probabilities)
    temp_hb[:trunc_ix] = 0.0
    h_est = np.cumsum(temp_hb / x_pts)
    h_est = h_est / h_est[-1]
    return h_est


def de_bias(x_pts, estimate, reference_sample):
    """
    A separate function which performs the de-biasing step which may also be performed by the function
    pysizeunfolder.estimate_size after the optimization procedure.

    :param x_pts: A numpy.ndarray of shape (n,) representing the points at which the CDF is evaluated. Such as x_pts
        returned by the function pysizeunfolder.estimate_size.
    :param estimate: A numpy.ndarray of shape (n,) representing the values the CDF takes in x_pts. Such as y_pts
        returned by the function pysizeunfolder.estimate_size.
    :param reference_sample: A sample of section areas of the reference shape, a numpy.ndarray of shape (N,) where N is
        the sample size. Ideally N is very large.
    :return: A numpy arrays: y_pts. A numpy.ndarray of shape (n,). This arrays represent a piece-wise constant
    distribution functions. The CDF is constant on [x_pts[i],x_pts[i+1]). Here: [a,b) = {x: a <= x < b}.
        y_pts[i] is the CDF value in x_pts[i].
    """
    n = len(x_pts)
    observed_x = np.linspace(0, 1, n)
    distances = np.zeros(n)
    Hb_probabilities = np.append(estimate[0], np.diff(estimate))
    GS_cdf = ECDF(np.sqrt(reference_sample))

    FS_estimates = np.zeros((n, n))
    for i in range(n):
        kernel = GS_cdf(x_pts[i] / x_pts)
        divisor = np.cumsum(np.flip(Hb_probabilities))
        divisor[divisor == 0.0] = 1.0
        FS_estimates[:, i] = np.flip(np.cumsum(np.flip(kernel * Hb_probabilities)) / divisor)

    for i in range(n):
        term1 = np.abs(observed_x - FS_estimates[i, :])[:(n - 1)]
        term2 = np.abs(observed_x[1:] - FS_estimates[i, :(n - 1)])
        distances[i] = 0.5 * np.dot(term1 + term2, np.diff(x_pts))

    trunc_ix = np.argmin(distances)
    temp_hb = np.copy(Hb_probabilities)
    temp_hb[:trunc_ix] = 0.0
    h_est = np.cumsum(temp_hb / x_pts)
    h_est = h_est / h_est[-1]
    return h_est


# def de_bias_AV_older(x_pts, v_pts, estimate, reference_sample):
#     """
#     A separate function which performs the de-biasing step which may also be performed by the function
#     pysizeunfolder.estimate_size_AV after the optimization procedure. This function is an altered version of "de_bias",
#     incorporating number of vertices data.
#
#     :param x_pts: A numpy.ndarray of shape (n,) representing the points at which the CDF is evaluated. Such as x_pts
#         returned by the function pysizeunfolder.estimate_size_AV.
#     :param v_pts: A numpy.ndarray of shape (n,) representing the number of vertices corresponding to a point in "x_pts".
#         Such as v_pts returned by the function pysizeunfolder.estimate_size_AV.
#     :param estimate: A numpy.ndarray of shape (n,) representing the values the biased CDF takes in x_pts. Such as y_pts
#         returned by the function pysizeunfolder.estimate_size_AV.
#     :param reference_sample: A sample of pairs of section profile areas and numbers of vertices of the reference shape,
#         in a list of two numpy.ndarrays of shape (N,), the first containing areas and the second containing numbers of
#         vertices, where N is the sample size. Ideally N is very large.
#     :return: y_pts: a numpy.ndarray of shape (n,). This array represents a piece-wise constant distribution function,
#         which is the unbiased version of the input function "estimate". The CDF is constant on [x_pts[i],x_pts[i+1]).
#         Here: [a,b) = {x: a <= x < b}. y_pts[i] is the CDF value in x_pts[i].
#     """
#     n = len(x_pts)
#     Vmax = reference_sample[1].max()
#     Pv = []
#     G_per_v_cdf = []
#     for v in range(Vmax - 2):
#         Pv.append(sum(reference_sample[1] - 3 <= v) / len(reference_sample[1]))  # Probability that V <= v + 3.
#         G_per_v_cdf.append(ECDF(np.sqrt(reference_sample[0][reference_sample[1] - 3 <= v])))
#
#     Hb_probabilities = np.append(estimate[0], np.diff(estimate))  # ^H_n^b(s_(j)) - ^H_n^b(s_(j-1)) for all j.
#     FS_estimates = np.zeros((n, n))
#     for i in range(n):
#         # kernel: G_K(s_i / s_(j), v_i) * P(V <= v) for all s_(j).
#         kernel = G_per_v_cdf[v_pts[i]-3](x_pts[i] / x_pts) * Pv[v_pts[i]-3]
#         divisor = np.cumsum(np.flip(Hb_probabilities))
#         divisor[divisor == 0.0] = 1.0
#         FS_estimates[:, i] = np.flip(np.cumsum(np.flip(kernel * Hb_probabilities)) / divisor)  # ^F_n(s_i, v_i)
#
#     observed_x = np.linspace(0, 1, n)
#     distances = np.zeros(n)
#     for i in range(n):
#         term1 = np.abs(observed_x - FS_estimates[i, :])[:(n - 1)]
#         term2 = np.abs(observed_x[1:] - FS_estimates[i, :(n - 1)])
#         distances[i] = 0.5 * np.dot(term1 + term2, np.diff(x_pts))
#
#     # Truncate start of the estimate until trunc_ix.
#     trunc_ix = np.argmin(distances)  # Truncation param ^t_n = x_pts[trunc_ix] = s_(trunc_ix)
#     temp_hb = np.copy(Hb_probabilities)
#     temp_hb[:trunc_ix] = 0.0  # Truncated ^H_n^b
#
#     h_est = np.cumsum(temp_hb / x_pts)  # Numerator of truncated ^H_n
#     h_est = h_est / h_est[-1]  # Truncated ^H_n
#     return h_est


# def de_bias_AV_old(x_pts, v_pts, estimate, reference_sample):
#     """
#     A separate function which performs the de-biasing step which may also be performed by the function
#     pysizeunfolder.estimate_size_AV after the optimization procedure. This function is an altered version of "de_bias",
#     incorporating number of vertices data.
#
#     :param x_pts: A numpy.ndarray of shape (n,) representing the points at which the CDF is evaluated. Such as x_pts
#         returned by the function pysizeunfolder.estimate_size_AV.
#     :param v_pts: A numpy.ndarray of shape (n,) representing the number of vertices corresponding to a point in "x_pts".
#         Such as v_pts returned by the function pysizeunfolder.estimate_size_AV.
#     :param estimate: A numpy.ndarray of shape (n,) representing the values the biased CDF takes in x_pts. Such as y_pts
#         returned by the function pysizeunfolder.estimate_size_AV.
#     :param reference_sample: A sample of pairs of section profile areas and numbers of vertices of the reference shape,
#         in a list of two numpy.ndarrays of shape (N,), the first containing areas and the second containing numbers of
#         vertices, where N is the sample size. Ideally N is very large.
#     :return: y_pts: a numpy.ndarray of shape (n,). This array represents a piece-wise constant distribution function,
#         which is the unbiased version of the input function "estimate". The CDF is constant on [x_pts[i],x_pts[i+1]).
#         Here: [a,b) = {x: a <= x < b}. y_pts[i] is the CDF value in x_pts[i].
#     """
#     n = len(x_pts)
#     Vmax = reference_sample[1].max()
#
#     # Compute matrix for ~G_K:
#     s_mat = np.broadcast_to(x_pts, (n, n))
#     s_frac_mat = s_mat.T / s_mat
#     G_estimates = np.zeros((Vmax - 2, n, n), dtype=float)  # Entry j,k,l: ~G_K(s_k / s_l, v_j)
#     for v in range(Vmax - 2):
#         Pv = sum(reference_sample[1] - 3 <= v) / len(reference_sample[1])  # Probability that V <= v + 3.
#         G_per_v_cdf = ECDF(np.sqrt(reference_sample[0][reference_sample[1] - 3 <= v]))  # Emperical approx ~G_K(., v)
#         # Compute all entries with v vertices.
#         G_estimates[v] = np.reshape(G_per_v_cdf(s_frac_mat.flatten()), (n, n)) * Pv
#
#     # Compute matrix for truncated ^H_n^b:
#     Hb_truncated = np.zeros((n, n + 1), dtype=float)  # Entry i,l+1: ^H_n^b(s_l; t = s_i)
#     for i in range(n-1):
#         if estimate[i] < 1:
#             # Truncated Hb only nonzero if s_l > s_i <=> l > i.
#             # est[i] = ^H_n^b(s_i) => Entry i,l+1: (est[l] - est[i]) / (1 - est[i])
#             Hb_truncated[i, i+2:] = (estimate[i+1:] - estimate[i]) / (1 - estimate[i])
#     # Compute matrix for diffs in consecutive entries along last axis.
#     Hb_probabilities = np.diff(Hb_truncated)  # Entry i,l: ^H_n^b(s_l; s_i) - ^H_n^b(s_l-1; s_i)
#
#     # # (OLD) Compute matrices for ^F_n and Fbar_n:
#     # F_estimates = np.zeros((n, Vmax-2, n), dtype=float)  # Entry i,j,k: ^F_n(s_k, v_j; t = s_i)
#     # Fbar = np.zeros((Vmax-2, n), dtype=int)  # Entry j,k: Fbar_n(s_k, v_j)
#     # for k in range(n):
#     #     for j in range(Vmax - 2):
#     #         for i in range(n):
#     #             F_estimates[i, j, k] = sum(G_estimates[j, k] * Hb_probabilities[i])
#     #             Fbar[j, k] += (i <= k) * (v_pts[i]-3 <= j)
#     # Fbar = Fbar / n
#
#     # Compute matrix for ^F_n:
#     # Entry i,j,k: ^F_n(s_k, v_j; t = s_i) = sum{l}(G_estimates[j, k, l] * Hb_probabilities[i, l])
#     F_estimates = np.einsum('jkl,il->ijk', G_estimates, Hb_probabilities)
#
#     # Compute matrix for Fbar_n:
#     # Create masks for broadcasted logic: (v_pts[i] - 3 <= j) and (i <= k)
#     v_idx = np.arange(Vmax - 2)[:, None]  # shape (Vmax-2, 1)
#     mask_v = (v_pts[None, :] - 3) <= v_idx  # shape (Vmax-2, n)
#
#     i_idx = np.arange(n)[:, None]  # shape (n, 1)
#     k_idx = np.arange(n)[None, :]  # shape (1, n)
#     mask_i = i_idx <= k_idx  # shape (n, n)
#
#     # Combine with einsum to compute counts of valid (i) for each (j,k)
#     # Fbar[j, k] = sum over i: mask_v[j, i] * mask_i[i, k]
#     Fbar = np.einsum('ji,ik->jk', mask_v, mask_i) / n  # Entry j,k: Fbar_n(s_k, v_j)
#
#     # Compute expression to minimise:
#     distances = np.zeros(n, dtype=float)  # Entry i: Sum{v_j}(Sum{s_k}(|^F_n(s_k, v_j; t = s_i) - Fbar(s_k, v_j)|))
#     for i in range(n):
#         distances[i] = np.sum(np.abs(F_estimates[i] - Fbar))
#
#     # Assign truncation param ^t_n minimising distances:
#     trunc_idx = np.argmin(distances)  # Truncation param ^t_n = x_pts[trunc_idx] = s_(trunc_idx)
#     # Truncated ^H_n^b = Hb_truncated[trunc_idx, 1:]
#
#     # Compute truncated unbiased estimate ^H_n:
#     # ^H_n^b(s_i; ^t_n) - ^H_n^b(s_i-1; ^t_n) = Hb_probabilities[trunc_idx, i]
#     H_est = np.cumsum(Hb_probabilities[trunc_idx] / x_pts)  # Numerator of truncated ^H_n
#     H_est /= H_est[-1]  # Truncated ^H_n
#     return H_est


def de_bias_AV(x_pts, v_pts, estimate, reference_sample):
    """
    A separate function which performs the de-biasing step which may also be performed by the function
    pysizeunfolder.estimate_size_AV after the optimization procedure. This function is an altered version of "de_bias",
    incorporating number of vertices data.

    :param x_pts: A numpy.ndarray of shape (n,) representing the points at which the CDF is evaluated. Such as x_pts
        returned by the function pysizeunfolder.estimate_size_AV.
    :param v_pts: A numpy.ndarray of shape (n,) representing the number of vertices corresponding to a point in "x_pts".
        Such as v_pts returned by the function pysizeunfolder.estimate_size_AV.
    :param estimate: A numpy.ndarray of shape (n,) representing the values the biased CDF takes in x_pts. Such as y_pts
        returned by the function pysizeunfolder.estimate_size_AV.
    :param reference_sample: A sample of pairs of section profile areas and numbers of vertices of the reference shape,
        in a list of two numpy.ndarrays of shape (N,), the first containing areas and the second containing numbers of
        vertices, where N is the sample size. Ideally N is very large.
    :return: y_pts: a numpy.ndarray of shape (n,). This array represents a piece-wise constant distribution function,
        which is the unbiased version of the input function "estimate". The CDF is constant on [x_pts[i],x_pts[i+1]).
        Here: [a,b) = {x: a <= x < b}. y_pts[i] is the CDF value in x_pts[i].
    """
    n = len(x_pts)
    Vmax = reference_sample[1].max()
    s_pts = np.linspace(x_pts.min(), x_pts.max(), n)  # Evenly spaced s-values, used for s_k; x_pts used for s_l & s_i.

    # Compute matrix for ~G_K:
    x_mat = np.broadcast_to(x_pts, (n, n))
    s_mat = np.broadcast_to(s_pts, (n, n))
    s_frac_mat = s_mat.T / x_mat
    G_estimates = np.zeros((Vmax - 2, n, n), dtype=float)  # Entry j,k,l: ~G_K(s_k / s_l, v_j)
    for v in range(Vmax - 2):
        Pv = sum(reference_sample[1] - 3 <= v) / len(reference_sample[1])  # Probability that V <= v + 3.
        G_per_v_cdf = ECDF(np.sqrt(reference_sample[0][reference_sample[1] - 3 <= v]))  # Emperical approx ~G_K(., v)
        # Compute all entries with v vertices.
        G_estimates[v] = np.reshape(G_per_v_cdf(s_frac_mat.flatten()), (n, n)) * Pv

    # Compute matrix for truncated ^H_n^b:
    # Hb_truncated = np.zeros((n, n + 1), dtype=float)  # Entry i,l+1: ^H_n^b(s_l; t = s_i)
    Hb_truncated = np.zeros((200, n + 1), dtype=float)  # Entry i,l+1: ^H_n^b(s_l; t = s_i)
    # for i in range(n - 1):
    for i in range(200):
        if estimate[i] < 1:
            # Truncated Hb only nonzero if s_l > s_i <=> l > i.
            # est[i] = ^H_n^b(s_i) => Entry i,l+1: (est[l] - est[i]) / (1 - est[i])
            Hb_truncated[i, i+2:] = (estimate[i+1:] - estimate[i]) / (1 - estimate[i])
    # Compute matrix for diffs in consecutive entries along last axis.
    Hb_probabilities = np.diff(Hb_truncated)  # Entry i,l: ^H_n^b(s_l; s_i) - ^H_n^b(s_l-1; s_i)

    # Compute matrix for ^F_n:
    # Entry i,j,k: ^F_n(s_k, v_j; t = s_i) = sum{l}(G_estimates[j, k, l] * Hb_probabilities[i, l])
    F_estimates = np.einsum('jkl,il->ijk', G_estimates, Hb_probabilities)
    # Compute matrix for Fbar_n:
    Fbar = np.zeros((Vmax-2, n), dtype=int)  # Entry j,k: Fbar_n(s_k, v_j)
    for k in range(n):
        for j in range(Vmax - 2):
            Fbar[j, k] = sum((x_pts <= s_pts[k]) * (v_pts - 3 <= j))
    Fbar = Fbar / n

    # Compute expression to minimise:
    # distances = np.zeros(n, dtype=float)  # Entry i: Sum{v_j}(Sum{s_k}(|^F_n(s_k, v_j; t = s_i) - Fbar(s_k, v_j)|))
    distances = np.zeros(200, dtype=float)  # Entry i: Sum{v_j}(Sum{s_k}(|^F_n(s_k, v_j; t = s_i) - Fbar(s_k, v_j)|))
    # condition = np.zeros(n)  # Condition to prevent low x_pts values from exploding the estimate.
    condition = np.zeros(200)  # Condition to prevent low x_pts values from exploding the estimate.
    cond_met = False  # TO BE REMOVED
    f = open(f"outputlog{Vmax}.txt", "a")  # TO BE REMOVED
    # for i in range(n):
    for i in range(200):
        distances[i] = np.sum(np.abs(F_estimates[i] - Fbar))
        condition[i] = (Hb_probabilities[0, i] / x_pts[i] < 3*Hb_probabilities[0].max())
        # TO BE REMOVED
        if condition[i] == False:
            f.write(f"{i}:\t{Hb_probabilities[0, i]} / {x_pts[i]} > 3*{Hb_probabilities[0].max()}\t")
            cond_met = True
    f.write(f"Done! Condition {(not cond_met) * 'not'} met\n")  # TO BE REMOVED
    f.close()  # TO BE REMOVED

    # Assign truncation param ^t_n at last index (meeting the condition) that minimises distances:
    # trunc_idx = n - 1 - np.argmin(np.flip(distances[np.argmin(condition):]))
    trunc_idx = 200 - 1 - np.argmin(np.flip(distances[np.argmin(condition):]))
    # Truncation param ^t_n = x_pts[trunc_idx] = s_(trunc_idx)

    # Compute truncated unbiased estimate ^H_n:
    # ^H_n^b(s_i; ^t_n) - ^H_n^b(s_i-1; ^t_n) = Hb_probabilities[trunc_idx, i]
    H_est = np.cumsum(Hb_probabilities[trunc_idx] / x_pts)  # Numerator of truncated ^H_n
    H_est /= H_est[-1]  # Truncated ^H_n
    return H_est


def estimate_size(observed_areas, reference_sample, debias=True, algorithm="icm_em",
                  tol=0.0001, stop_iterations=10, em_max_iterations=None, sphere=False):
    """
    A function for estimating the size distribution CDF given a sample of observed section areas.
    The so-called length-biased size distribution CDF $H^b$ is estimated via nonparametric maximum likelihood.
    If 'debias' is set to true an additional step is performed and the estimate of $H^b$ is used to
    estimate the size distribution CDF. For details of the interpretation of these distributions and the estimation
    procedure see the paper: "Stereological determination of particle size distributions".

    NOTE: The given sample of section areas should not contain any duplicates. Within the model setting this
    is a probability zero event. In practice it may be the case that this does occur, especially when the
    measurement device used to obtain the data does not have a very high resolution. If this kind of data
    is presented to this function, a very small amount of noise is added to the data, to ensure unique values.
    This functionality is not extensively tested, you may mannually add a negligible amount of noise to your data for
    more control.

    :param observed_areas: A sample of observed section areas, a numpy.ndarray of shape (n,) where n is the sample size.
    :param reference_sample: A sample of section areas of the reference shape, a numpy.ndarray of shape (N,) where N is
        the sample size. Ideally N is very large.
    :param debias: A boolean, if True the size distribution is estimated, sometimes called number weighted size
        distribution. If False, the length-biased size distribution is estimated, may be interpreted as diameter
        weighted size distribution.
    :param algorithm: A string, one of "icm_em", "icm" or "em". Unless testing algorithm performance, it is best to
        leave this at the default setting.
    :param tol: A double, the tolerance used for stopping the optimization procedure (maximum likelihood). If the
        largest change in probability mass between estimates of successive iterations is below 'tol' for
        'stop_iterations' successive iterations the algorithm is terminated.
    :param stop_iterations: An integer, indicating for how many iterations the optimization procedure should run such
        that the largest change in probability mass of the estimate compared to the previous estimate is below 'tol'.
    :param em_max_iterations: An integer, only used if algorithm="em". The amount of iterations to be run by EM.
    :param sphere: A boolean, indicating that we assume a spherical shape for the particles. Reference_sample will not
        be used if set to true.
    :return: Two numpy arrays: x_pts, y_pts. Both are a numpy.ndarray of shape (n,). These arrays represent a piece-wise
        constant distribution function. The CDF is constant on [x_pts[i],x_pts[i+1]). Here: [a,b) = {x: a <= x < b}.
        y_pts[i] is the CDF value in x_pts[i].
    """
    sqrt_sample = np.sqrt(observed_areas)
    n = len(sqrt_sample)
    rng = np.random.default_rng(0)
    sigma = np.std(sqrt_sample)

    if len(np.unique(sqrt_sample)) < n:
        print("Warning: input contains duplicate values, adding a small amount of noise.")
        sqrt_sample = np.abs(sqrt_sample + rng.normal(loc=0, scale=0.000001*sigma, size=n))
    sqrt_sample = np.sort(sqrt_sample)

    # compute matrix alpha_{i,j}
    mat = np.broadcast_to(sqrt_sample, (n, n))
    input_mat = mat.T / mat
    if sphere:
        data_matrix = np.reshape(gs_sphere(input_mat.flatten()), (n, n)) / mat
    else:
        # compute kernel density estimate of g_K^S
        kde_x, kde_y = approx_area_density(np.sqrt(reference_sample), sqrt_data=True)
        cs = CubicSpline(kde_x, kde_y)

        def gs_density(x):
            y = cs(x)
            y[y < 0] = 0.0
            y[x > np.max(kde_x * 1.05)] = 0.0
            y[x < 0] = 0.0
            return y
        data_matrix = np.reshape(gs_density(input_mat.flatten()), (n, n)) / mat

    # Compute MLE
    if algorithm == "icm_em":
        est = hybrid_icm_em(data_matrix, tol=tol, stop_iterations=stop_iterations)
    elif algorithm == "icm":
        est = icm(data_matrix, tol=tol, stop_iterations=stop_iterations)
    elif algorithm == "em":
        if em_max_iterations is None:
            em_max_iterations = 3 * n
        est = em(data_matrix, iterations=em_max_iterations)

    # Perform debiasing step if required
    if debias:
        if sphere:
            est = de_bias_sphere(sqrt_sample, est, sphere_G)
        else:
            est = de_bias(sqrt_sample, est, reference_sample)

    return sqrt_sample, est


def estimate_size_AV(observed_data, reference_sample, debias=True, algorithm="icm_em",
                     tol=0.0001, stop_iterations=10, em_max_iterations=None):
    """
    A function for estimating the size distribution CDF given a sample of paired observed section areas and numbers of
    vertices. The so-called length-biased size distribution CDF $H^b$ is estimated via nonparametric maximum likelihood.
    If 'debias' is set to true an additional step is performed and the estimate of $H^b$ is used to estimate the size
    distribution CDF. This function is an altered version of "estimate_size", incorporating number of vertices data.

    NOTE: The given sample of section areas should not contain any duplicates. Within the model setting this
    is a probability zero event. In practice it may be the case that this does occur, especially when the
    measurement device used to obtain the data does not have a very high resolution. If this kind of data
    is presented to this function, a very small amount of noise is added to the data, to ensure unique values.
    This functionality is not extensively tested, you may mannually add a negligible amount of noise to your data for
    more control.

    :param observed_data: A list of two numpy.ndarrays of shape (n,), containing a sample of observed pairs of section
            area (first array, floats) and number of vertices (second array, integers), at the same index in both
            arrays. Here, n is the sample size.
    :param reference_sample: A list of two numpy.ndarrays of shape (N,), containing a sample of pairs of section area
            (first array, floats) and number of vertices (second arrray, integers) from the reference shape, at the same
            index in both arrays. Here, N is the sample size and is ideally very large.
    :param debias: A boolean, if True the size distribution is estimated, sometimes called number weighted size
            distribution. If False, the length-biased size distribution is estimated, may be interpreted as diameter
            weighted size distribution.
    :param algorithm: A string, one of "icm_em", "icm" or "em". Unless testing algorithm performance, it is best to
            leave this at the default setting.
    :param tol: A double, the tolerance used for stopping the optimization procedure (maximum likelihood). If the
            largest change in probability mass between estimates of successive iterations is below 'tol' for
            'stop_iterations' successive iterations the algorithm is terminated.
    :param stop_iterations: An integer, indicating for how many iterations the optimization procedure should run such
            that the largest change in probability mass of estimate compared to the previous estimate is below 'tol'.
    :param em_max_iterations: An integer, only used if algorithm="em". The amount of iterations to be run by EM.
    :return: Three numpy arrays: x_pts, v_pts, y_pts. All are a numpy.ndarray of shape (n,). ("x_pts", "y_pts")
            represent a piece-wise constant distribution function. The CDF is constant on [x_pts[i], x_pts[i+1]).
            Here: [a,b) = {x: a <= x < b}. y_pts[i] is the CDF value in x_pts[i]. "v_pts" is the number of vertices
            corresponding to the original data point of "x_pts".
    """
    n = len(observed_data[0])
    Smax = np.sqrt(reference_sample[0].max())
    Vmax = int(reference_sample[1].max())

    # Set up sample with sqrt areas:
    sqrt_sample = np.array([np.sqrt(observed_data[0]), observed_data[1]])
    if len(np.unique(sqrt_sample[0])) < n:
        print("Warning: input contains duplicate values, adding a small amount of noise.")
        rng = np.random.default_rng(0)
        sigma = np.std(sqrt_sample[0])
        sqrt_sample[0] = np.abs(sqrt_sample[0] + rng.normal(loc=0, scale=0.000001*sigma, size=n))

    # WLOG, sort data pairs by sqrt areas:
    idx_sort = sqrt_sample[0].argsort()
    idx_sort = np.stack([idx_sort, idx_sort])
    sqrt_sample = np.take_along_axis(sqrt_sample, idx_sort, axis=1)

    # Collect indices for data with different nums of verts
    idx_num_verts = [[] for _ in range(Vmax - 2)]  # Indices of data with V = v (at index v-3 in list idx_num_verts)
    for i in range(n):
        idx_num_verts[int(sqrt_sample[1, i]) - 3].append(i)

    # Approximate ~g_K per num of verts
    cs = [None for _ in range(Vmax - 2)]  # List of all estimated ~g_K functions per number of verts.

    for v in range(Vmax - 2):
        Pv = sum(reference_sample[1] - 3 == v) / len(reference_sample[1])  # Probability that V = v + 3.
        if Pv > 0:
            # Compute kernel density estimate of ~g_K(s | V = v)
            xv_kde, yv_kde = approx_area_density(np.sqrt(reference_sample[0][reference_sample[1] - 3 == v]),
                                                 sqrt_data=True)
            yv_kde *= Pv  # Scale with respect to prob(V = v) to obtain true ~g_K with v verts.
        else:
            # Density estimate should be equal to 0 everywhere
            xv_kde = np.linspace(1e-10, Smax, 16384)
            yv_kde = np.zeros(16384)
        cs[v] = CubicSpline(xv_kde, yv_kde)  # Polynomial approximation of ~g_K with v verts.

    def g_per_v_density(x, v: int):
        y = cs[v](x)
        y[y < 0] = 0.0
        y[x < 0] = 0.0
        y[x > Smax * 1.05] = 0.0
        return y

    # compute matrix alpha_{i,j}
    s_mat = np.broadcast_to(sqrt_sample[0], (n, n))
    s_frac_mat = s_mat.T / s_mat
    data_matrix = np.zeros((n, n), dtype=float)
    for v in range(Vmax - 2):
        # Compute alphas in rows with v verts.
        data_matrix[[idx_num_verts[v]]] = np.reshape(g_per_v_density(s_frac_mat[idx_num_verts[v]].flatten(), v),
                                                     (len(idx_num_verts[v]), n))
    data_matrix /= s_mat  # Final step (divide by s_(j)) to compute the alphas.

    # Compute MLE
    if algorithm == "icm_em":
        est = hybrid_icm_em(data_matrix, tol=tol, stop_iterations=stop_iterations)
    elif algorithm == "icm":
        est = icm(data_matrix, tol=tol, stop_iterations=stop_iterations)
    elif algorithm == "em":
        if em_max_iterations is None:
            em_max_iterations = 3 * n
        est = em(data_matrix, iterations=em_max_iterations)

    # Perform debiasing step if required
    if debias:
        est = de_bias_AV(sqrt_sample[0], sqrt_sample[1], est, reference_sample)

    return sqrt_sample[0], sqrt_sample[1], est


def error_estimate(x_true, y_true, x_est, y_est):
    """
    Method to calculate the errors, defined as the supremum distances, of given estimate step functions with respect to
    the true function. Distances between functions are evaluated at the x-coordinates in the given "x_true" array.
    :param x_true: A numpy.ndarray of shape (n,), containing the x-coordinates for the true function values in "y_true".
    :param y_true: A numpy.ndarray of shape (n,), containing the true function values at the x-coordinates in "x_true".
    :param x_est: A numpy.ndarray of shape (k, m), where row i out of k contains the x-coordinates of jump locations for
            the i-th estimate function with jump values in "y_est[i]".
    :param y_est: A numpy.ndarray of shape (k, m), where row i out of k contains the jump values for the i-th estimate
            function with jump locations at the x-coordinates in "x_est[i]".
    :return: A numpy.ndarray of shape (k,), where entry i contains the supremum distance between the estimate step
            function "(x_est[i], y_est[i])" and the true function "(x_true, y_true)".
    """
    error_sup = np.zeros(len(x_est), dtype=float)
    indices = np.zeros(len(x_est), dtype=int)

    for i in range(len(x_true)):
        # For each estimate:
        for j in range(len(x_est)):
            # Skip x-coords of the j-th estimate until point of evaluation is reached.
            while indices[j] < x_est.shape[-1] - 1 and x_est[j, indices[j]] <= x_true[i]:
                indices[j] += 1
            # Point of evaluation for j-th estimate: (x_true[i], y_true[i]) and
            #                                        (x_est[j, indices[j]], y_est[j, indices[j]]).
            error = abs(y_est[j, indices[j]] - y_true[i])
            # Update value of supremum error if necessary:
            if error > error_sup[j]:
                error_sup[j] = error

    return error_sup


def error_pairwise_estimate(x_true, y_true, x_est, y_est_a, y_est_b):
    """
    Method to evaluate the estimation performances of two methods: "a" and "b". Calculates the pairwise errors, defined
    as the extreme values of the pointwise difference in distances from each estimate step function to the true
    function: inf/sup( |y_est_a - y_true| - |y_est_b - y_true| ). Distances between functions are evaluated at the
    x-coordinates of the given "x_true" array.
    :param x_true: A numpy.ndarray of shape (m,), containing the x-coordinates for the true function values in "y_true".
    :param y_true: A numpy.ndarray of shape (m,), containing the true function values at the x-coordinates in "x_true".
    :param x_est: A numpy.ndarray of shape (k, n), where row i out of k contains the x-coordinates of jump locations for
            the i-th pair of estimate functions with jump values in "y_est_a[i]" and "y_est_b[i]".
    :param y_est_a: A numpy.ndarray of shape (k, n), where row i out of k contains the jump values for the i-th estimate
            function from method a, with jump locations at the x-coordinates in "x_est[i]", the same as "y_est_b".
    :param y_est_b: A numpy.ndarray of shape (k, n), where row i out of k contains the jump values for the i-th estimate
            function from method b, with jump locations at the x-coordinates in "x_est[i]", the same as "y_est_a".
    :return: A numpy.ndarray of shape (2, k), where column i contains the infimum (entry [0, i]) and supremum
            (entry [1, i]) of the pointwise difference in distances from each estimate step function,
            "(x_est[i], y_est_a[i])" and "(x_est[i], y_est_b[i])", and the true function "(x_true, y_true)". If the
            infimum is negative, method a improves over method b in some point. If the supremum is negative, method a
            improves over method b in all points. If the supremum is positive, method b improves over method a in some
            point. If the infimum is positive, method b improves over method a in all points.
    """
    k, n = x_est.shape
    error_inf = np.ones(k, dtype=float)
    error_sup = -np.ones(k, dtype=float)
    indices = np.zeros(k, dtype=int)

    for i in range(n):
        # For each point x_est_i of evaluation
        for j in range(k):
            # For each esimate (x_est_j, y_est_a/b_j)
            while x_true[indices[j]] <= x_est[j, i]:
                indices[j] += 1
        # Points(!) of evaluation for j-th estimate:
        #  1) (x_true[indices[j]], y_true[indices[j]]) vs (x_est[j, i], y_est_a/b[j, i])
        #  2) (x_true[indices[j]-1], y_true[indices[j]-1]) vs (x_est[j, i-1], y_est_a/b[j, i-1])
        if i < n-1:
            # Evaluate error_a - error_b at (1)
            error_diffs_1 = np.abs(y_true[indices] - y_est_a[:, i]) - np.abs(y_true[indices] - y_est_b[:, i])
        else:
            error_diffs_1 = None
        if i > 0:
            # Evaluate error_a - error_b at (2)
            error_diffs_2 = np.abs(y_true[indices-1] - y_est_a[:, i-1]) - np.abs(y_true[indices-1] - y_est_b[:, i-1])
        else:
            error_diffs_2 = None

        for j in range(k):
            # Update inf/sup values if necessary:
            if error_diffs_1 is not None:
                if error_diffs_1[j] < error_inf[j]:
                    error_inf[j] = error_diffs_1[j]
                if error_diffs_1[j] > error_sup[j]:
                    error_sup[j] = error_diffs_1[j]
            if error_diffs_2 is not None:
                if error_diffs_2[j] < error_inf[j]:
                    error_inf[j] = error_diffs_2[j]
                if error_diffs_2[j] > error_sup[j]:
                    error_sup[j] = error_diffs_2[j]

    return np.stack([error_inf, error_sup])
