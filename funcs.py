import numpy as np
# from scipy.spatial.distance import cdist
from numba import njit, prange


@njit(cache=True)
def simulate(boids, D, perception, asp, coeffs):
    calc_dist(boids[:, :2], D)
    M = D < perception
    np.fill_diagonal(M, False)
    wa = wall_avoidance(boids, asp)
    for i in prange(boids.shape[0]):
        idx = np.where(M[i])[0]
        accels = np.zeros((5, 2))
        if idx.size > 0:
            accels[0] = alignment(boids, i, idx)
            accels[1] = cohesion(boids, i, idx)
            accels[2] = separation(boids, i, idx, D)
        accels[3] = wa[i]
        # clip_mag(accels, *arange)
        boids[i, 4:6] = np.sum(accels * coeffs.reshape(-1, 1), axis=0)


@njit(cache=True)
def mean_axis0(arr):
    """
    Return mean for each column of 2D array
    :param arr: input array, at least 2D
    :return:
    """
    n = arr.shape[1]
    res = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        res[i] = arr[:, i].mean()
    return res


@njit
def clip_mag(arr, low, high):
    mag = np.sum(arr * arr, axis=1) ** 0.5
    mask = mag > 1e-16
    mag_cl = np.clip(mag[mask], low, high)
    arr[mask] *= (mag_cl / mag[mask]).reshape(-1, 1)


@njit(cache=True)
def init_boids(boids, asp, vrange=(0., 1.), seed=0):
    N = boids.shape[0]
    np.random.seed(seed)
    boids[:, 0] = np.random.rand(N) * asp
    boids[:, 1] = np.random.rand(N)
    v = np.random.rand(N) * (vrange[1] - vrange[0]) + vrange[0]
    alpha = np.random.rand(N) * 2 * np.pi
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s


@njit(cache=True)
def directions(boids):
    return np.hstack((boids[:, :2] - boids[:, 2:4], boids[:, :2]))


@njit(cache=True)
def propagate(boids, dt, vrange):
    boids[:, :2] += dt * boids[:, 2:4] + 0.5 * dt ** 2 * boids[:, 4:6]
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_mag(boids[:, 2:4], vrange[0], vrange[1])


@njit(cache=True)
def calc_dist(arr, D):
    n = arr.shape[0]
    for i in prange(n):
        for j in range(i):
            v = arr[j] - arr[i]
            d = (v @ v) ** 0.5
            D[i, j] = d
            D[j, i] = d


# @njit(cache=True)
# def calc_neighbors(boids, D, perception):
#     N = boids.shape[0]
#
#     # mask[range(N), range(N)] = False
#     return mask, D

@njit(cache=True)
def periodic_walls(boids, asp):
    boids[:, :2] %= np.array([asp, 1.])


@njit(cache=True)
def alignment(boids, i, idx):
    avg = mean_axis0(boids[idx, 2:4])
    a = avg - boids[i, 2:4]
    return a


@njit(cache=True)
def cohesion(boids, i, idx):
    center = mean_axis0(boids[idx, 0:2])
    a = center - boids[i, 0:2]
    return a


@njit(cache=True)
def separation(boids, i, idx, D):
    d = boids[i, 0:2] - boids[idx, 0:2]
    a = np.sum(d / D[i][idx].reshape(-1, 1), axis=0)
    return a


@njit
def wall_avoidance(boids, asp):
    left = np.abs(boids[:, 0])
    right = np.abs(asp - boids[:, 0])
    bottom = np.abs(boids[:, 1])
    top = np.abs(1 - boids[:, 1])

    ax = 1 / left ** 2 - 1 / right ** 2
    ay = 1 / bottom ** 2 - 1 / top ** 2

    return np.column_stack((ax, ay))