import numpy as np
# from scipy.spatial.distance import cdist
from numba import njit, prange
from math import floor, ceil
from numba.typed import Dict


@njit(cache=True)
def insert_boids(boids, cell_to_boids, width, asp):
    """
        inserts every boid in dictionary cell -> boid, where cell is hashed by boids coordinates
        ---
        input:
        boids - array of boids
        cell_to_boids - dictionary (empty in the beginning of a step)
        width - heigth of image in cells
        asp - aspect of image
        ---
        returns:
        cell_to_boids dictionary
    """
    N = boids.shape[0]
    for i in range(N):
        cell = int(boids[i][0]*width*asp) + int(boids[i][1]*width) * int(width * asp)
        if cell in cell_to_boids:
            n = cell_to_boids[cell][0]
            cell_to_boids[cell][n+1] = i
            cell_to_boids[cell][0] += 1
        else:
            cell_to_boids[cell] = np.zeros(shape=N+1, dtype=np.int64)
            cell_to_boids[cell][1] = i
            cell_to_boids[cell][0] = 1


@njit(cache=True)
def simulate_hashed(boids, perception, cell_to_boids, asp, coeffs, width):
    """
        general simulation function
        ---
        input:
        boids - array of boids
        cell_to_boids - dictionary (empty in the beginning of a step)
        perception - radius of vision for a boid
        coeffs - coefficients for affects of all causes
        width - heigth of image in cells
        asp - aspect of image
        ---
        returns:
        parameters at the end of step for each boid
    """
    insert_boids(boids, cell_to_boids, width, asp)
    wall_avoidance_hash(boids, asp, coeffs[4])
    for cell in list(cell_to_boids.keys()):
        if cell_to_boids[cell][0] > 1:
            cell_elements, num = cell_to_boids[cell], cell_to_boids[cell][0]
            calc_dist_and_accels_hash(boids, cell_elements[1: num+1], perception, coeffs)


@njit(cache=True)
def calc_dist_and_accels_hash(boids, idx, perception, coeffs): # idx: [e1, e2, e3,..., eN]
    """
        calculatess all distances in a cell and evaluates accels for each boid in it
        ---
        input:
        boids - array of boids
        idx - array of all indexes in the current cell
        perception - radius of vision for a boid
        coeffs - coefficients for affects of all causes
        ---
        distances matrix and mask for it (with perception)
    """
    n = idx.shape[0]
    dist = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i):
            v = boids[idx[i], 0:2] - boids[idx[j], 0:2]
            d = (v @ v) ** 0.5
            dist[i, j] = d
    dist += dist.T
    mask = dist < perception
    np.fill_diagonal(mask, False)
    calculate_accels(boids, mask, dist, idx, coeffs)


@njit(cache=True)
def calculate_accels(boids, mask, dist, idx, coeffs):
    """
        calculates accels for each boid in a cell
        ---
        input:
        boids - array of boids
        mask - bool matrix for dist <= perception (is the boid visible to ssome other boid i?)
        dist - matrix of distances between boids in a cell
        idx - array of all indexes in the current cell
        coeffs - coefficients for affects of all causes
        ---
        returns: boids with accels for the step
    """
    n = idx.shape[0]
    low, high = 0.0, 1.0
    for i in range(n):
        idx_percepted_mask = np.where(mask[i])[0]          # idx_perc_mask: [k | k < n && mask[k] == True]
        idx_percepted = idx[idx_percepted_mask]            # idx_perc: [j | j {= idx && dist(j, idx[i]) <= perception
        accels = np.zeros((4, 2))
        if idx_percepted.shape[0] > 0:
            accels[0] = alignment_hash(boids, idx[i], idx_percepted)
            accels[1] = cohesion_hash(boids, idx[i], idx_percepted)
            accels[2] = separation_hash(boids, i, idx, idx_percepted, idx_percepted_mask, dist)
            accels[3] = noise(boids, idx)
            clip_mag(accels, low, high)
        boids[idx[i], 4:6] = np.sum(accels * coeffs[:4].reshape(-1, 1), axis=0)


@njit(cache=True)
def alignment_hash(boids, i, idx_percepted):
    """
        calculates alignment parameter for interacrion between current boid and boids in his view range
        ---
        input:
        boids - array of boids
        i - index of current boid to check
        idx_percepted - array of indexes for boids that it can see
        ---
        returns: alignment component of accles
    """
    avg = mean_axis0(boids[idx_percepted, 2:4])
    a = avg - boids[i, 2:4]
    return a


@njit(cache=True)
def cohesion_hash(boids, i, idx_percepted):
    """
        calculates cohesion parameter for interacrion between current boid and boids in his view range
        ---
        input:
        boids - array of boids
        i - index of current boid to check
        idx_percepted - array of indexes for boids that it can see
        ---
        returns: cohesion component of accles
    """
    center = mean_axis0(boids[idx_percepted, 0:2])
    a = center - boids[i, 0:2]
    return a


@njit(cache=True)
def separation_hash(boids, i, idx, idx_percepted, idx_percepted_mask, D):
    """
        calculates separation parameter for interacrion between current boid and boids in his view range
        ---
        input:
        boids - array of boids
        i - index of current boid to check
        idx_percepted - array of indexes for boids that it can see
        ---
        returns: separation component of accles
    """
    d = boids[idx[i], 0:2] - boids[idx_percepted, 0:2]
    a = np.sum(d/D[i][idx_percepted_mask].reshape(-1, 1), axis=0)
    return a


@njit(cache=True)
def noise(boids, idx):
    """
        calculates noise parameter for interacrion between current boid and boids in his view range
        ---
        input:
        boids - array of boids
        idx - indexes of all boids in a cell
        ---
        returns: noise component of accles
    """
    a = -idx.shape[0]*mean_axis0(boids[idx, 4:6])
    return a


@njit(cache=True)
def wall_avoidance_hash(boids, asp, coef):
    """
        calculates accels out of interaction with walls for boid
        ---
        input:
        boids - array of boids
        coeffs - coefficients for affects of all causes
        asp - aspect of image
        ---
        returns: boids with wall avoidences assels
    """
    lft = np.abs(boids[:, 0])
    rht = np.abs(asp - boids[:, 0])
    btm = np.abs(boids[:, 1])
    tp = np.abs(1 - boids[:, 1])

    ax = 1 / lft**4 - 1 / rht**4
    ay = 1 / btm**4 - 1 / tp**4
    boids[:, 4] = ax*coef
    boids[:, 5] = ay*coef


@njit(cache=True)
def mean_axis0(arr):
    """
    Return mean for each column of 2D array
    :param arr: input array, at least 2D
    :return: mean for axis 0 of 2d array
    """
    n = arr.shape[1]
    res = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        res[i] = arr[:, i].mean()
    return res


@njit(cache=True)
def clip_mag(arr, low, high):
    """
        clips array of values to some limitations
        ---
        input:
        arr - some 1d array of values
        low - lower bound
        high - upper bound
        ---
        returns: clipped array
    """
    mag = np.sum(arr * arr, axis=1) ** 0.5
    mask = mag > 1e-16
    mag_cl = np.clip(mag[mask], low, high)
    arr[mask] *= (mag_cl / mag[mask]).reshape(-1, 1)


@njit(cache=True)
def init_boids(boids, asp, vrange=(0., 1.), seed=0):
    """
        initialises all boids in the beginning of program
        ---
        input:
        boids - array of boids (zeros)
        asp - aspect of image
        varange - range for speed of boids
        seed - rng seed
        ---
        returns: boids array for step 0
    """
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
    """
    Evaluates velocity and coordinates for each boids by solving ODE numerically (Eulers)

    dt - time step
    vrange - range for velocity

    returns: boids with their coordinates and velocities at the end of a step
    """
    boids[:, :2] += dt * boids[:, 2:4] + 0.5 * dt ** 2 * boids[:, 4:6]
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_mag(boids[:, 2:4], vrange[0], vrange[1])


@njit(cache=True)
def periodic_walls(boids, asp):
    """
    initialises walls
    """
    boids[:, :2] %= np.array([asp, 1.])

