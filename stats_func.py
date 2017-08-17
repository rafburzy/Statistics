import numpy as np

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

def pearson_r(data_1, data_2):
    '''Calculates Pearson correlection coefficient'''
    return np.corrcoef(data_1, data_2)[0,1]

# bootstrap functions

def bootstrap_replicate_1d(data, func):
    '''Generate bootstrap replicate of 1D data'''
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for replicates."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates
    bs_replicates = np.empty(size)


    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates
