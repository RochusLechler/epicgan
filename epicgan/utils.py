"""General utilities.
"""
import logging
import sys
import numpy as np

import pickle
from scipy.stats import gaussian_kde



logger = logging.getLogger("main")


def calc_multiplicities(p_dataset):
    """calculates the number of particles with p_t > 0 within a jet and calculates
    frequencies of these numbers throughout the dataset.
    Assumed shape of dataset: [total_size, n_points, n_features]


    Arguments:
    -------------
    p_dataset: np.ndarray
        np.ndarray of the dataset in shape [total_size, n_points, n_features]

    """

    try:
        #look for particles w/ p_t = 0
        nonzero_counts = np.count_nonzero(p_dataset[:,:,0], axis = 1)

    except IndexError as e:
        logger.exception(e)
        logger.warning("""Cannot calculate multiplicities of dataset %s, expected shape
                         [total_size, n_points, n_features]""" % p_dataset)
        sys.exit(-1)

    unique_vals, frequencies = np.unique(nonzero_counts, return_counts = True)

    return unique_vals, frequencies



def calc_kde(p_dataset, file_path = None):
    """Calculates the kernel density estimation (kde) of the number of particles
    with nonzero p_t for a jet using Gaussian kernels.
    Has an option to save the kde to a .pkl-file

    Arguments:
    ----------------
    p_dataset: np.ndarray
        np.ndarray of the dataset in shape [total_size, n_points, n_features]

    file_path: str, default = None
        When specified, the function will try to dump the calculated KDE here;
        should end in .pkl
    """
    unique_vals, frequencies = calc_multiplicities(p_dataset)
    kde = gaussian_kde(unique_vals, weights = frequencies)

    if file_path is not None:
        try:
            with open(file_path, "wb") as f:
                pickle.dump(kde, f)
        except TypeError as e:
            logger.exception(e)
            logger.warning("""When specifying file_path, please give a
                           string ending in .pkl""")
            logger.warning("The KDE-object was not saved")
    return kde
