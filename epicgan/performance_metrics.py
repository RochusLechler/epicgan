"""Defines all the metrics that are employed to evaluate the performance.
"""

import sys
import logging
import numpy as np
from scipy.stats import wasserstein_distance
import jetnet

from epicgan import utils

logger = logging.getLogger("main")

def wasserstein_mass(real_jets, fake_jets, num_samples = 10000, runs = 5,
                return_std = True, rng = None):
    """Returns the mean Wasserstein distance and optionally the standard deviation
    between the masses of two sets of jets. Mean is computed over distances
    between real_jets and runs sets of samples from fake_jets of size
    num_samples.

    Arguments
    -------------

    real_jets: np.array
        set of events

    fake_jets: np.array
        set of events; make sure it has length greater or equal to
        num_samples*runs

    num_samples: int, default: 10000
        number of samples from fake_jets used for computation of Wasserstein
        distance each iteration

    runs: int, default: 5
        number of iterations/batches from fake_jets for computation of Wasserstein
        distance

    return_std: bool, default: True
        if True, function additionally returns the standard deviation

    rng: np.random.Generator, default: None
        random number generator used for shuffling; if equal to None, no shuffling
        is performed


    Returns
    ------------

    wasserstein_dists.mean(): float
        mean Wasserstein distance between the sets of jets

    wasserstein_dists.std(): float, optional
        standard deviation of the distribution of Wasserstein distances
    """

    if rng is not None:
        permutation1 = rng.permutation(len(real_jets))
        permutation2 = rng.permutation(len(fake_jets))
        real_jets = real_jets[permutation1]
        fake_jets = fake_jets[permutation2]

    #get masses of the jets
    masses_real = utils.jet_masses(real_jets)
    masses_fake = utils.jet_masses(fake_jets)

    #ensure that masses are stored array-like also for size of jets 1
    if masses_real.ndim == 0:
        masses_real = np.array([masses_real])
    if masses_fake.ndim == 0:
        masses_fake = np.array([masses_fake])

    wasserstein_dist_list = []

    k = 0
    for _ in range(runs):
        used_fake_masses = masses_fake[k:int(k+num_samples)]
        k += num_samples

        wasserstein_dist_list.append(wasserstein_distance(masses_real, used_fake_masses))

    wasserstein_dists = np.array(wasserstein_dist_list)

    if return_std:
        return wasserstein_dists.mean(), wasserstein_dists.std()

    return wasserstein_dists.mean()



def wasserstein_coords(real_jets, fake_jets,
                exclude_zeros = True, num_samples = 10000, runs = 5,
                avg_over_features = True, return_std = True, rng = None):
    """Returns either the mean Wasserstein distances and optionally the standard
    deviations between the particle features of two sets of jets separately, or
    the mean of means and norm of standard deviations over the particle features.
    Means are computed over distances between real_jets and runs sets of
    samples from fake_jets each of size num_samples.

    Arguments
    -------------

    real_jets: np.array
        set of events

    fake_jets: np.array
        set of events; make sure it has length greater or equal to
        num_samples*runs

    exclude_zeros: bool, default: True
        if True, the function ignores zero-padded particles

    num_samples: int, default: 10000
        number of samples from fake_jets used for computation of Wasserstein
        distance each iteration

    runs: int, default: 5
        number of iterations/batches from fake_jets for computation of Wasserstein
        distance

    avg_over_features: bool, default: True
        if True, the mean of all Wasserstein distance means and the norm of all
        standard deviations are returned

    return_std: bool, default: True
        if True, function additionally returns the standard deviation

    rng: np.random.Generator, default: None
        random number generator used for shuffling; if equal to None, no shuffling
        is performed


    Returns
    ------------

    np.mean(means) or means: float or list
        mean Wasserstein distance(s) between the particle features; which one is
        returned is determined by argument avg_over_features

    np.linalg.norm(stds) or stds: float or list, optional
        standard deviation(s) of the distributions of Wasserstein distances; which
        one is returned is determined by argument avg_over_features
    """


    if rng is not None:
        permutation1 = rng.permutation(len(real_jets))
        permutation2 = rng.permutation(len(fake_jets))
        real_jets = real_jets[permutation1]
        fake_jets = fake_jets[permutation2]

    #exclude zero-padded particles
    if exclude_zeros:
        zeros_real = np.linalg.norm(real_jets, axis = 2) == 0
        #take the NON-zero elements as mask
        mask_real = ~zeros_real

        zeros_fake = np.linalg.norm(fake_jets, axis = 2) == 0
        #take the NON-zero elements as mask
        mask_fake = ~zeros_fake

    wasserstein_dist_list = []

    if exclude_zeros:
        used_real_jets = real_jets[mask_real]

    k = 0
    for j in range(runs):
        used_fake_jets = fake_jets[k:int(k+num_samples)]

        if exclude_zeros:
            mask = mask_fake[k:int(k+num_samples)]
            used_fake_jets = used_fake_jets[mask]

        k += num_samples

        if real_jets.shape[0] == 0:
            logger.warning("real jets are completely zero-padded")
            wasserstein_d = [np.inf, np.inf, np.inf]

        elif fake_jets.shape[0] == 0:
            logger.warning("fake jets in batch %d are completely zero-padded", j)
            wasserstein_d = [np.inf, np.inf, np.inf]

        else:
            wasserstein_d = []
            for l in range(3): #number of features hard-coded!!
                wasserstein_d.append(wasserstein_distance(used_real_jets[:,l], used_fake_jets[:,l]))

        wasserstein_dist_list.append(wasserstein_d)

    wasserstein_dists = np.array(wasserstein_dist_list)

    means = wasserstein_dists.mean(axis = 0)

    if return_std:
        stds = wasserstein_dists.std(axis = 0)
        if avg_over_features:
            return np.mean(means), np.linalg.norm(stds)
        return means, stds

    if avg_over_features:
        return np.mean(means)
    return means




def wasserstein_efps(real_jets, fake_jets,
    efpset_args = [("n==", 4), ("d==", 4), ("p==", 1)],
    num_samples = 10000, runs = 5, avg_over_efps = True,
    return_std = True, rng = None):
    """Returns either the mean Wasserstein distances and optionally the standard
    deviations between the energyflow polynomials of two sets of jets separately, or
    the mean of means and norm of standard deviations over the polynomials.
    Means are computed over distances between real_jets and runs sets of
    samples from fake_jets each of size num_samples.

    Arguments
    -------------

    real_jets: np.array
        set of events

    fake_jets: np.array
        set of events; make sure it has length greater or equal to
        num_samples*runs

    efpset_args: list, default: [("n==", 4), ("d==", 4), ("p==", 1)]
        list of arguments for computation of energyflow polynomials; for further
        details see https://jetnet.readthedocs.io/en/latest/pages/utils.html#jetnet.utils.efps

    num_samples: int, default: 10000
        number of samples from fake_jets used for computation of Wasserstein
        distance each iteration

    runs: int, default: 5
        number of iterations/batches from fake_jets for computation of Wasserstein
        distance

    avg_over_efps: bool, default: True
        if True, the mean of all Wasserstein distance means and the norm of all
        standard deviations are returned

    return_std: bool, default: True
        if True, function additionally returns the standard deviation

    rng: np.random.Generator, default: None
        random number generator used for shuffling; if equal to None, no shuffling
        is performed


    Returns
    ------------

    np.mean(mean) or means: float or list
        mean Wasserstein distance(s) between the energyflow polynomials; which
        one is returned is determined by argument avg_over_efps

    np.linalg.norm(stds) or stds: float or list, optional
        standard deviation(s) of the distributions of Wasserstein distances; which
        one is returned is determined by argument avg_over_efps
    """

    if rng is not None:
        permutation1 = rng.permutation(len(real_jets))
        permutation2 = rng.permutation(len(fake_jets))
        real_jets = real_jets[permutation1]
        fake_jets = fake_jets[permutation2]

    real_jets = real_jets[:,:,[1,2,0]]
    fake_jets = fake_jets[:,:,[1,2,0]]

    efps_real = jetnet.utils.efps(real_jets, efpset_args = efpset_args)
    efps_fake = jetnet.utils.efps(fake_jets, efpset_args = efpset_args)

    num_efps = efps_real.shape[1]

    wasserstein_dist_list = []

    k = 0
    for _ in range(runs):
        used_fake_efps = efps_fake[k:int(k+num_samples)]
        k += num_samples

        wasserstein_d = []
        for l in range(num_efps):
            wasserstein_d.append(wasserstein_distance(efps_real[:,l], used_fake_efps[:,l]))

        wasserstein_dist_list.append(wasserstein_d)

    wasserstein_dists = np.array(wasserstein_dist_list)
    means = wasserstein_dists.mean(axis = 0)
    stds = wasserstein_dists.std(axis = 0)

    if return_std:
        stds = wasserstein_dists.std(axis = 0)
        if avg_over_efps:
            return np.mean(means), np.linalg.norm(stds)
        return means, stds

    if avg_over_efps:
        return np.mean(means)
    return means


#dictionary that maps names of datasets to values accepted by fpnd-function
jettype_dict = {
                "gluon30": "g",
                "quark30": "q",
                "top30": "t",
}


def fpnd_score(fake_jets, dataname, num_samples = 10000, runs = 10,
                return_std = True):
    """Returns the mean Frechet ParticleNet distance and optionally the standard
    deviation between two sets of jets. Mean is computed over distances
    between real_jets and runs sets of samples from fake_jets of size
    num_samples.
    Employs jetnet.evaluation.fpnd(), which requires torch-cluster, which requires
    Python version <= 3.10

    Arguments
    -------------

    fake_jets: np.array
        set of events; make sure it has length greater or equal to
        num_samples*runs

    num_samples: int, default: 10000
        number of samples from fake_jets used for computation of FPND at each
        iteration

    runs: int, default: 5
        number of iterations/batches from fake_jets for computation of FPND

    return_std: bool, default: True
        if True, function additionally returns the standard deviation

    rng: np.random.Generator, default: None
        random number generator used for shuffling; if equal to None, no shuffling
        is performed


    Returns
    ------------

    fpnd_array.mean(): float
        mean FPND between the sets of jets

    fpnd_array.std(): float, optional
        standard deviation of the distribution of FPNDs
    """

    try:
        jettype = jettype_dict[dataname]
    except KeyError as e:
        logger.exception(e)
        logger.critical("""Please specify a valid dataname, if you want to
                        compute the FPND as an additional evaluation means;
                        note that only JetNet30 datasets are possible""")
        sys.exit()

    #features need to be in order [eta, phi, p_t] as in original JetNet datasets
    fake_jets = fake_jets[:,:,[1,2,0]]

    fpnd_list = []
    k = 0

    for _ in range(runs):
        used_jets = fake_jets[k:int(k + num_samples)]
        k += num_samples
        fpnd = jetnet.evaluation.fpnd(used_jets, jet_type = jettype, use_tqdm = False)
        fpnd_list.append(fpnd)


    if return_std:
        fpnd_array = np.array(fpnd_list)
        return fpnd_array.mean(), fpnd_array.std()

    return np.array(fpnd_list).mean()
