"""Implementations of functions used for evaluation of the performance of the
network.
"""

import sys
import logging

import numpy as np
from scipy.stats import wasserstein_distance
from epicgan import data_proc, utils


logger = logging.getLogger("main")



def compute_wasserstein_distance(network, data_set, kde, batch_size = 128,
                    n_tot_generation = 300000, dim_global = 10, dim_particle = 3,
                    rng = None, set_min_pt = True, min_pt = 0, center_gen = True,
                    order_by_pt = True,
                    inv_normalise_data = True, inv_means = np.zeros(3), inv_stds = np.ones(3),
                    inv_norm_sigma = 1, runs = 10, device = "cuda"):
    """Computes the Wasserstein distance between masses of the jets of the
    given dataset and generated jets. The return is the mean value of the
    Wasserstein distances for 'runs' number of generated jet datasets.

    Arguments
    --------------

    network: Generator
        generator network with which to perform the evaluation

    data_set: np.array
        dataset for which to perform the evaluation means

    kde: scipy.stats.gaussian_kde
        kernel density estimation of n_eff for the dataset

    batch_size: int, default: 128
        batch size to be created by generator

    n_tot_generation: int, default: 300000
        number of points that will be sampled from kde

    dim_global: int, default: 10
        dimension of the space of global variables in the network

    dim_particle: int, default: 3
        dimension of the space of particle features

    rng: np.random.Generator, default: None
        random number generator used for kde-resampling; please note that
        resampling of kde needs a random number generator, it will default to
        np.random, if None is given

    set_min_pt: bool, default: True
        if True, all generated particles will be enforced to have p_t of at least
        the value specified in min_pt

    min_pt: int or float, default: 0
        minimum value to which all p_t below will be set, if set_min_pt is True

    center_gen: bool, default: True
        if True, the eta- and phi-coordinates of the generated events will be
        centered

    order_by_pt: bool, default: True
        if True, the particles within each generated fake jet will be ordered by
        p_t in descending order

    inv_normalise_data: bool, default: True
        if True, the generated events will be renormalised to have the statistical
        properties of the training set

    inv_means: list or np.array, default: np.zeros(3)
        mean value for each particle feature to which to renormalise the generated events

    inv_stds: list or np.array, default: np.ones(3)
        standard deviation value for each particle feature to which to renormalise
        the generated events

    inv_norm_sigma: float or int, default: 1
        std-value to which the real input data was normalised

    runs: int, default: 10
        number of disjoint subsets of generated events that will be used to compute
        the Wasserstein distances, the returned Wasserstein distances will be the
        means of all the computed distances. Note that this must be specified s.t.
        runs times the length of the dataset is <= n_tot_generation

    device: str, default: "cuda"
        device to which to send variables


    Returns
    ------------

    w_dist_mean: float
        Wasserstein distance for the mass averaged over all runs
    """

    len_data_set, n_points = data_set.shape[0], data_set.shape[1]
    generated_events = generation_loop(network, n_points, kde,
                        batch_size = batch_size, n_tot_generation = n_tot_generation,
                        dim_global = dim_global, dim_particle = dim_particle, rng = rng,
                        order_by_pt = order_by_pt, set_min_pt = set_min_pt,
                        min_pt = min_pt, center_gen = center_gen,
                        inv_normalise_data = inv_normalise_data, inv_means = inv_means,
                        inv_stds = inv_stds, inv_norm_sigma = inv_norm_sigma, device = device)


    #order also real data
    if order_by_pt:
        data_set = utils.order_array_pt(data_set)

    #get masses of the jets
    data_set_masses = utils.jet_masses(data_set)
    generated_masses = utils.jet_masses(generated_events)

    w_distances_list = []
    k = 0
    for _ in range(runs):
        used_masses = generated_masses[k:int(k + len_data_set)]
        k += len_data_set
        w_distance = wasserstein_distance(data_set_masses, used_masses)
        w_distances_list.append(w_distance)

    w_dist_mean = np.array(w_distances_list).mean()

    return w_dist_mean




def generation_loop(network, n_points, kde, batch_size = 128, n_tot_generation = 300000,
                    dim_global = 10, dim_particle = 3, rng = None, order_by_pt = True,
                    set_min_pt = True, min_pt = 0, center_gen = True,
                    inv_normalise_data = True, inv_means = np.zeros(3), inv_stds = np.ones(3),
                    inv_norm_sigma = 1, device = "cuda"):
    """This function generates simulated events mimicking the appearance of the
    JetNet datasets.
    First, a distribution of n_eff, which is the number of particles per jet
    with p_t != 0, is sampled from the precomputed kernel density estimation for
    the respective dataset. Values out of the range [1, n_points] are discarded.
    Note that this means that the number of returned events  can be
    a bit smaller than the specified n_tot_generation.
    Then, the specified generator network generates jets
    of particle multiplicity n_eff as often as the kernel density estimation
    yielded that value n_eff. Afterwards, zero-padding increases the final
    particle multiplicity to n_points (either 30 or 150). This way the generated
    dataset mimicks the structure of the JetNet-datasets.
    With default settings, the validation set is 15% of the total dataset
    (ca. 27,000 samples) and for each evaluation ten runs of comparisons between
    validation set and generated jets set are done. Therefore ca. 270,000 generated
    samples are required, which is why n_tot_generation defaults to 300,000 (a few
    samples could be killed by the requirement 1 <= n_eff <= n_points).


    Arguments
    -------------

    network: Generator
        generator network with which to perform the evaluation

    n_points: int
        number of particles per jet, 30 or 150

    kde: scipy.stats.gaussian_kde
        kernel density estimation of n_eff for the dataset

    batch_size: int, default: 128
        batch size to be created by generator

    n_tot_generation: int, default: 300000
        number of points that will be sampled from kde

    dim_global: int, default: 10
        dimension of the space of global variables in the network

    dim_particle: int, default: 3
        dimension of the space of particle features

    rng: np.random.Generator, default: None
        random number generator used for kde-resampling; please note that
        resampling of kde needs a random number generator, it will default to
        np.random, if None is given

    order_by_pt: bool, default: True
        if True, the particles within each generated fake jet will be ordered by
        p_t in descending order

    set_min_pt: bool, default: True
        if True, all generated particles will be enforced to have p_t of at least
        the value specified in min_pt

    min_pt: int or float, default: 0
        minimum value to which all p_t below will be set, if set_min_pt is True

    center_gen: bool, default: True
        if True, the eta- and phi-coordinates of the generated events will be
        centered

    inv_normalise_data: bool, default: True
        if True, the generated events will be renormalised to have the statistical
        properties of the training set

    inv_means: list or np.array, default: np.zeros(3)
        mean value for each particle feature to which to renormalise the generated events

    inv_stds: list or np.array, default: np.ones(3)
        standard deviation value for each particle feature to which to renormalise
        the generated events

    inv_norm_sigma: float or int, default: 1
        std-value to which the real input data was normalised

    device: str, default: "cuda"
        device to which to send variables


    Returns
    -----------

    generated_events: np.array
        array of generated events
    """

    network.eval()

    #sample representative distribution of n_eff from kde
    sampled_kde = kde.resample(n_tot_generation, seed = rng)
    #make sure sampled values are int and in range from 1 to n_points
    sampled_kde = np.rint(sampled_kde)
    sampled_kde = sampled_kde[(sampled_kde >= 1) & (sampled_kde <= n_points)]



    unique_vals, unique_freqs = np.unique(sampled_kde, return_counts = True)
    #now generate for every value n_eff in unique_vals the required amount of data
    generated_events_list = []
    for j, unique_freq in enumerate(unique_freqs):
        #find out current batch size, accounting for the possibility that for
        #each unique value the last batch size might be < batch_size
        n_full_batches, remaining_batch_size = divmod(unique_freq, batch_size)
        k = 0
        while True:
            if k < n_full_batches:
                current_batch_size = batch_size
                k += 1
            elif remaining_batch_size > 0:
                current_batch_size = remaining_batch_size
                remaining_batch_size = 0
            else:
                break

            n_eff = int(unique_vals[j])
            #produce simulated jets of size [current_batch_size, n_eff, dim_particle]
            #mimicking the structure of the JetNet-datasets
            noise_global, noise_particle = data_proc.get_noise(n_eff, batch_size = current_batch_size,
                                            dim_global = dim_global, dim_particle = dim_particle,
                                            rng = rng, device = device)

            #generating the simulated events
            #pull the result back to cpu, detatched from the network (we are only
            #interested in the resulting events)
            gen_out_no_pad = network(noise_particle, noise_global).detach().cpu().numpy()

            if inv_normalise_data:
                gen_out_no_pad = data_proc.inverse_normalise_dataset(gen_out_no_pad, inv_means, inv_stds, norm_sigma = inv_norm_sigma)

            #zero-padding to reobtain total particle number n_points
            gen_out = np.zeros((gen_out_no_pad.shape[0], n_points, dim_particle))

            try:
                gen_out[:,0:n_eff,:] = gen_out_no_pad
            except IndexError as e:
                logger.exception(e)
                logger.critical("""Generator has produced output of unexpected shape
                                in validation loop; expected shape [%d,%d,%d]""",
                                gen_out_no_pad.shape[0], n_eff, dim_particle)
                sys.exit()


            generated_events_list.append(gen_out)

    generated_events = np.vstack(generated_events_list)

    #shuffle, if rng is given
    #not that if shuffling is disabled, the returned events will be ordered
    #ascending in n_eff
    if rng is not None:
        permutation = rng.permutation(len(generated_events))
        generated_events = generated_events[permutation]

    if set_min_pt:
        generated_events = data_proc.set_min_pt(generated_events, min_pt)

    if order_by_pt:
        generated_events = utils.order_array_pt(generated_events)

    if center_gen:
        generated_events = utils.center_jets(generated_events)

    return generated_events
