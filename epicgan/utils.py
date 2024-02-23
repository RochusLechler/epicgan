"""General utilities.
"""
import logging
import sys
import os
import pickle

import numpy as np
import torch


from scipy.stats import gaussian_kde
import energyflow



logger = logging.getLogger("main")


def calc_multiplicities(p_dataset):
    """Calculates the number of particles with p_t > 0 within a jet and calculates
    frequencies of these numbers throughout the dataset.
    Assumed shape of dataset: [total_size, n_points, n_features]


    Arguments
    -------------

    p_dataset: np.array
        array containing the dataset in shape [total_size, n_points, n_features]

    Returns
    ---------

    unique_vals: np.array
        contains the unique numbers of particles with nonzero p_t within a jet
        occuring in the dataset

    frequencies: np.array
        contains the corresponding frequencies, with which the value in unique_vals
        occur.
    """

    try:
        #look for particles w/ p_t = 0
        nonzero_counts = np.count_nonzero(p_dataset[:,:,0], axis = 1)

    except IndexError as e:
        logger.exception(e)
        logger.warning("""Cannot calculate multiplicities of dataset %s, expected shape
                         [total_size, n_points, n_features]""" , p_dataset)
        sys.exit(-1)

    unique_vals, frequencies = np.unique(nonzero_counts, return_counts = True)

    return unique_vals, frequencies



def calc_kde(p_dataset, file_path = None):
    """Calculates the kernel density estimation (kde) of the number of particles
    with nonzero p_t for a jet using Gaussian kernels.
    Has an option to save the kde to a .pkl-file using pickle

    Arguments
    ----------------
    p_dataset: np.array
        array containing the dataset in shape [total_size, n_points, n_features]

    file_path: str, default = None
        When specified, the function will try to dump the calculated KDE here;
        should end in .pkl

    Returns
    -----------

    kde: scipy.stats.gaussian_kde
        gaussian_kde-object which can be resampled using resample()-method
    """


    try:
        mults = np.count_nonzero(p_dataset[:,:,0], axis = 1)
    except IndexError as e:
        logger.exception(e)
        logger.critical("""Received dataset of unexpected shape; excpected shape
                        [total_size, n_points, n_features]""")
        sys.exit()

    kde = gaussian_kde(mults)


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




def jet_masses(data):
    """Calculates the total mass of each jet within the given dataset.

    Arguments
    -------------

    data: np.array
        data for which to compute the jet masses, shape [array_size, n_points, n_features]

    Returns
    -------------

    masses: np.array
        mass for every jet, shape [array_size]
    """
    #using package energyflow, one can compute the mass from a Cartesian
    #representation of the particle  -->  transform into Cartesian coordinates,
    #then compute mass

    jets_cartesian = energyflow.p4s_from_ptyphims(data)
    masses = energyflow.ms_from_p4s(jets_cartesian.sum(axis=1))
    return masses


def jet_pts(data):
    """Calculates the total transverse momentum of each jet within the given dataset.

    Arguments
    -------------

    data: np.array
        data for which to compute the jet p_t, shape [array_size, n_points, n_features]

    Returns
    -------------

    pts: np.array
        p_t for every jet, shape [array_size]
    """

    jets_cartesian = energyflow.p4s_from_ptyphims(data)
    pts = energyflow.pts_from_p4s(jets_cartesian.sum(axis=1))
    return pts



def jet_etas(data):
    """Calculates pseudorapidity eta of each jet within the given dataset.

    Arguments
    -------------

    data: np.array
        data for which to compute eta, shape [array_size, n_points, n_features]

    Returns
    -------------

    etas: np.array
        eta for every jet, shape [array_size]
    """

    jets_cartesian = energyflow.p4s_from_ptyphims(data)
    etas = energyflow.etas_from_p4s(jets_cartesian.sum(axis = 1))
    return etas


def jet_phis(data):
    """Calculates the azimuthal angle phi of each jet within the given dataset.

    Arguments
    -------------

    data: np.array
        data for which to compute the jet masses, shape [array_size, n_points, n_features]

    Returns
    -------------

    phis: np.array
        phi for every jet, shape [array_size]
    """

    jets_cartesian = energyflow.p4s_from_ptyphims(data)
    phis = energyflow.phis_from_p4s(jets_cartesian.sum(axis = 1), phi_ref = 0)

    return phis



def jet_multiplicities(data):
    """Calculates the effective particle multiplicities n_eff in data.

    Arguments
    ------------

    data: np.array
        data for which to compute multiplicities, shape [array_size, n_points, n_features]

    Returns
    -----------

    mults: np.array
        effective particle multiplicities, shape [array_size]
    """

    try:
        mults = np.count_nonzero(data[:,:,0], axis = 1)
    except IndexError as e:
        logger.exception(e)
        logger.critical("""got data of unexpected shape, expected shape is
                        [n_samples, n_points, n_features]; this function returns
                        None""")
        return None

    return mults



########## got this function from EPiC-GAN Github, needed for evaluation plots #############
def torch_p4s_from_ptyphi(ptetaphi):
    """Calculates Cartesian four-vectors from [p_t, eta, phi]-representations.
    Particles are assumed massless.

    Arguments
    -----------

    ptetaphi: np.array
        data in [p_t, eta, phi]-representations

    Returns
    -----------

    p4s: torch.Tensor
        data in Cartesian representation
    """

    pts, etas, phis = (ptetaphi[...,0,np.newaxis],
                     ptetaphi[...,1,np.newaxis],
                     ptetaphi[...,2,np.newaxis])

    Ets = torch.sqrt(pts**2) #  + ms**2) # everything assumed massless
    p4s = torch.cat((Ets*torch.cosh(etas), pts*torch.cos(phis),
                          pts*torch.sin(phis), Ets*torch.sinh(etas)), axis=-1)
    return p4s




def center_jets(data):
    """Centers the eta- and phi-coordinates for every jet.

    Arguments
    -----------

    data: np.array
        data to be centered

    Returns
    -----------

    data: np.array
        centered data
    """


    jets_cartesian = energyflow.p4s_from_ptyphims(data)
    jets_cartesian = jets_cartesian.sum(axis = 1)
    jet_etas = energyflow.etas_from_p4s(jets_cartesian)
    jet_phis = energyflow.phis_from_p4s(jets_cartesian, phi_ref = 0)

    #recreate shape of original data
    jet_etas = jet_etas[:,np.newaxis].repeat(data.shape[1], axis = 1)
    jet_phis = jet_phis[:,np.newaxis].repeat(data.shape[1], axis = 1)


    #consider only non zero-padded particles
    mask = data[:,:,0] > 0
    data[mask,1] -= jet_etas[mask]
    data[mask,2] -= jet_phis[mask]

    return data


def order_array_pt(data):
    """Sorts input array in p_t per jet in decreasing order.

    Arguments
    ------------

    data: np.array
        unordered data

    Returns
    ----------

    ordered_data:
        ordered data
    """

    try:
        mask = np.argsort(data[:,:,0], axis = -1)
    except IndexError as e:
        logger.exception(e)
        logger.warning("""input data has unexpected shape; expected shape is
                        [n_samples, n_points, n_features]; this function
                        returns the input unchanged""")

        return data


    dim_particle = data.shape[2]
    sorted_features_list = []

    for j in range(dim_particle):
        #apply sorting mask for every particle feature s.t. entries for p_t, eta
        #and phi correspond
        #reverse order to have particles decreasing in p_t (and therefore zero-
        #padded particles at the end)
        sorted_feature = np.take_along_axis(data[:,:,j], mask, axis = -1)[:,::-1]
        sorted_features_list.append(sorted_feature)
    #stack along last axis to reobtain original shape
    ordered_data = np.stack(sorted_features_list, axis = -1)

    return ordered_data



def save_model(generator, discriminator, optimizer_g, optimizer_d,
                file_name, folder = "./saved_models"):
    """Saves the overall GAN consisting of generator, discriminator and optimizers
    for each to a file. It saves a dictionary of all the parameters for
    all the structures using the .save_dict()-method.

    Arguments
    ------------

    generator: Generator
        generator network of the GAN

    discriminator: Discriminator
        discriminator network of the GAN

    optimizer_g: torch.optim.Optimizer
        optimizer of the generator network

    optimizer_d: torch.optim.Optimizer
        optimizer of the discriminator network

    file_name: str
        filename of the file to be created

    folder: str, default: "./saved_models"
        folder where to put the file
    """

    path = os.path.join(folder, file_name + ".tar")
    save_dict = {
                "generator_state": generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "optimizer_g_state": optimizer_g.state_dict(),
                "optimizer_d_state": optimizer_d.state_dict()
    }
    torch.save(save_dict, path)


def load_model(generator, discriminator, optimizer_g, optimizer_d,
                file_name, folder = "./saved_models", device = "cuda"):
    """Loads a GAN consisting of generator, discriminator and optimizers
    for each from a file that was created using the above function save_model

    Arguments
    ------------

    generator: Generator
        generator network of the GAN

    discriminator: Discriminator
        discriminator network of the GAN

    optimizer_g: torch.optim.Optimizer
        optimizer of the generator network

    optimizer_d: torch.optim.Optimizer
        optimizer of the discriminator network

    file_name: str
        filename of the file to load

    folder: str, default: "./saved_models"
        folder where to search for the file

    Returns
    ------------

    generator: Generator
        generator network with the parameters saved in the specified file

    discriminator: Discriminator
        discriminator network with the parameters saved in the specified file

    optimizer_g: torch.optim.Optimizer
        optimizer for the generator network with state stored in specified file

    optimizer_d: torch.optim.Optimizer
        optimizer for the discriminator network with state stored in specified file
    """

    path = os.path.join(folder, file_name + ".tar")
    try:
        model = torch.load(path, map_location = device)
    except FileNotFoundError as e:
        logger.exception(e)
        logger.critical("""could not find the file you want to load, program will
                        terminate""")

        sys.exit()

    generator.load_state_dict(model['generator_state'], strict = True)
    discriminator.load_state_dict(model['discriminator_state'], strict = True)

    optimizer_g.load_state_dict(model['optimizer_g_state'])
    optimizer_d.load_state_dict(model['optimizer_d_state'])

    logger.info("Model for %s loaded", file_name)
    return generator, discriminator, optimizer_g, optimizer_d
