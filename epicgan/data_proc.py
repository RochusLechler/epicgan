"""Implementing features regarding data processing.
"""
import logging
import sys
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset

from epicgan import utils

logger = logging.getLogger("main")

#folder where the datasets are stored in .hdf5 format,
#pickle-files for the KDE also stored here
datasets_folder = "./JetNet_datasets/"



def get_dataset_path(dataset_name):
    """Returns the path to the specified dataset. The folder 'JetNet_datasets' is hard-coded, the 
    datasets are to be stored here in the original .df5-format.

    Arguments
    ---------------

    dataset_name: str
        specification of the dataset, either "gluon30", "quark30", "top30",
        "gluon150", "quark150" or "top150"

    Returns
    ------------

    path: str
        path of the dataset in memory

    """
    file_suffix = ".hdf5"
    try:
        path = os.path.join(datasets_folder, dataset_name + file_suffix)
    except TypeError as e:
        logger.exception(e)
        logger.critical("%s is not a valid dataset specification", dataset_name)
        sys.exit()

    if not os.path.exists(path):
        logger.critical("File %s does not exist", path)
        sys.exit()

    return path



def get_dataset(dataset_name, drop_mask = True, reorder = True):
    """Returns the specified dataset as a numpy array.

    Arguments
    ---------------

    dataset_path: str
        path to the object to be loaded

    drop_mask: bool, default = True
        if True, the fourth particle feature of the datasets "mask" will be dropped

    reorder: bool, default = True
        if True, the particle features will be ordered [p_t, eta, phi]; should
        always be set to False, when drop_mask is False

    Returns
    ------------

    dataset: np.array
        contains the dataset specified in dataset_name
    """

    dataset_path = get_dataset_path(dataset_name)

    if reorder and not drop_mask:
        logger.critical("""drop_mask was set to False, don't know how to handle
                        reorder = True in this case""")
        sys.exit()


    dataset = h5py.File(dataset_path, "r")
    dataset = np.array(dataset["particle_features"], dtype = "float32")


    if not reorder:
        if not drop_mask:
            return dataset
        return dataset[:,:,0:3]

    return (dataset[:,:,0:3])[:,:,[2,0,1]]



def dataset_properties(dataset_train):
    """Computes mean, standard deviation, minimum value and maximum value for
    each particle feature not considering particles that were zero-padded.

    Arguments
    ---------------

    dataset_train: np.ndarray
        the dataset to be evaluated, usually the training set


    Returns
    ------------

    means: list
        contains the mean value of each particles feature

    stds: list
        contains the standard deviation of each particle feature

    mins: list
        contains the minimum value of each particle feature

    maxs: list
        contains the maximum value of each particle feature
    """

    means = []
    stds = []
    mins = []
    maxs = []
    #assuming shape [total_size, n_points, n_features = 3]
    for i in range(3):
        data = dataset_train[:,:,i].flatten()
        #remove the particles that were added via zero-padding
        data = data[data != 0.]
        means.append(data.mean())
        stds.append(data.std())
        mins.append(data.min())
        maxs.append(data.max())

    return means, stds, mins, maxs



def split_dataset(dataset, splits = [0.7, 0.15, 0.15], rng = None):
    """Splits the dataset according to given splits. It is possible to
    pass random generator for shuffling.

    Arguments
    -------------

    dataset: np.array
        dataset to be split

    splits: list, tuple or np.array, default: [0.7, 0.15, 0.15]
        fractions according to which the dataset is split into training set,
        validation set, test set; should contain 3 non-negative entries adding
        up to 1.

    rng: np.random.Generator, default: None
        random number generator; if equals None, no shuffling will be performed


    Returns
    ------------

    train_set: np.array
        dataset containing percentage of original dataset according to first
        entry of splits.

    val_set: np.array
        dataset containing percentage of original dataset according to second
        entry of splits.

    test_set: np.array
        dataset containing percentage of original dataset according to third
        entry of splits.
    """

    total_length = len(dataset)
    if rng is not None:
        permutation = rng.permutation(total_length)
        dataset = dataset[permutation]

    train_set = dataset[0:int(splits[0]*total_length)]
    val_set   = dataset[int(splits[0]*total_length):int((splits[0]+splits[1])*total_length)]
    test_set  = dataset[int((splits[0]+splits[1])*total_length):]

    return train_set, val_set, test_set


def normalise_dataset(data, means, stds, norm_sigma = 5):
    """Normalises the given data to mean 0 and standard deviation norm_sigma.
    Expected shape is [len_data, n_points, n_features]

    Arguments
    -------------

    data: np.array
        contains data of expected shape [len_data, n_points, n_features]

    means: list or np.array
        contains the mean value for every feature, has shape [n_features]

    stds: list or np.array
        contains the standard deviation for every feature, has shape [n_features]

    norm_sigma: int or float, default: 5
        standard deviation to which all of the data is normalised

    Returns
    ----------

    data: np.array
        contains the normalised dataset
    """

    n_features = data.shape[2]
    for j in range(n_features):
        data[:,:,j] = (data[:,:,j] - means[j])/(stds[j]/norm_sigma)

    return data


def inverse_normalise_dataset(data, means, stds, norm_sigma = 5):
    """Normalises the given data to specified mean values and standard deviations.
    Expected shape is [len_data, n_points, n_features]

    Arguments
    -------------

    data: np.array
        contains data of expected shape [len_data, n_points, n_features]

    means: list or np.array
        contains the output mean value for every feature

    stds: list or np.array
        contains the output standard deviation for every feature

    norm_sigma: int or float, default: 5
        standard deviation to which all of the input data is normalised

    Returns
    ----------

    data: np.array
        contains the dataset normalised to mean means and standard deviation stds.
    """

    n_features = data.shape[2]
    for j in range(n_features):
        data[:,:,j] = data[:,:,j]*(stds[j]/norm_sigma) + means[j]

    return data



def set_min_pt(dataset, min_pt):
    """Sets all values of 0 < p_t < min_pt within dataset to min_pt

    Arguments
    ------------

    dataset: np.array
        dataset for which to perform the operation

    min_pt: float
        minimum value to which to set all values of p_t lower than it

    Returns
    -------------

    dataset: np.array
        datset that has minimum p_t-value min_pt
    """

    mask = (dataset[:,:,0] < min_pt) & (dataset[:,:,0] > 0)
    (dataset[:,:,0])[mask] = min_pt

    return dataset


def get_kde(dataset_name):
    """Loads the (precomputed) kernel density estimation of n_eff (number of particles
    with nonzero p_t) for the given dataset

    Arguments
    ------------

    dataset_name: str
        specifies the dataset for which to load the kde

    Returns
    -------------

    kde: scipy.stats.gaussian_kde
        gaussian_kde-object containing the specified kde

    """
    path = datasets_folder + dataset_name + ".pkl"

    
    with open(path, "rb") as f:
        kde = pickle.load(f)

    return kde


def compute_kde(dataset_name):
    """Computes the kernel density estimation of n_eff (number of particles
    with nonzero p_t) for the given dataset

    Arguments
    ------------

    dataset_name: str
        specifies the dataset for which to load the kde

    Returns
    -------------

    kde: scipy.stats.gaussian_kde
        gaussian_kde-object containing the specified kde
    """
    try:
        dataset = get_dataset(dataset_name)
    except FileNotFoundError as e:
        logger.exception(e)
        logger.critical("could not find a dataset according to your dataset")

        sys.exit()

    save_path = datasets_folder + dataset_name

    kde = utils.calc_kde(dataset, file_path = save_path)

    return kde



def get_noise(n_points, batch_size = 128, dim_global = 10, dim_particle = 3, rng = None, 
              device = "cuda"):
    """Samples the noise needed as input for the generator with mean 0 and standard
    deviation 1

    Arguments
    --------------

    n_points: int
        number of particles per jet to be generated; note that this refers to the
        number of non-zero-padded particles, it does not have to equal 30 or 150

    batch_size: int, default: 128
        batch size to be sampled

    dim_global: int, default: 10
        dimension of the space of global features

    dim_local: int, default: 3
        dimension of the space of particle features

    rng: Generator, default: None
        random number generator, specify this to ensure results are reproducible;
        if equals None, the default torch.Generator will be employed

    device: string, default: "cuda"
        device to which to send variables

    Returns
    ------------

    noise_global: torch.Tensor
        contains noise samples of shape [batch_size, dim_global]

    noise_particle: torch.Tensor
        contains noise samples of shape [batch_size, n_points, dim_particle]
    """
    if rng is not None:
        noise_global = rng.normal(loc = 0., scale = 1., size = (batch_size, dim_global))
        noise_particle = rng.normal(loc = 0., scale = 1., size = (batch_size, n_points, dim_particle))
        noise_global = torch.Tensor(noise_global).to(device)
        noise_particle = torch.Tensor(noise_particle).to(device)

    else:
        noise_global = torch.empty((batch_size, dim_global), device = device)
        noise_particle  = torch.empty((batch_size, n_points, dim_particle), device = device)
        noise_global.normal_(mean = 0.0, std = 1.0)
        noise_particle.normal_(mean = 0.0, std = 1.0)


    return noise_global, noise_particle




class PreparedDataset(IterableDataset):
    """A class preparing the dataset for training. The batches defined in method define_batches()
    contain events of equal effective particle multiplicity n_eff, i.e. equal number of particles 
    that are not zero-padded. The __iter__-method was overrode s.t. if rng is given the batches are 
    yielded in random order and after each cycle the dataset is reshuffled and the batches are re-
    defined. Conversely, if rng is None, the batches are yielded in ascending order in n_eff and for
    each cycle the same batches are yielded in the same order.
    The __iter__-method does not raise a StopIteration, that has to be implemented in the actual 
    training loop.

    Arguments
    ------------

    dataset: np.array
        dataset over which to iterate

    batch_size: int, default: 128
        batch size to be returned by an associated iterator

    rng: np.random.Generator, default: None
        random number generator used for shuffling; if equal to None, data
        will not be shuffled
    """

    def __init__(self, dataset, batch_size = 128, rng = None):

        super(PreparedDataset).__init__()

        self.dataset    = dataset
        self.batch_size = batch_size
        self.rng        = rng

        self.len_dataset = self.dataset.shape[0]

        self.batch_dict = {}
        self.batch_ids  = []
        self.batches_defined = False

    def __getitem__(self, index):
        """index refers to a batch.
        """

        if not self.batches_defined:
            self.define_batches()

        k, l  = self.batch_ids[index]
        batch = self.batch_dict[k][l]
        return batch
        


    def num_iter_per_ep(self):
        """Gives as output the number of iterations needed to complete an
        epoch given the specified batch size.
        Note that training is performed on batches each containing jets with equal
        number of particles with p_t != 0, meaning there can (and most likely will)
        be more than one batch of batch size < self.batch_size. 

        Returns
        -----------

        num_iters: int
            number of iterations required to complete an epoch
        """

        #frequencies is an array containing the number of samples corresponding to
        #each value n_eff
        _, frequencies = utils.calc_multiplicities(self.dataset)
        num_iters = 0
        for f in frequencies:
            while f > self.batch_size:
                #add an iteration for every time f fits in batch_size
                f -= self.batch_size
                num_iters += 1
            #add another iteration for the "rest" that has length <= batch_size
            num_iters += 1

        return int(num_iters)

    def define_batches(self):
        """Makes batches of equal effective particle multiplicity n_eff and
        size self.batch_size (or less for remaining samples).
        """

        #same functionality as utils.calc_multiplicities, but we need the intermediate nonzero counts
        nonzero_counts = np.count_nonzero(self.dataset[:,:,0], axis = 1)
        #we do not need the frequencies here
        unique_vals = np.unique(nonzero_counts)
        #containers for the overall dataset
        batch_ids_list  = []

        #loop over all the unique values of n_eff
        for unique_idx, unique_val in enumerate(unique_vals):
            #extract values with that specific value

            mask = nonzero_counts == unique_val
            #unique dataset for the specific value of n
            data_unique = self.dataset[mask]
            #drop the particles that were zero-padded, note that the particles
            #are ordered in descending order in p_t
            data_unique = data_unique[:,0:unique_val,:]

            n_samples = len(data_unique)
            n_batches = n_samples // self.batch_size
            #add 1 for the rest if n_samples is not a multiple of batch_size
            if n_samples % self.batch_size != 0:
                n_batches += 1

            #dictionary to contain the batches within a unique value of n
            batch_dict_unique = {}
            batch_ids_list_unique  = []
            k = 0
            for batch_idx in range(n_batches):
                batch = data_unique[k:int(k+self.batch_size)]
                #add the j-th batch into the dictionary with index j
                batch_dict_unique[batch_idx] = batch
                #add index of the unique value of n and the batch index for unique identification
                batch_ids_list_unique.append([unique_idx, batch_idx])
                k += self.batch_size

            #stack the ids to be able to draw a tuple indexing a batch with a single index
            batch_ids_unique = np.vstack(batch_ids_list_unique)

            #add dictionary entry containing all the batches for unique value of n
            self.batch_dict[unique_idx] = batch_dict_unique
            #process the indices to uniquely identify each batch
            batch_ids_list.append(batch_ids_unique)

        #stack the ids to be able to draw a tuple indexing a batch with a single index
        self.batch_ids = np.vstack(batch_ids_list)

        self.batches_defined = True




    def __iter__(self):
        """Defining how to sample a batch from the dataset. Note that this scheme
        does not raise a StopIteration. The breakout has to be implemented
        explicitly when defining the training.
        """
        #shuffle data
        if self.rng is not None:
            permutation = self.rng.permutation(self.len_dataset)
            self.dataset = self.dataset[permutation]

        #obtain batches, each containing only events with equal number of particles
        #with nonzero p_t
        if not self.batches_defined:
            self.define_batches()
        
        #shuffle batch indexing list to make sample drawing random
        if self.rng is not None:
            permutation = self.rng.permutation(len(self.batch_ids))
            self.batch_ids = self.batch_ids[permutation]


        #implement now the loop that yields the batches
        j = 0 #index used to draw samples
        while True:
            while j < len(self.batch_ids):
                #draw batch and yield it
                k, l  = self.batch_ids[j]
                batch = self.batch_dict[k][l]
                j += 1
                yield batch

            #this part will get called when j exceeds len(batch_ids), which means
            #all samples have been dealt with
            #now start over again
            j = 0
            #if shuffling, we reshuffle the whole dataset to obtain different batches
            if self.rng is not None:
                permutation = self.rng.permutation(len(self.dataset))
                self.dataset = self.dataset[permutation]

                self.define_batches()

                permutation = self.rng.permutation(len(self.batch_ids))
                self.batch_ids = self.batch_ids[permutation]
