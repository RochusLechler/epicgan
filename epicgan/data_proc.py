"""Implementing features regarding data processing
"""
import logging
import sys
import os
import h5py
import numpy as np
import pickle
from torch.utils.data import IterableDataset, DataLoader

from epicgan import utils

logger = logging.getLogger("main")

datasets_folder = "/home/rochus/Documents/Studium/semester_pisa/cmepda/exam_project/JetNet_datasets"



def get_dataset_path(dataset_name):
    """returns path to required dataset

    Arguments:
    ---------------

    dataset_name: str
        specification of the dataset, either "gluon30", "quark30", "top30",
        "gluon150", "quark150" or "top150"
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
    """returns the dataset as a numpy object

    Arguments:
    ---------------

    dataset_path: str
        path to the object to be loaded

    drop_mask: bool, default = True
        if True, the fourth particle feature of the datasets "mask" will be dropped

    reorder: bool, default = True
        if True, the particle features will be ordered [p_t, eta, phi]; should
        always be set to False, when drop_mask is False
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
    """computes mean, standard deviation, minmum value and maximum value for
    each feature and each particle of each jet that was not zero-padded.

    Arguments:
    ---------------

    dataset_train: np.ndarray
        the dataset to be evaluated, usually the training set
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



def split_dataset(dataset, splits = [0.7, 0.15, 0.15]):
    """splits the dataset according to splits, also shuffling. It is possible to
    pass a seeded random generator for shuffling.

    """
    total_length = len(dataset)

    train_set = dataset[0:int(splits[0]*total_length)]
    val_set   = dataset[int(splits[0]*total_length):int((splits[0]+splits[1])*total_length)]
    test_set  = dataset[int((splits[0]+splits[1])*total_length):]

    return train_set, val_set, test_set


def normalise_dataset(data, means, stds, norm_sigma = 5):
    """Normalises the given data to mean 0 and standard deviation norm_sigma.
    Expected shape is [len_data, n_points, n_features]

    Arguments:
    -------------

    data: torch.Tensor
        contains data of shape [len_data, n_points, n_features]

    means: list or np.array
        contains the mean value for every feature, has shape [n_features]

    stds: list or np.array
        contains the standard deviation for every feature, has shape [n_features]

    norm_sigma: int or float, default: 5
        standard deviation to which all of the data is normalised
    """

    n_features = data.size(2)
    for j in range(n_features):
        data[:,:,j] = (data[:,:,j] - means[j])/(stds[j]/norm_sigma)

    return data



def get_kde(dataset_name):
    """loads the (precomputed) kernel density estimation for the given dataset
    """
    path = datasets_folder + dataset_name + ".pkl"

    with open(path, "rb") as f:
        kde = pickle.load(f)

    return kde


def get_noise(n_points, batch_size = 128, dim_global = 10, dim_particle = 3, rng = np.random, device = "cuda"):
    """sample the noise needed as input for the generator.

    Arguments:
    --------------

    n_points: int
        number of particles per jet, accepts 30 and 150

    batch_size: int, default: 128
        batch size to be sampled

    dim_global: int, default: 10
        dimension of the space of global features

    dim_local: int, default: 3
        dimension of the space of particle features

    rng: Generator, default: np.random
        random number generator, specify this to ensure results are reproducible

    device: string, default: "cuda"
        device to which to send variables
    """

    noise_global = torch.empty((batch_size, dim_global), device = device)
    noise_particle  = torch.empty((batch_size, n_points, dim_particle), device = device)

    noise_global.normal(mean = 0.0, std = 1.0)
    noise_particle.normal(mean = 0.0, std = 1.0)

    return noise_global, noise_particle




class PreparedDataset(IterableDataset):
    """
    """

    def __init__(self, dataset, batch_size = 128, rng = None):
        """Initialising the iterable dataset.

        Arguments:
        ------------

        dataset:

        batch_size: int, default: 128
            batch size to be returned by an associated iterator

        rng: Generator, default: None
            random number generator used for shuffling; if equal to None, data
            will not be shuffled
        """

        super(PreparedDataset).__init__()

        self.dataset    = dataset
        self.batch_size = batch_size
        self.rng        = rng

        self.len_dataset = self.dataset.shape[0]

    def __getitem__(self):
        super(PreparedDataset).__getitem__()



    def num_iter_per_ep(self):
        """Method gives as output the number of iterations needed to complete an
        epoch given the specified batch size.
        Note that training is performed on batches each containing jets with equal
        number of particles with p_t != 0
        """

        _, frequencies = utils.calc_multiplicities(self.dataset)
        num_iters = 0
        for f in frequencies:
            while f > self.batch_size:
                #add an iteration for every time f fits in batch_size
                f -= self.batch_size
                num_iters += 1
            #add another iteration for the "rest" that has length <= batch_size
            num_iters += 1

        return num_iters

    def define_batches(self, data):
        """Makes batches
        """

        #same functionality as utils.calc_multiplicities, but we need the intermediate nonzero counts
        nonzero_counts = np.count_nonzero(data[:,:,0], axis = 1)
        #we do not need the frequencies here
        unique_vals = np.unique(nonzero_counts)

        #containers for the overall dataset
        batch_dict = {}
        batch_ids_list  = []

        #loop over all the unique values
        for unique_idx, unique_val in enumerate(unique_vals):
            #extract values with that specific value

            mask = nonzero_counts == unique_val
            #unique dataset for the specific value of n
            data_unique = data[mask]
            #drop the particles that were zero-padded
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
            batch_dict[unique_idx] = batch_dict_unique
            #process the indices to uniquely identify each batch
            batch_ids_list.append(batch_ids_unique)

        #stack the ids to be able to draw a tuple indexing a batch with a single index
        batch_ids = np.vstack(batch_ids_list)

        return batch_dict, batch_ids




    def __iter__(self):
        """Defining how to sample a batch from the dataset
        """
        #shuffle data
        if self.rng is not None:
            permutation = self.rng.permutation(self.len_dataset)
            data = self.dataset[permutation]
        else:
            data = self.dataset

        #obtain batches, each containing only events with equal number of particles
        #with nonzero p_t
        batch_dict, batch_ids = self.define_batches(data)
        #shuffle batch indexing list to make sample drawing random
        if self.rng is not None:
            permutation = self.rng.permutation(len(batch_ids))
            batch_ids = batch_ids[permutation]


        #implement now the loop that yields the n_samples
        j = 0 #index used to draw samples
        while True:
            while j < len(batch_ids):
                #draw batch and yield it
                k, l  = batch_ids[j]
                batch = batch_dict[k][l]
                j += 1
                yield batch

            #this part will get called when j exceeds len(batch_ids), which means
            #all samples have been dealt with
            #now start over again
            j = 0
            #if shuffling, we reshuffle the whole dataset to obtain different batches
            if self.rng is not None:
                permutation = self.rng.permutation(len(data))
                data = data[permutation]

                batch_dict, batch_ids = define_batches(data)

                permutation = self.rng.permutation(len(batch_ids))
                batch_ids = batch_ids[permutation]
