"""Implementing features regarding data processing
"""
import logging
import sys
import os
import h5py
import numpy as np
import pickle
from torch.utils.data import IterableDataset

logger = logging.getLogger("main")

datasets_folder = "/home/rochus/Documents/Studium/semester_pisa/cmepda/exam_project/JetNet_datasets"
file_suffix = ".hdf5"


def get_dataset_path(dataset_name):
    """returns path to required dataset

    Arguments:
    ---------------

    dataset_name: str
        specification of the dataset, either "gluon30", "quark30", "top30",
        "gluon150", "quark150" or "top150"
    """

    try:
        path = os.path.join(datasets_folder, dataset_name + file_suffix)
    except TypeError as e:
        logger.exception(e)
        logger.error("""%s is not a valid dataset specification, nothing
                     will be returned""" % dataset_name)
        return

    if not os.path.exists(path):
        logger.critical("File %s does not exist" % dataset_name)
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
    dataset = np.array(dataset["particle_features"])


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


def get_kde(dataset_name):
    """loads the (precomputed) kernel density estimation for the given dataset
    """
    path = datasets_folder + dataset_name + ".pkl"

    with open(path, "rb") as f:
        kde = pickle.load(f)

    return kde



class PreparedDataset(IterableDataset):
    """
    """

    def __init__(self, dataset, batch_size = 128, rng = np.random):

        super(PreparedDataset).__init__()

        self.dataset    = dataset
        self.batch_size = batch_size
        self.rng        = rng

    #def num_iter_per_ep(self):
