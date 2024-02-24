"""Unit tests for the training class
"""

import os
import unittest
import numpy as np
from torch import Tensor

import training
from epicgan import data_proc


class TestTraining(unittest.TestCase):
    """tests the implicit methods of the custom training class
    """

    def setUp(self):
        self.model = training.TrainableModel("gluon30", file_suffix = "test", rng = np.random.default_rng(1))

        #make the validation and test set small dummy sets
        self.model.val_set = np.random.normal(size = (20,30,3))
        self.model.test_set = np.random.normal(size = (20,30,3))

    @classmethod
    def tearDownClass(cls):
        os.remove("logbooks/logbook_training_gluon30test.log")
        os.remove("saved_models/gluon30test.tar")

    def test_training_steps(self):
        batch = Tensor(self.model.dataset[0])
        batch_size = batch.shape[0]

        self.model.discriminator_training(batch, batch_size)
        self.model.generator_training(batch_size)

    def test_validation_step(self):

        self.model.validation_loop(n_tot_generation = 120, runs = 5, set_min_pt = True, order_by_pt = True, inv_normalise_data = True, center_gen = True)
        self.model.epoch_counter += 1
        self.model.validation_loop(n_tot_generation = 120, runs = 5, set_min_pt = True, order_by_pt = True, inv_normalise_data = True, center_gen = True)

    def test_trainin_loop(self):

        #make actual training set a dummy set
        dataset = np.array(np.random.normal(size = (110,30,3)), dtype = "float32")
        self.model.dataset = data_proc.PreparedDataset(dataset, batch_size = 20, rng = None)
        self.model.num_iter_per_ep = self.model.dataset.num_iter_per_ep()

        self.model.training(num_epochs = 1, n_tot_generation = 120, runs = 5)
