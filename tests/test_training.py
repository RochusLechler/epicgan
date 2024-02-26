"""Unit tests for the training class
"""

import os
import unittest
import numpy as np
from torch import Tensor

import training

from epicgan import data_proc
import evaluate_performance


class TestTraining(unittest.TestCase):
    """tests the implicit methods of the custom training class, as well 
    as one full training loop including a validation step on a dummy
    dataset. In the setup, the dummy dataset is created, if it doesn't 
    exist already.
    """

    def setUp(self):

        dummy_set_file_path = "JetNet_datasets/test.hdf5"
        if not os.path.exists(dummy_set_file_path):
            import h5py
            file_new = h5py.File(dummy_set_file_path, "w")
            data = np.zeros((200,30,3))
            for k in range(200):
                randi = np.random.randint(low = 20, high = 31)
                data[k,:randi,:] = np.random.normal(size = (randi,3))
            file_new.create_dataset("particle_features", (200,30,3), dtype = "f", data = data)


        self.model = training.TrainableModel("test", file_suffix = "test", rng = np.random.default_rng(1))

    @classmethod
    def tearDownClass(cls):

        path = "logbooks/logbook_training_testtest.log"
        if os.path.exists(path):
            os.remove(path)

        path = "saved_models/testtest.tar"
        if os.path.exists(path):
            os.remove(path)

        path = "saved_models/test_training_test.pkl"
        if os.path.exists(path):
            os.remove(path)

        path = "saved_models/eval_scores_testtest.pkl"
        if os.path.exists(path):
            os.remove(path)

        path = "saved_plots/testtest.png"
        if os.path.exists(path):
            os.remove(path)

    def test_training_steps(self):
        batch = Tensor(self.model.dataset[0])
        batch_size = batch.shape[0]

        self.model.discriminator_training(batch, batch_size)
        self.model.generator_training(batch_size)

    def test_validation_step(self):

        self.model.validation_loop(n_tot_generation = 200, runs = 5, batch_size_gen = 150, 
                            set_min_pt = True, order_by_pt = True, normalise_data = True, center_gen = True)
        self.model.epoch_counter += 1
        self.model.validation_loop(n_tot_generation = 200, runs = 5, batch_size_gen = 150, 
                            set_min_pt = True, order_by_pt = True, normalise_data = True, center_gen = True)

    def test_training_loop(self):

        self.model.training(num_epochs = 1, save_result_dict = True, n_tot_generation = 200, runs = 5)

    def test_evaluation_method(self):

        self.model.evaluation(make_plots = True, save_plots = True, save_result_dict = True, 
                              n_tot_generation = 200, runs = 5)

