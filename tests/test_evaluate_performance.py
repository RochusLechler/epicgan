"""This script implements one test for the main evaluation function 
evaluate_perofrmance.evaluate_performance()
"""

import os
import unittest

import numpy as np
from torch.optim import Adam
from epicgan import models, utils
import evaluate_performance



class TestEvaluatePerformance(unittest.TestCase):
    """Testing the main evaluation function using a dummy dataset and its 
    KDE. Because the dataset is random noise and therefore unphysical, there
    might be errors that the plot axes cannot be set to logscale due to 
    negative values. 
    In the setup, the dummy dataset is created, if it doesn't already exist.
    """

    def setUp(self):
        self.dummy_gen = models.Generator()
        self.dummy_dis = models.Discriminator()
        self.dummy_opt_g = Adam(self.dummy_gen.parameters())
        self.dummy_opt_d = Adam(self.dummy_dis.parameters())

        self.filename = "test"
        self.folder = "saved_models"
        utils.save_model(self.dummy_gen, self.dummy_dis, self.dummy_opt_g, self.dummy_opt_d, self.filename, self.folder)

        dummy_set_file_path = "JetNet_datasets/test.hdf5"
        if not os.path.exists(dummy_set_file_path):
            import h5py
            file_new = h5py.File(dummy_set_file_path, "w")
            data = np.zeros((200,30,3))
            for k in range(200):
                randi = np.random.randint(low = 20, high = 31)
                data[k,:randi,:] = np.random.normal(size = (randi,3))
            file_new.create_dataset("particle_features", (200,30,3), dtype = "f", data = data)

        

    @classmethod
    def tearDownClass(cls):
        path = "saved_models/test.tar"
        if os.path.exists(path):
            os.remove(path)

        path = "saved_plots/test.png"
        if os.path.exists(path):
            os.remove(path)

        path = "saved_models/test_evaluation_test.pkl"
        if os.path.exists(path):
            os.remove(path)

        path = "logbooks/logbook_evaluation_test.log"
        if os.path.exists(path):
            os.remove(path)

    def test_evaluate_performance(self):

        evaluate_performance.evaluate_performance(self.filename, self.filename, 30, make_plots = True, save_plots = True, save_result_dict = True, save_file_name = "test", 
                                                  rng = np.random.default_rng(1), n_tot_generation = 300, runs = 5)
