"""Implements tests regarding the models and the datasets. This includes
testing the implementations of the Epic-GAN-layer and the generator
and discriminator as well as general functionalities.
"""
import os

import unittest
import numpy as np
import torch
from epicgan import utils, models, data_proc


class TestModels(unittest.TestCase):
    """Tests related to the NNs.
    """

    def setUp(self):

        self.n_points = 30
        hid_size_p = 128
        hid_size_g = 10
        self.batch_size = 128
        dim_p = 3
        dim_g = 10

        self.dummy_p_data_batch     = torch.empty(size = (self.batch_size, self.n_points, dim_p))
        self.dummy_g_data_batch     = torch.empty(size = (self.batch_size, dim_g))
        self.dummy_p_hid_data_batch = torch.empty(size = (self.batch_size, self.n_points,
                                                       hid_size_p))
        self.dummy_g_hid_data_batch = torch.empty(size = (self.batch_size, hid_size_g))

        #construct dummy networks with default values
        self.dummy_gen = models.Generator()
        self.dummy_dis = models.Discriminator()
        #for epic layer, set hid_size_g_in = hid_size_p = 128 as in paper
        self.dummy_epic_layer = models.EpicGanLayer(hid_size_p, hid_size_g, hid_size_p)


    @classmethod
    def classTearDown(self):
        os.remove("./saved_models/test_save.tar")


    def test_dims(self):
        """Various tests on the dimensionality of datasets after passing them through networks.
        """

        gen_out                = self.dummy_gen(self.dummy_p_data_batch, self.dummy_g_data_batch)
        dis_out                = self.dummy_dis(self.dummy_p_data_batch)
        epic_out_p, epic_out_g = self.dummy_epic_layer.forward(self.dummy_p_hid_data_batch,
                                                                    self.dummy_g_hid_data_batch)


        self.assertEqual(tuple(gen_out.size()), tuple((self.batch_size, self.n_points, 3)))
        self.assertEqual(tuple(dis_out.size()), tuple((self.batch_size, 1)))

        self.assertEqual(epic_out_p.size(), self.dummy_p_hid_data_batch.size())
        self.assertEqual(epic_out_g.size(), self.dummy_g_hid_data_batch.size())


    def test_save_load(self):
        """Tests that saving and loading of model works
        """

        optimizer_g = torch.optim.Adam(self.dummy_gen.parameters(), lr = 1e-4,
                                            betas = (0.9, 0.999), eps = 1e-14)
        optimizer_d = torch.optim.Adam(self.dummy_dis.parameters(), lr = 1e-4,
                                            betas = (0.9, 0.999), eps = 1e-14)
        
        folder = "./saved_models"
        file_name = "test_save"
        utils.save_model(self.dummy_gen, self.dummy_dis, optimizer_g, optimizer_d,
                file_name, folder = folder)
        
        utils.load_model(self.dummy_gen, self.dummy_dis, optimizer_g, optimizer_d,
                file_name, folder = folder, device = "cpu")


    def test_kde(self):
        """Assures that KDE-calculation exits when given invalid dataset
        """
        #using dataset with incompatible dimensions
        self.assertRaises(SystemExit, utils.calc_kde, self.dummy_g_data_batch)



class TestDatasetClass(unittest.TestCase):


    def setUp(self):

        #don't need a large batch size here
        self.batch_size = 20
        n_points = 30
        p_dim = 3

        #simulate data array with some zero-padded particles
        size = int(10*self.batch_size)
        dummy_data = np.zeros((size, n_points, p_dim))
        for k in range(size):
            n_eff = np.random.randint(low = int(n_points-5), high = n_points)
            dummy_data[:,:n_eff,:] = np.random.normal((size, n_eff, p_dim))

        #specify a rng to ensure shuffling is performed
        self.dataset = data_proc.PreparedDataset(dummy_data, self.batch_size, rng = np.random.default_rng(1))


    def test_dataindexing_batches(self):
        """tests that program 
        """
        item = self.dataset[0]
        self.assertTrue(self.dataset.batches_defined)

        for id in self.dataset.batch_ids:
            k, l = id
            batch = self.dataset.batch_dict[k][l]
            self.assertLessEqual(batch.shape[0], self.batch_size)

        self.assertEqual(len(self.dataset.batch_ids), self.dataset.num_iter_per_ep())




    

    


