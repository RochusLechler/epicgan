"""Implements tests regarding the dimensions of the datasets. This includes
testing the implementations of the Epic-GAN-layer and the generator
and discriminator as well as general functionalities.
"""

import unittest
import torch
from epicgan import utils, models


class TestDataset(unittest.TestCase):
    """parent class for all dataset related tests, defines dummy datasets
    """

    def setUp(self):

        self.n_points = 30
        self.hid_size_p = 128
        self.hid_size_g = 10
        self.batch_size = 128

        self.dummy_p_dataset     = torch.empty(size = (self.batch_size, self.n_points, 3))
        self.dummy_g_dataset     = torch.empty(size = (self.batch_size, 10))
        self.dummy_p_hid_dataset = torch.empty(size = (self.batch_size, self.n_points,
                                                       self.hid_size_p))
        self.dummy_g_hid_dataset = torch.empty(size = (self.batch_size, self.hid_size_g))




class TestModels(TestDataset):
    """For a batch size of 128 (equals the default)
    """

    #def setUp(self):
        #super(self.__class__).setUp()




    def test__dims(self):
        dummy_gen        = models.Generator(self.n_points)
        dummy_dis        = models.Discriminator(self.n_points)
        #for epic layer, set hid_size_g_in = hid_size_p = 128 as in paper
        dummy_epic_layer = models.EpicGanLayer(self.n_points, self.hid_size_p,
                                               self.hid_size_g, self.hid_size_p)


        self.gen_out                = dummy_gen.forward(self.dummy_p_dataset, self.dummy_g_dataset)
        self.dis_out                = dummy_dis.forward(self.dummy_p_dataset)
        self.epic_out_p, self.epic_out_g = dummy_epic_layer.forward(self.dummy_p_hid_dataset,
                                                                    self.dummy_g_hid_dataset)


        self.assertEqual(tuple(self.gen_out.size()), tuple((self.batch_size, self.n_points, 3)))

        self.assertEqual(tuple(self.dis_out.size()), tuple((self.batch_size, 1)))

        self.assertEqual(self.epic_out_p.size(), self.dummy_p_hid_dataset.size())
        self.assertEqual(self.epic_out_g.size(), self.dummy_g_hid_dataset.size())




class TestKDE(TestDataset):
    """
    """
    #def setUp(self):
        #super(self.__class__).setUp()

    def test_kde(self):
        #using dataset with incompatible dimensions
        self.assertRaises(SystemExit, utils.calc_kde, self.dummy_g_dataset)
