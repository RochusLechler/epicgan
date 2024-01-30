"""Testing the implementations of the Epic-GAN-layer and the generator
and discriminator.
"""

import unittest
import torch
from utils import models


class TestModules(unittest.TestCase):
    """For a batch size of 128 (equals the default)
    """

    def setUp(self):

        self.n_points = 30
        self.hid_size_p = 128
        self.hid_size_g = 10
        self.batch_size = 128

        dummy_p_dataset     = torch.empty(size = (self.batch_size, self.n_points, 3))
        dummy_g_dataset     = torch.empty(size = (self.batch_size, 10))
        self.dummy_p_hid_dataset = torch.empty(size = (self.batch_size, self.n_points,
                                                       self.hid_size_p))
        self.dummy_g_hid_dataset = torch.empty(size = (self.batch_size, self.hid_size_g))

        dummy_gen        = models.Generator(self.n_points)
        dummy_dis        = models.Discriminator(self.n_points)
        #for epic layer, set hid_size_g_in = hid_size_p = 128 as in paper
        dummy_epic_layer = models.EpicGanLayer(self.n_points, self.hid_size_p,
                                               self.hid_size_g, self.hid_size_p)

        self.gen_out                = dummy_gen.forward(dummy_p_dataset, dummy_g_dataset)
        self.dis_out                = dummy_dis.forward(dummy_p_dataset)
        self.epic_out_p, self.epic_out_g = dummy_epic_layer.forward(self.dummy_p_hid_dataset,
                                                                    self.dummy_g_hid_dataset)

    def test_dims(self):
        self.assertEqual(tuple(self.gen_out.size()), tuple((self.batch_size, self.n_points, 3)))

        self.assertEqual(tuple(self.dis_out.size()), tuple((self.batch_size, 1)))

        self.assertEqual(self.epic_out_p.size(), self.dummy_p_hid_dataset.size())
        self.assertEqual(self.epic_out_g.size(), self.dummy_g_hid_dataset.size())
