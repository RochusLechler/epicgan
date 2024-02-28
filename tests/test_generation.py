"""Tests the event generation function.
"""



import os
import numpy as np
import unittest
from epicgan import generation



class TestGeneration(unittest.TestCase):
    """Tests the event generation function generation.generate_events()
    """

    def setUp(self):

        dummy_model_file_path = "saved_models/test_model.tar"
        if not os.path.exists(dummy_model_file_path):
            from epicgan import training, utils
            dummy_model = training.TrainableModel("test")
            utils.save_model(dummy_model.generator, dummy_model.discriminator, 
                             dummy_model.optimizer_g, dummy_model.optimizer_d,
                            file_name = "test_model")

            
    def test_generation(self):
        rng = np.random.default_rng(1)
        generation.generate_events("test", "test_model", 200, 30, rng = rng)
