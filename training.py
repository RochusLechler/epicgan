"""This script performs the training of a model to be specified.
Due to the sheer amount of variables it defines global variables in order to
make the function calls more concise.
"""

import logging
import time
import tqdm

import numpy as np
import torch
from epicgan import utils, data_proc, models, evaluation





#default values

#which dataset to use out of gluon30, quark30, top30, gluon150, quark150, top150
dataset_name = "gluon30"
#the number of points (particles) per jet; either 30 or 150
n_points     = 30
#learning rate of the generator
lr_G         = 1e-14
#learning rate of the discriminator
lr_D         = 1e-14
#beta_1 parameter of the Adam optimizers
beta_1       = 0.9
#batch size used in training
batch_size   = 128
#dimension of the particle space; default 3 are p_t, eta, phi
dim_particle = 3
#dimension of the global variable space within the networks
dim_global   = 10
#number of EPiC-layers in the generator
num_epic_layers_gen = 6
#number of EPiC-layers in the discriminator
num_epic_layers_dis = 3
#random number generator used throughout the script for shuffling
rng          = np.random.default_rng(3)
#used to normalise input data to this std
norm_sigma   = 5
#total number of events generated for each evaluation assessment
n_tot_generation = 300000
#n_tot_generation = 3000 #just for now
#number of comparison runs for each Wasserstein validation step
#make sure runs is <= the number of times the length of validation/test sets fit into n_tot_generation
runs = 10
#number of epochs to be performed
num_epochs = 3
#whether to set p_t coordinates of generated events to specified minimum value
set_min_pt = True
#whether to normalise generated events to mean & std of training set
inv_normalise_data = True
#whether to center the eta- and phi-coordinates of generated events
center_gen = True
#whether to calculate Wasserstein distances for particle features in ev. loop
calc_w_dist_p = True
#whether to calculate FPND in ev. loop; only possible for multiplicity 30
if n_points == 30:
    calc_fpnd = True
elif n_points == 150:
    calc_fpnd = False


logfile_name = "logbook_" + dataset_name + ".log"
logging.basicConfig(format = '%(asctime)s[%(levelname)s] %(funcName)s: %(message)s',
                          datefmt = '%d/%m/%Y %I:%M:%S %p',
                          filename = logfile_name,
                          level = logging.DEBUG,
                          filemode = 'w')


logger = logging.getLogger("main")



### setup training
"""Here the training is initialised
"""


class TrainableModel:
    """A class incorporating the GAN model. It has a member function that
    performs the training.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.real_label = 1
        self.fake_label = 0
        #load the dataset
        self.dataset = data_proc.get_dataset(dataset_name)
        #split into sets according to splits = [0.7, 0.15, 0.15]
        self.train_set, self.val_set, self.test_set = data_proc.split_dataset(self.dataset, rng = rng)

        #load the precomputed kde for this dataset
        self.kde = data_proc.get_kde(dataset_name)

        #get the properties needed for normalisation
        self.train_set_means, self.train_set_stds, self.train_set_mins, self.train_set_maxs = data_proc.dataset_properties(self.train_set)

        #initialise the models
        self.generator = models.Generator(n_points, input_size_p = dim_particle,
                     input_size_g = dim_global, hid_size_p = 128, hid_size_g = 10,
                     hid_size_g_in = 128, num_epic_layers = num_epic_layers_gen)
        self.discriminator = models.Discriminator(n_points, input_size_p = dim_particle,
                     input_size_g = dim_global, hid_size_p = 128, hid_size_g = 10,
                     hid_size_g_in = 128, num_epic_layers = num_epic_layers_dis)

        #initialise optimisers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr = lr_G,
                                            betas = (beta_1, 0.999), eps = 1e-14)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr = lr_D,
                                            betas = (beta_1, 0.999), eps = 1e-14)

        #put models in training mode
        self.generator.train()
        self.discriminator.train()

        #normalise the training set
        self.train_set = data_proc.normalise_dataset(self.train_set, self.train_set_means,
                                        self.train_set_stds, norm_sigma = norm_sigma)

        #use custom class to prepare dataset
        self.dataset = data_proc.PreparedDataset(self.train_set, batch_size = batch_size, rng = rng)
        self.num_iter_per_ep = self.dataset.num_iter_per_ep()

        #specify batch_size to None to ensure dataloader will employ __iter__ method
        #as defined in PreparedDataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = None)

        self.epoch_counter = 0

        self.best_w_dist = 0
        self.best_epoch = 0
        self.test_w_distance = 0
        self.test_w_dists_p = 0
        if n_points == 30:
            self.test_fpnd = 0



        logger.info("used device is %s", self.device)
        logger.info("Model and data initialised")
        logger.info("Training will take %d iterations per epoch", self.num_iter_per_ep)





    def training(self):
        """This function performs the actual training of the specified model.
        As optimizers for both generator and discriminator Adam is used.
        """


        #start the training loop
        start = time.time()
        iteration_counter = 0
        #set epoch counter to 0 explicitly for case of multiple trainings
        self.epoch_counter = 0

        #use tqdm in order to display a progress bar
        iterator = tqdm.tqdm(self.dataloader, total = int(self.num_iter_per_ep*num_epochs - 1))
        #training loop
        for batch in iterator:
            iteration_counter += 200
            if iteration_counter > self.num_iter_per_ep:
                iteration_counter = 0

            #validation loop
            if iteration_counter % self.num_iter_per_ep == 0:
                logger.info("Epoch %i done", self.epoch_counter + 1)

                self.validation_loop()

                self.epoch_counter += 1
                if self.epoch_counter == num_epochs: #breaking condition
                    logger.info("All %d epochs done, training finished", num_epochs)
                    logger.info("Best epoch was epoch %d with a Wasserstein distance of %.2f", self.best_epoch, self.test_w_distance)
                    iterator.close()
                    break

            data = batch.to(self.device)

            #Discriminator training
            self.discriminator_training(data)

            #generator training
            self.generator_training()

        total_time = time.time() - start
        hours, rest = divmod(total_time, 3600)
        mins, secs  = divmod(rest, 60)
        logger.info("In total the training took %f h %f min %f sec", hours, mins, secs)

        if n_points == 30:
            return self.test_w_distance, self.test_w_dists_p, self.test_fpnd, self.best_epoch
        elif n_points == 150:
            return self.test_w_distance, self.test_w_dists_p, self.best_epoch



    def validation_loop(self):
        """Performs a single validation loop
        """
        """
        w_distance = evaluation.compute_wasserstein_distance(self.generator, self.val_set, self.kde,
                        batch_size = batch_size, n_tot_generation = n_tot_generation,
                        dim_global = dim_global, dim_particle = dim_particle,
                        rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                        center_gen = center_gen, inv_normalise_data = inv_normalise_data,
                        inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                        inv_norm_sigma = norm_sigma, runs = runs, device = self.device)
        """
        #
        if self.epoch_counter == 0: #initialise running variables after first epoch
            #self.best_w_dist = w_distance
            self.best_epoch = self.epoch_counter + 1
            #get Wasserstein distance for the test set
            """self.test_w_distance = evaluation.compute_wasserstein_distance(self.generator, self.test_set, self.kde,
                            batch_size = batch_size, n_tot_generation = n_tot_generation,
                            dim_global = dim_global, dim_particle = dim_particle,
                            rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                            center_gen = center_gen, inv_normalise_data = inv_normalise_data,
                            inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                            inv_norm_sigma = norm_sigma, runs = runs, device = self.device)"""

            if n_points == 30:
                self.test_w_dists_p, self.test_fpnd = evaluation.evaluation_means(self.generator,
                            self.test_set, self.kde, calc_fpnd = calc_fpnd, calc_w_dist_p = calc_w_dist_p,
                            dataname = dataset_name, batch_size = batch_size,
                            n_tot_generation = n_tot_generation,
                            dim_global = dim_global, dim_particle = dim_particle,
                            rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                            center_gen = center_gen, inv_normalise_data = inv_normalise_data,
                            inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                            inv_norm_sigma = norm_sigma, runs = runs, device = self.device)

            elif n_points == 150:
                self.test_w_dists_p = evaluation.evaluation_means(self.generator,
                            self.test_set, self.kde, calc_fpnd = calc_fpnd, calc_w_dist_p = calc_w_dist_p,
                            dataname = dataset_name, batch_size = batch_size,
                            n_tot_generation = n_tot_generation,
                            dim_global = dim_global, dim_particle = dim_particle,
                            rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                            center_gen = center_gen, inv_normalise_data = inv_normalise_data,
                            inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                            inv_norm_sigma = norm_sigma, runs = runs, device = self.device)


            utils.save_model(self.generator, self.discriminator, self.optimizer_G, self.optimizer_D,
                            file_name = dataset_name)

            logger.info("first epoch done, model saved")
            logger.info("Wasserstein distance is %.2f", self.test_w_distance)

        else: #from second epoch on, do this
            if w_distance < self.best_w_dist: # -> better model found
                self.best_w_dist = w_distance
                self.best_epoch = self.epoch_counter + 1

                self.test_w_distance = evaluation.compute_wasserstein_distance(self.generator, self.test_set, self.kde,
                                batch_size = batch_size, n_tot_generation = n_tot_generation,
                                dim_global = dim_global, dim_particle = dim_particle,
                                rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                                center_gen = center_gen, inv_normalise_data = inv_normalise_data,
                                inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                                inv_norm_sigma = norm_sigma, runs = runs, device = self.device)

                if n_points == 30:
                    self.test_w_dists_p, self.test_fpnd = evaluation.evaluation_means(self.generator,
                                self.test_set, self.kde, calc_fpnd = calc_fpnd, calc_w_dist_p = calc_w_dist_p,
                                dataname = dataset_name, batch_size = batch_size,
                                n_tot_generation = n_tot_generation,
                                dim_global = dim_global, dim_particle = dim_particle,
                                rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                                center_gen = center_gen, inv_normalise_data = inv_normalise_data,
                                inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                                inv_norm_sigma = norm_sigma, runs = runs, device = self.device)

                elif n_points == 150:
                    self.test_w_dists_p = evaluation.evaluation_means(self.generator,
                                self.test_set, self.kde, calc_fpnd = calc_fpnd, calc_w_dist_p = calc_w_dist_p,
                                dataname = dataset_name, batch_size = batch_size,
                                n_tot_generation = n_tot_generation,
                                dim_global = dim_global, dim_particle = dim_particle,
                                rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                                center_gen = center_gen, inv_normalise_data = inv_normalise_data,
                                inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                                inv_norm_sigma = norm_sigma, runs = runs, device = self.device)

                utils.save_model(self.generator, self.discriminator, self.optimizer_G, self.optimizer_D,
                                file_name = dataset_name)

                logger.info("Better model found and saved after epoch %i", int(self.epoch_counter+1))
                logger.info("Its Wasserstein distance on the test set is %.2f", self.test_w_distance)




    def discriminator_training(self, data):
        """This function performs the discriminator training
        """

        self.discriminator.train()
        self.generator.eval()

        self.discriminator.zero_grad()
        self.optimizer_D.zero_grad()

        noise_global, noise_particle = data_proc.get_noise(n_points, batch_size = batch_size,
                                        dim_global = dim_global, dim_particle = dim_particle,
                                        rng = rng, device = self.device)

        gen_out = self.generator(noise_particle, noise_global)
        #normalise output
        gen_out = data_proc.normalise_dataset(gen_out, means = self.train_set_means,
                                            stds = self.train_set_stds, norm_sigma = norm_sigma)

        print("1st data entry: ", data[0,0,0])
        #outputs of the Discriminator
        discr_out_real = self.discriminator(data)
        discr_out_fake = self.discriminator(gen_out)

        #loss is the least-squares-GAN loss
        discr_loss = 0.5 * (torch.mean((discr_out_real - self.real_label)**2) + torch.mean((discr_out_fake - self.fake_label)**2))
        #compute gradients, perform update
        discr_loss.backward()
        self.optimizer_D.step()


    def generator_training(self):
        """This function performs the generator training
        """

        self.discriminator.eval()
        self.generator.train()
        self.generator.zero_grad()
        self.optimizer_G.zero_grad()

        noise_global, noise_particle = data_proc.get_noise(n_points, batch_size = batch_size,
                                        dim_global = dim_global, dim_particle = dim_particle,
                                        rng = rng, device = self.device)

        gen_out = self.generator(noise_particle, noise_global)
        #normalise output
        gen_out = data_proc.normalise_dataset(gen_out, means = self.train_set_means,
                                            stds = self.train_set_stds, norm_sigma = norm_sigma)
        #output of discriminator
        discr_out = self.discriminator(gen_out)
        #loss: real_label, because generator wants to fool discriminator
        gen_loss = 0.5 * torch.mean((discr_out - self.real_label)**2)
        #gradient and update
        gen_loss.backward()
        self.optimizer_G.step()
