"""The class defined here performs the training of a model to be specified.
Due to the sheer amount of variables the ones that are usually not changed are
hard-coded above the class definition. They can be changed there.
"""

import os
import sys
import logging
import time
import tqdm

import numpy as np
import torch
from epicgan import utils, data_proc, models, evaluation





#hard-coded parameters

#learning rate of the generator
lr_G         = 1e-4
#learning rate of the discriminator
lr_D         = 1e-4
#beta_1 parameter of the Adam optimizers
beta_1       = 0.9
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
#number of comparison runs for each Wasserstein validation step
#make sure runs is <= n_tot_generation divided by the length of validation and test set
runs = 10
#whether to set p_t coordinates of generated events to minimum value found in training set
set_min_pt   = True
#whether to normalise generated events to mean & std of training set
inv_normalise_data = True
#whether to order particles by p_t in validation loop
order_by_pt = True
#whether to center the eta- and phi-coordinates of generated events
center_gen   = True





class TrainableModel:
    """A class incorporating the GAN model. It has a member function that
    performs the training.
    When running the training, ensure the place from where you run the training
    has the following folders and contents:
    1. The dataset is stored in folder 'JetNet_datasets' in '.hdf5'-format
    2. There is a folder 'saved_models', the best model will be stored here
    3. There is a folder 'logbooks', the logfile will be stored here
    """

    def __init__(self, dataset_name, n_points, batch_size = 128, file_suffix = None, load = False, load_file_name = None):
        """Class constructor

        Arguments
        --------------
        dataset_name: str
            dataset specification

        n_points: int
            number of particles per jet, either 30 or 150

        batch_size: int, default: 128
            batch size used for the training

        file_suffix: str, default: None
            suffix added to the logfile name and the filename of the model that
            will be saved; if None, defaults to "main"

        load : bool, default: False
            if True, tries to initialise the model by loading a model according
            to dataset_name

        load_file_name: str, default: None
            file name from which to load the model; needs to be specified whenever
            load is set to True
        """

        self.dataset_name = dataset_name

        if file_suffix is None:
            file_suffix = "main"

        try:
            self.file_name_suffix = self.dataset_name + file_suffix
        except TypeError:
            print("file_suffix should be a string")
            sys.exit()

        self.n_points = n_points
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.real_label = 1.
        self.fake_label = 0.
        #load the dataset
        self.dataset = data_proc.get_dataset(self.dataset_name)
        #split into sets according to splits = [0.7, 0.15, 0.15]
        self.train_set, self.val_set, self.test_set = data_proc.split_dataset(self.dataset, rng = rng)

        #load the precomputed kde for this dataset
        self.kde = data_proc.get_kde(self.dataset_name)

        #get the properties needed for normalisation
        self.train_set_means, self.train_set_stds, self.train_set_mins, self.train_set_maxs = data_proc.dataset_properties(self.train_set)

        #initialise the models
        self.generator = models.Generator(self.n_points, input_size_p = dim_particle,
                     input_size_g = dim_global, hid_size_p = 128, hid_size_g = 10,
                     hid_size_g_in = 128, num_epic_layers = num_epic_layers_gen)
        self.discriminator = models.Discriminator(self.n_points, input_size_p = dim_particle,
                     input_size_g = dim_global, hid_size_p = 128, hid_size_g = 10,
                     hid_size_g_in = 128, num_epic_layers = num_epic_layers_dis)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        #initialise optimisers
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr = lr_G,
                                            betas = (beta_1, 0.999), eps = 1e-14)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr = lr_D,
                                            betas = (beta_1, 0.999), eps = 1e-14)

        #put models in training mode
        self.generator.train()
        self.discriminator.train()


        #use custom class to prepare dataset
        self.dataset = data_proc.PreparedDataset(self.train_set, batch_size = self.batch_size, rng = rng)
        self.num_iter_per_ep = self.dataset.num_iter_per_ep()

        #specify self.batch_size to None to ensure dataloader will employ __iter__ method
        #as defined in PreparedDataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = None)

        self.epoch_counter = 0

        self.best_w_dist = 0
        self.best_epoch = 0
        self.test_w_distance = 0


        self.loss_gen = 0
        self.loss_dis = 0
        #lists track the mean loss per batch for every epoch
        self.mean_loss_gen_list = []
        self.mean_loss_dis_list = []
        #tracks the Wasserstein distance on the validation set for every epoch
        self.w_dist_list = []

        log_folder = "./logbooks"

        logfile_name = "logbook_training_" + self.file_name_suffix + ".log"

        logging.basicConfig(format = '%(asctime)s[%(levelname)s] %(funcName)s: %(message)s',
                                  datefmt = '%d/%m/%Y %I:%M:%S %p',
                                  filename = os.path.join(log_folder, logfile_name),
                                  level = logging.INFO,
                                  filemode = 'w')

        self.logger = logging.getLogger("main")

        if load:
            try:
                self.generator, self.discriminator, self.optimizer_g, self.optimizer_d = utils.load_model(self.generator,
                                                                                                    self.discriminator,
                                                                                                    self.optimizer_g,
                                                                                                    self.optimizer_d,
                                                                                                    file_name = load_file_name,
                                                                                                    device = self.device)
            except FileNotFoundError as e:
                self.logger.exception(e)
                self.logger.warning("""Could not find a loadable model; training will start from scratch""")
                print("Warning: No loadable model found, the training will start from scratch")


        self.logger.info("used device is %s", self.device)
        self.logger.info("Model and data initialised for %s", self.dataset_name)
        self.logger.info("Training will take %d iterations per epoch", self.num_iter_per_ep)


    def training(self, num_epochs, loss = "BCE"):
        """This function performs the actual training of the specified model.
        As optimizers for both generator and discriminator Adam is used.

        Arguments
        -----------

        num_epochs: int
            number of epochs for which to run training

        loss: str, default: "BCE"
            whether to use binary cross entropy ("BCE") or least-squares-GAN loss
            ("LS-GAN") for training; defaults to "BCE"


        Returns
        -----------

        self.test_w_distance: float
            Wasserstein distance score on the test set for the best epoch

        self.best_epoch: int
            epoch for which the Wasserstein score on the validation set was lowest

        self.mean_loss_dis_list: list
            list of all the average loss values of the discriminator training

        self.mean_loss_gen_list: list
            list of all the average loss values of the generator training

        self.w_dist_list: list
            list of all the Wasserstein distance scores on the validation set

        """
        self.logger.info("loss is %s", loss)

        #start the training loop
        start = time.time()
        iteration_counter = 0
        #set epoch counter to 0 explicitly for case of multiple trainings
        self.epoch_counter = 0

        #use tqdm in order to display a progress bar
        iterator = tqdm.tqdm(self.dataloader, total = int(self.num_iter_per_ep*num_epochs - 1))
        #training loop
        for batch in iterator:
            iteration_counter += 1

            #validation loop
            if iteration_counter % self.num_iter_per_ep == 0:
                self.logger.info("Epoch %d done", int(self.epoch_counter + 1))

                self.loss_dis /= self.num_iter_per_ep
                self.loss_gen /= self.num_iter_per_ep
                self.mean_loss_dis_list.append(self.loss_dis)
                self.mean_loss_gen_list.append(self.loss_gen)
                self.logger.info("losses: Discriminator: %.3f; Generator: %.3f", self.loss_dis, self.loss_gen)
                self.loss_dis = 0
                self.loss_gen = 0

                self.validation_loop()

                self.epoch_counter += 1

            if self.epoch_counter == num_epochs: #breaking condition
                self.logger.info("All %d epochs done, training finished", num_epochs)
                self.logger.info("Best epoch was epoch %d with a Wasserstein distance of %.5f", self.best_epoch, self.test_w_distance)
                iterator.close()
                break

            if inv_normalise_data:
                batch = data_proc.normalise_dataset(batch, self.train_set_means, self.train_set_stds, norm_sigma = norm_sigma)

            data = batch.to(self.device)
            #this might be smaller than self.batch_size
            local_batch_size = data.size(0)

            #Discriminator training
            self.discriminator_training(data, local_batch_size, loss = loss)

            #generator training
            self.generator_training(local_batch_size, loss = loss)

        total_time = time.time() - start
        hours, rest = divmod(total_time, 3600)
        mins, secs  = divmod(rest, 60)
        self.logger.info("In total the training took %d h %d min %d sec", hours, mins, secs)

        return self.test_w_distance, self.best_epoch, self.mean_loss_dis_list, self.mean_loss_gen_list, self.w_dist_list




    def validation_loop(self):
        """Performs a single validation loop
        """

        self.mean_loss_gen_list.append(self.loss_gen/self.num_iter_per_ep)
        self.mean_loss_dis_list.append(self.loss_dis/self.num_iter_per_ep)
        self.loss_gen = 0
        self.loss_dis = 0

        w_distance = evaluation.compute_wasserstein_distance(self.generator, self.val_set, self.kde,
                        batch_size = self.batch_size, n_tot_generation = n_tot_generation,
                        dim_global = dim_global, dim_particle = dim_particle,
                        rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                        center_gen = center_gen, order_by_pt = order_by_pt,
                        inv_normalise_data = inv_normalise_data,
                        inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                        inv_norm_sigma = norm_sigma, runs = runs, device = self.device)

        self.logger.info("Wasserstein-distance for epoch %d is %.5f", int(self.epoch_counter+1), w_distance)

        #
        if self.epoch_counter == 0: #initialise running variables after first epoch
            self.best_w_dist = w_distance
            self.best_epoch = self.epoch_counter + 1
            #get Wasserstein distance for the test set
            self.test_w_distance = evaluation.compute_wasserstein_distance(self.generator, self.test_set, self.kde,
                            batch_size = self.batch_size, n_tot_generation = n_tot_generation,
                            dim_global = dim_global, dim_particle = dim_particle,
                            rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                            center_gen = center_gen, order_by_pt = order_by_pt,
                            inv_normalise_data = inv_normalise_data,
                            inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                            inv_norm_sigma = norm_sigma, runs = runs, device = self.device)
            self.w_dist_list.append(self.test_w_distance)

            utils.save_model(self.generator, self.discriminator, self.optimizer_g, self.optimizer_d,
                            file_name = self.file_name_suffix)

            self.logger.info("first epoch done, model saved")
            self.logger.info("Its Wasserstein distance on the test set is %.5f", self.test_w_distance)

        else: #from second epoch on, do this
            if w_distance < self.best_w_dist: # -> better model found
                self.best_w_dist = w_distance
                self.best_epoch = self.epoch_counter + 1

                self.test_w_distance = evaluation.compute_wasserstein_distance(self.generator, self.test_set, self.kde,
                                batch_size = self.batch_size, n_tot_generation = n_tot_generation,
                                dim_global = dim_global, dim_particle = dim_particle,
                                rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                                center_gen = center_gen, order_by_pt = order_by_pt,
                                inv_normalise_data = inv_normalise_data,
                                inv_means = self.train_set_means, inv_stds = self.train_set_stds,
                                inv_norm_sigma = norm_sigma, runs = runs, device = self.device)
                self.w_dist_list.append(self.test_w_distance)


                utils.save_model(self.generator, self.discriminator, self.optimizer_g, self.optimizer_d,
                                file_name = self.file_name_suffix)

                self.logger.info("Better model found and saved after epoch %i", int(self.epoch_counter+1))
                self.logger.info("Its Wasserstein distance on the test set is %.5f", self.test_w_distance)




    def discriminator_training(self, data, local_batch_size, loss = "BCE"):
        """This function performs the discriminator training

        Arguments
        ------------

        data: torch.Tensor
            data batch of size local_batch_size to use for training step

        local_batch_size: int
            number of samples in data, equals self.batch_size most of the time,
            but can be smaller when number of samples for every effective particle
            multiplicity is not divisible by self.batch_size

        loss: str, default: "BCE"
            defines which loss to use, either binary cross entropy ("BCE") or
            least-squares-GAN loss ("LS-GAN")
        """

        self.discriminator.train()
        self.generator.eval()

        self.optimizer_d.zero_grad()
        self.discriminator.zero_grad()

        noise_global, noise_particle = data_proc.get_noise(self.n_points, batch_size = local_batch_size,
                                        dim_global = dim_global, dim_particle = dim_particle,
                                        rng = rng, device = self.device)

        gen_out = self.generator(noise_particle, noise_global)
        #normalise output
        gen_out = data_proc.normalise_dataset(gen_out, means = self.train_set_means,
                                            stds = self.train_set_stds, norm_sigma = norm_sigma)

        #outputs of the Discriminator
        discr_out_real = self.discriminator(data)
        discr_out_fake = self.discriminator(gen_out)

        #loss is the least-squares-GAN loss
        if loss == "LS-GAN":
            discr_loss = 0.5 * (torch.mean(torch.square(discr_out_real - self.real_label)) + torch.mean(torch.square(discr_out_fake - self.fake_label)))
        elif loss == "BCE":
            loss = torch.nn.BCEWithLogitsLoss()
            real_label_tensor = torch.full((local_batch_size,1), self.real_label, dtype = torch.float, device = self.device)
            fake_label_tensor = torch.full((local_batch_size,1), self.fake_label, dtype = torch.float, device = self.device)
            label_tensor = torch.cat((real_label_tensor, fake_label_tensor), dim = 0)

            output_tensor = torch.cat((discr_out_real, discr_out_fake), dim = 0)
            discr_loss = loss(output_tensor, label_tensor)
        else:
            self.logger.critical("""invalid loss specification, possibilities are
                            "LS-GAN" or "BCE" """)
            sys.exit()

        self.loss_dis += discr_loss.item()

        #compute gradients, perform update
        discr_loss.backward()
        self.optimizer_d.step()


    def generator_training(self, local_batch_size, loss = "BCE"):
        """This function performs the generator training

        Arguments
        -------------

        local_batch_size: int
            number of samples in data used for discriminator training, equals
            self.batch_size most of the time, but can be smaller when number of
            samples for every effective particle multiplicity is not divisible
            by self.batch_size

        loss: str, default: "BCE"
            defines which loss to use, either binary cross entropy ("BCE") or
            least-squares-GAN loss ("LS-GAN")
        """

        self.discriminator.eval()
        self.generator.train()

        self.optimizer_g.zero_grad()
        self.generator.zero_grad()

        noise_global, noise_particle = data_proc.get_noise(self.n_points, batch_size = local_batch_size,
                                        dim_global = dim_global, dim_particle = dim_particle,
                                        rng = rng, device = self.device)

        gen_out = self.generator(noise_particle, noise_global)
        #normalise output
        gen_out = data_proc.normalise_dataset(gen_out, means = self.train_set_means,
                                            stds = self.train_set_stds, norm_sigma = norm_sigma)
        #output of discriminator
        discr_out = self.discriminator(gen_out)
        #loss: real_label, because generator wants to fool discriminator
        if loss == "LS-GAN":
            gen_loss = 0.5 * torch.mean((discr_out - self.real_label)**2)
        elif loss == "BCE":
            loss = torch.nn.BCEWithLogitsLoss()
            label_tensor = torch.full((local_batch_size, 1), self.real_label, device = self.device)
            gen_loss = loss(discr_out, label_tensor)

        self.loss_gen += gen_loss.item()
        #gradient and update
        gen_loss.backward()
        self.optimizer_g.step()





#self.test_w_dists_p = evaluation.evaluation_means(self.generator,
#            self.test_set, self.kde, calc_fpnd = calc_fpnd, calc_w_dist_p = calc_w_dist_p,
#            dataname = dataset_name, self.batch_size = self.batch_size,
#            n_tot_generation = n_tot_generation,
#            dim_global = dim_global, dim_particle = dim_particle,
#            rng = rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
#            center_gen = center_gen, inv_normalise_data = inv_normalise_data,
#            inv_means = self.train_set_means, inv_stds = self.train_set_stds,
#            inv_norm_sigma = norm_sigma, runs = runs, device = self.device)
