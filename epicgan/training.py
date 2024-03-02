"""The class defined here is a trainable model. It has methods for performing training, 
evaluation and event generation.
"""

import os
import sys
import logging
import time
import pickle
import tqdm


import torch
from epicgan import utils, data_proc, models, evaluation, performance_metrics, evaluate_performance




class TrainableModel:
    """A class incorporating the GAN model. It has a method training() that performs the training 
    and a method evaluation() that performs the evaluation.\n
    When running the training, ensure the location from where you run it has the following folders 
    and contents:\n
    1. The specified dataset is stored in folder 'JetNet_datasets' in '.hdf5'-format\n
    2. There is a folder 'saved_models', the best model will be stored here\n
    3. There is a folder 'logbooks', the logfile will be stored here; the log-filename is a concate-
    nation of the dataset-name and the specified 'file_suffix'


    Arguments
    --------------
    dataset_name: str
        dataset specification

    n_points: int
        number of particles per jet, either 30 or 150

    batch_size: int, default: 128
        batch size used for training

    rng: np.random.Generator, default: None
        random number generator used for shuffling throughout the training;
        if equal to None, data will not be shuffled

    file_suffix: str, default: None
        suffix added to the logfile name and the filename of the model that
        will be saved; if None, defaults to "main"

    load : bool, default: False
        if True, tries to initialise the model by loading a model according
        to load_file_name

    load_file_name: str, default: None
        file name from which to load the model; needs to be specified whenever
        load is set to True

    **center_jets: bool, default: True
        if True, centers jets within each dataset (training, validation, test)

    **w_dist_per_iter: bool, default: False
        if True, the dictionary returned by training method contains a list of the 
        Wasserstein distances between true data and generated events for every 
        iteration (computed in the discriminator training step)

    **dim_particle: int, default: 3
        dimension of the particle space, default 3 are [p_t, eta, phi]

    **dim_global: int, default: 10
        dimension of the global variable space within the networks

    **num_epic_layers_gen: int, default: 6
        number of EPiC-layers in the generator

    **num_epic_layers_dis: int, default: 3
        number of EPiC-layers in the discriminator

    **norm_sigma: float, default: 5.
        used to normalise data to this std

    **beta_1: float, default: 0.9
        beta_1 parameter of the (Adam) optimizers
    """

    def __init__(self, dataset_name, batch_size = 128, rng = None, file_suffix = None, load = False, 
                load_file_name = None, **kwargs):

        center_jets = kwargs.get("center_jets", True)
        self.w_dist_per_iter = kwargs.get("w_dist_per_iter", False)
        self.dim_particle = kwargs.get("dim_particle", 3)
        self.dim_global = kwargs.get("dim_global", 10)
        num_epic_layers_gen = kwargs.get("num_epic_layers_gen", 6)
        num_epic_layers_dis = kwargs.get("num_epic_layers_dis", 3)
        self.norm_sigma = kwargs.get("norm_sigma", 5)
        beta_1 = kwargs.get("beta_1", 0.9)



        self.dataset_name = dataset_name
        self.rng = rng

        if file_suffix is None:
            file_suffix = "main"
        self.file_suffix = file_suffix

        try:
            self.file_name_suffix = self.dataset_name + "_" + self.file_suffix
        except TypeError:
            print("file_suffix should be a string")
            sys.exit()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size

        self.real_label = 1.
        self.fake_label = 0.
        #load the dataset
        try:
            dataset = data_proc.get_dataset(self.dataset_name)
        except FileNotFoundError:
            print("the dataset you specified could not be found")
            sys.exit()

        self.n_points = dataset.shape[1]
        #split into sets according to splits = [0.7, 0.15, 0.15]
        train_set, self.val_set, self.test_set = data_proc.split_dataset(dataset, rng = self.rng)

        if center_jets:
            train_set = utils.center_jets(train_set)
            self.val_set = utils.center_jets(self.val_set)
            self.test_set = utils.center_jets(self.test_set)

        #get kde for this dataset
        try:
            self.kde = data_proc.get_kde(dataset_name)
        except FileNotFoundError:
            print("could not find KDE, will compute it myself")
            self.kde = data_proc.compute_kde(dataset_name)
            

        #get the properties needed for normalisation
        self.train_set_means, self.train_set_stds, self.train_set_mins, _ = data_proc.dataset_properties(train_set)

        #initialise the models
        self.generator = models.Generator(input_size_p = self.dim_particle,
                     input_size_g = self.dim_global, hid_size_p = 128, hid_size_g = self.dim_global,
                     hid_size_g_in = 128, num_epic_layers = num_epic_layers_gen)
        self.discriminator = models.Discriminator(input_size_p = self.dim_particle,
                     hid_size_p = 128, hid_size_g = self.dim_global, hid_size_g_in = 128, 
                     num_epic_layers = num_epic_layers_dis)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        #initialise optimisers
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr = 1e-4,
                                            betas = (beta_1, 0.999), eps = 1e-14)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr = 1e-4,
                                            betas = (beta_1, 0.999), eps = 1e-14)

        #put models in training mode
        self.generator.train()
        self.discriminator.train()


        #use custom class to prepare dataset
        self.dataset = data_proc.PreparedDataset(train_set, batch_size = self.batch_size, rng = self.rng)
        self.num_iter_per_ep = self.dataset.num_iter_per_ep()

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
        if self.w_dist_per_iter:
            self.w_dist_per_iter_list = []
            

        log_folder = "./logbooks"

        logfile_name = "logbook_" + self.file_name_suffix + ".log"

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



    def training(self, num_epochs, loss = "BCE", save_result_dict = False, **kwargs):
        """This function performs the actual training of the specified model.
        As optimizers for both generator and discriminator Adam is used. The 
        returns are contained in a dictionary.

        Arguments
        -----------

        num_epochs: int
            number of epochs for which to run training

        loss: str, default: "BCE"
            whether to use binary cross entropy ("BCE") or least-squares-GAN loss
            ("LS-GAN") for training; defaults to "BCE"

        save_result_dict: bool, default: False
            if True, dictionary containing results is stored to a .pkl-file with
            name dataset_name + "_training_" + file_suffix in folder saved_models. Differing 
            file_suffix can be specified with keyword 'save_file_suffix'. Differing folder
            can be specified with keyword 'dict_save_folder'.

        **save_file_suffix: str, default: file_suffix (specified in object initialisation)
            dictionary containing results will be stored to .pkl-file with name
            dataset_name + "_training_" + save_file_suffix
        
        **dict_save_folder: str, default: "saved_models"
            folder where to store the result dictionary

        **lr_gen: float, default: 1e-4
            sets the learning rate of the generator

        **lr_dis: float, default: 1e-4
            sets the learning rate of the discriminator

        **n_tot_generation: int, default: 300000
            number of samples generated for each validation step

        **runs: int, default: 10
            number of comparison runs for each validation step
            make sure n_tot_generation/runs is larger than the length of validation and test set

        **batch_size_gen: int, default: 500
            batch size at which noise samples are passed through the generator

        **set_min_pt: bool, default: True
            if True, sets p_t coordinates of generated events to minimum value found in training set

        **order_by_pt: bool, default: True
            if True, orders particles by p_t in validation loops

        **normalise_data: bool, default: True
            if True, normalises generated events in validation to mean & std of training set

        **center_gen: bool, default: True
            if True, centers the eta- and phi-coordinates of generated events in validation

        Returns
        -----------

        result_dict: dict
            dictionary with the following keys:
                best_w_distance: float
                    Wasserstein distance score on the test set for the best epoch

                best_epoch: int
                    epoch for which the Wasserstein score on the validation set was lowest

                mean_loss_dis_list: list
                    list of all the average loss values of the discriminator training

                mean_loss_gen_list: list
                    list of all the average loss values of the generator training

                w_dist_list: list
                    list of all the Wasserstein distance scores on the validation set

        """

        save_file_suffix = kwargs.get("save_file_suffix", self.file_suffix)

        #if a lr was specified; if not, both are 1e-4
        if "lr_gen" in kwargs: 
            for g in self.optimizer_g.param_groups:
                g['lr'] = kwargs["lr_gen"]

        if "lr_dis" in kwargs: 
            for g in self.optimizer_d.param_groups:
                g['lr'] = kwargs["lr_dis"]

        n_tot_generation = kwargs.get("n_tot_generation", 300000)
        runs = kwargs.get("runs", 10)
        batch_size_gen = kwargs.get("batch_size_gen", 500)
        set_min_pt = kwargs.get("set_min_pt", True)
        order_by_pt = kwargs.get("order_by_pt", True)
        normalise_data = kwargs.get("normalise_data", True)
        center_gen = kwargs.get("center_gen", True)



        self.logger.info("loss is %s", loss)

        #start the training loop
        start = time.time()
        iteration_counter = 0
        #set epoch counter to 0 explicitly for case of multiple trainings
        self.epoch_counter = 0

        #specify self.batch_size to None to ensure dataloader will employ __iter__ method
        #as defined in PreparedDataset
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = None)

        #use tqdm in order to display a progress bar
        tqdm_dataloader = tqdm.tqdm(dataloader, total = int(self.num_iter_per_ep*num_epochs - 1))
        #training loop
        for batch in tqdm_dataloader:
            iteration_counter += 1

            #validation loop
            if iteration_counter % self.num_iter_per_ep == 0:
                self.logger.info("Epoch %d done", int(self.epoch_counter + 1))

                self.loss_dis /= self.num_iter_per_ep
                self.loss_gen /= self.num_iter_per_ep
                self.mean_loss_dis_list.append(self.loss_dis)
                self.mean_loss_gen_list.append(self.loss_gen)
                self.logger.info("losses: Discriminator: %.3f; Generator: %.3f",
                                  self.loss_dis, self.loss_gen)
                self.loss_dis = 0
                self.loss_gen = 0

                self.validation_loop(n_tot_generation, runs, batch_size_gen, set_min_pt, 
                                     order_by_pt, normalise_data, center_gen)

                self.epoch_counter += 1

            if self.epoch_counter == num_epochs: #breaking condition
                self.logger.info("All %d epochs done, training finished", num_epochs)
                self.logger.info("Best epoch was epoch %d with a Wasserstein distance of %.5f", 
                                 self.best_epoch, self.test_w_distance)
                tqdm_dataloader.close()
                break

            if normalise_data:
                batch = data_proc.normalise_dataset(batch, self.train_set_means, self.train_set_stds, 
                                                    norm_sigma = self.norm_sigma)

            data = batch.to(self.device)
            #this might be smaller than self.batch_size
            
            #Discriminator training
            self.discriminator_training(data, loss = loss)

            #generator training
            local_batch_size = data.size(0)
            self.generator_training(local_batch_size, loss = loss)

        total_time = time.time() - start
        hours, rest = divmod(total_time, 3600)
        mins, secs  = divmod(rest, 60)
        self.logger.info("In total the training took %d h %d min %d sec", hours, mins, secs)

        results = {}
        results["best_w_distance"]    = self.test_w_distance
        results["best_epoch"]         = self.best_epoch
        results["mean_loss_dis_list"] = self.mean_loss_dis_list
        results["mean_loss_gen_list"] = self.mean_loss_gen_list
        results["w_dist_list"]        = self.w_dist_list 

        if self.w_dist_per_iter:
            results["w_dist_per_iter"] = self.w_dist_per_iter_list   

        if save_result_dict:
            folder = "saved_models"
            path = os.path.join(folder, self.dataset_name + "_training_" + save_file_suffix + ".pkl")
            with open(path, "wb") as f:
                pickle.dump(results, f)
                f.close()

        return results




    def validation_loop(self, n_tot_generation, runs, batch_size_gen, set_min_pt, order_by_pt, 
                        normalise_data, center_gen):
        """Performs a single validation loop, i.e.creates n_tot_generation fake events and computes
        the Wasserstein distance between the mass distributions of the validation set and an equal
        amount of fake events 'runs' times (for different fake samples). The validation score is the
        mean over the values computed. If it is lower than the previously lowest validation score, 
        the epoch is accepted as the new best epoch and the same computation is repeated using the
        test instead of the validation set. The resulting score is stored.


        Arguments
        -----------

        n_tot_generation: int
            number of samples generated for the validation step

        runs: int
            number of comparison runs for the validation step
            make sure n_tot_generation/runs is larger than the length of validation and test set

        batch_size_gen: int
            batch size at which noise samples are passed through the generator

        set_min_pt: bool
            if True, sets p_t coordinates of generated events to minimum value found in training set

        order_by_pt: bool, default: True
            if True, orders particles by p_t

        normalise_data: bool, default: True
            if True, normalises generated events in validation to mean & std of training set

        center_gen: bool, default: True
            if True, centers the eta- and phi-coordinates of generated events in validation
        """

        w_distance = evaluation.compute_wasserstein_distance(self.generator, self.val_set, self.kde,
                        batch_size_gen = batch_size_gen, n_tot_generation = n_tot_generation,
                        dim_global = self.dim_global, dim_particle = self.dim_particle,
                        rng = self.rng, set_min_pt = set_min_pt, min_pt = self.train_set_mins[0],
                        center_gen = center_gen, order_by_pt = order_by_pt,
                        normalise_data = normalise_data,
                        means = self.train_set_means, stds = self.train_set_stds,
                        norm_sigma = self.norm_sigma, runs = runs, device = self.device)
        self.w_dist_list.append(w_distance)

        self.logger.info("Wasserstein-distance for epoch %d is %.5f", int(self.epoch_counter+1), w_distance)

        #
        if self.epoch_counter == 0: #initialise running variables after first epoch
            self.best_w_dist = w_distance
            self.best_epoch = self.epoch_counter + 1
            #get Wasserstein distance for the test set
            self.test_w_distance = evaluation.compute_wasserstein_distance(self.generator, 
                            self.test_set, self.kde, batch_size_gen = batch_size_gen, 
                            n_tot_generation = n_tot_generation, dim_global = self.dim_global, 
                            dim_particle = self.dim_particle, rng = self.rng, set_min_pt = set_min_pt, 
                            min_pt = self.train_set_mins[0], center_gen = center_gen, 
                            order_by_pt = order_by_pt, normalise_data = normalise_data,
                            means = self.train_set_means, stds = self.train_set_stds,
                            norm_sigma = self.norm_sigma, runs = runs, device = self.device)

            utils.save_model(self.generator, self.discriminator, self.optimizer_g, self.optimizer_d,
                            file_name = self.file_name_suffix)

            self.logger.info("first epoch done, model saved")
            self.logger.info("Its Wasserstein distance on the test set is %.5f", self.test_w_distance)

        else: #from second epoch on, do this
            if w_distance < self.best_w_dist: # -> better model found
                self.best_w_dist = w_distance
                self.best_epoch = self.epoch_counter + 1

                self.test_w_distance = evaluation.compute_wasserstein_distance(self.generator, 
                                self.test_set, self.kde, batch_size_gen = batch_size_gen, 
                                n_tot_generation = n_tot_generation, dim_global = self.dim_global, 
                                dim_particle = self.dim_particle, rng = self.rng, set_min_pt = set_min_pt, 
                                min_pt = self.train_set_mins[0], center_gen = center_gen, 
                                order_by_pt = order_by_pt, normalise_data = normalise_data,
                                means = self.train_set_means, stds = self.train_set_stds,
                                norm_sigma = self.norm_sigma, runs = runs, device = self.device)
                


                utils.save_model(self.generator, self.discriminator, self.optimizer_g, 
                                self.optimizer_d, file_name = self.file_name_suffix)

                self.logger.info("Better model found and saved after epoch %i", 
                                 int(self.epoch_counter+1))
                self.logger.info("Its Wasserstein distance on the test set is %.5f", 
                                 self.test_w_distance)




    def discriminator_training(self, data, loss = "BCE"):
        """Performs the discriminator training.

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

        local_batch_size = int(data.size(0))

        self.discriminator.train()
        self.generator.eval()

        self.optimizer_d.zero_grad()
        self.discriminator.zero_grad()

        noise_global, noise_particle = data_proc.get_noise(self.n_points, batch_size = local_batch_size,
                                        dim_global = self.dim_global, dim_particle = self.dim_particle,
                                        rng = self.rng, device = self.device)

        gen_out = self.generator(noise_particle, noise_global)

        #outputs of the Discriminator
        discr_out_real = self.discriminator(data)
        discr_out_fake = self.discriminator(gen_out)

        #loss is the least-squares-GAN loss
        if loss == "LS-GAN":
            discr_loss = 0.5 * (torch.mean(torch.square(discr_out_real - self.real_label)) + torch.mean(torch.square(discr_out_fake - self.fake_label)))
        elif loss == "BCE":
            loss = torch.nn.BCEWithLogitsLoss()
            real_label_tensor = torch.full((local_batch_size,1), self.real_label, 
                                           dtype = torch.float, device = self.device)
            fake_label_tensor = torch.full((local_batch_size,1), self.fake_label, 
                                           dtype = torch.float, device = self.device)
            
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

        if self.w_dist_per_iter:
            data = data.detach().cpu().numpy()
            gen_out = gen_out.detach().cpu().numpy()

            
            data = data_proc.inverse_normalise_dataset(data, self.train_set_means, self.train_set_stds, 
                                                       norm_sigma = self.norm_sigma)
            gen_out = data_proc.inverse_normalise_dataset(gen_out, self.train_set_means, self.train_set_stds, 
                                                          norm_sigma = self.norm_sigma)
            data = utils.order_array_pt(data)
            gen_out = utils.order_array_pt(gen_out)
            gen_out = data_proc.set_min_pt(gen_out, self.train_set_mins[0])
            gen_out = utils.center_jets(gen_out)

            w_dist = performance_metrics.wasserstein_mass(data, gen_out, num_samples = data.shape[0], 
                                                        runs = 1, return_std = False, rng = self.rng)
            self.w_dist_per_iter_list.append(w_dist)



    def generator_training(self, local_batch_size, loss = "BCE"):
        """Performs the generator training

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
                                        dim_global = self.dim_global, dim_particle = self.dim_particle,
                                        rng = self.rng, device = self.device)

        gen_out = self.generator(noise_particle, noise_global)
        
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


    def evaluate(self, make_plots = True, save_plots = True, save_result_dict = False, **kwargs):
        """Computes the evaluation scores and optionally the evaluation plots that are given in 
        the EPiC-GAN paper.\n
        WARNING: This method evaluates the current state of the object, which does not necessarily 
        coincide with the best model found in training. If you want to evaluate the best model 
        found, load that model either using function evaluate_performance.evaluate_performance() or 
        instantiate a new object of class TrainableModel setting 'load' to True and specifying a 
        model to load.

        Arguments
        -------------

        make_plots: bool, default: True
            if True, evaluation plots are made

        save_plots: bool, default: True
            if True, evaluation plots are saved to .png-file save_file_name in folder "saved_plots"

        save_result_dict: bool  default: False
            if True, result dictionary is saved to .pkl-file named "eval_scores" + save_file_name
            in folder dict_save_folder (can be specified as kwarg)

        **batch_size_gen: int, default: 500
            batch size at which noise samples are passed through the generator

        **n_tot_generation: int, default: 300000
            number of samples generated for each validation step

        **runs: int, default: 10
            number of comparison runs for each validation step
            make sure n_tot_generation/runs is larger than the length of validation and test set

        **set_min_pt: bool, default: True
            if True, sets p_t coordinates of generated events to minimum value found in training set

        **order_by_pt: bool, default: True
            if True, orders particles by p_t in validation loops

        **normalise_data: bool, default: True
            if True, normalises generated events in validation to mean & std of training set

        **center_gen: bool, default: True
            if True, centers the eta- and phi-coordinates of generated events in validation

        **dict_save_folder: str, default: "saved_models"
            folder where to store the result dictionary

        **name_plots: str, default: dataset_name (specified in initialisation)
            label that will appear in the plots

        **save_file_name: str, default: dataset_name + file_suffix (both specified in initialisation)
            file name to which result dictionary and/or plots will be stored

        Returns
        ---------

        result_dict: dict
            dictionary containing the evaluation scores, keys: "w_mass_mean", "w_mass_std", 
            "w_coords_mean", "w_coords_std", "w_efps_mean", "w_efps_std"
            they refer to mean and standard deviation of the Wasserstein distances between real and
            generated events using the mass distribution, the particle feature distributions p_t,
            eta, phi (mean over those) and the distributions of the energyflow polynomials (mean
            over those, see https://energyflow.network/docs/efp/)

        fig: matplotlib.figure, optional
            the evaluation plots

        """

        if save_plots and not make_plots:
            self.logger.warning("""save_plots was detected True, although no plots are made; if you
                                want to make plots and save them, set make_plots to True""")

        batch_size_gen = kwargs.get("batch_size_gen", 500)
        n_tot_generation = kwargs.get("n_tot_generation", 300000)
        runs = kwargs.get("runs", 10)
        set_min_pt = kwargs.get("set_min_pt", True)
        order_by_pt = kwargs.get("order_by_pt", True)
        normalise_data = kwargs.get("normalise_data", True)
        center_gen = kwargs.get("center_gen", True)
        dict_save_folder = kwargs.get("dict_save_folder", "saved_models")
        name_plots = kwargs.get("name_plots", self.dataset_name)
        save_file_name = kwargs.get("save_file_name", self.file_name_suffix)


        generated_events = evaluation.generation_loop(self.generator, self.n_points, self.kde,
                        batch_size = batch_size_gen, n_tot_generation = n_tot_generation,
                        dim_global = self.dim_global, dim_particle = self.dim_particle, rng = self.rng,
                        order_by_pt = order_by_pt, set_min_pt = set_min_pt,
                        min_pt = self.train_set_mins[0], center_gen = center_gen,
                        normalise_data = normalise_data, means = self.train_set_means,
                        stds = self.train_set_stds, norm_sigma = self.norm_sigma, device = self.device)
        
        if order_by_pt:
            #is performed already in generation_loop for fake samples
            self.test_set = utils.order_array_pt(self.test_set)
        
        if make_plots:
            result_dict, fig = evaluate_performance.evaluation_scores_plots(self.test_set, 
                                generated_events, runs, make_plots = make_plots, name_plots = name_plots, 
                                save_plots = save_plots, save_result_dict = save_result_dict, 
                                save_file_name = save_file_name, rng = self.rng, 
                                dict_save_folder = dict_save_folder)
        
            return result_dict, fig
        
        result_dict = evaluate_performance.evaluation_scores_plots(self.test_set, generated_events, 
                            runs, make_plots = make_plots, save_result_dict = save_result_dict, 
                            save_file_name = save_file_name, rng = self.rng, dict_save_folder = dict_save_folder)
        
        return result_dict
    


    def generate(self, n_generation, **kwargs):
        """Generates n_generation events.\n
        WARNING: This method uses the current state of the object, which does not necessarily 
        coincide with the best model found in training. If you want to use the best model 
        found, instantiate a new object of class TrainableModel setting 'load' to True and 
        specifying a model to load or alternatively, use function generation.generate(). Make sure
        to specify all parameters that were differing from default in the training.

        Arguments
        ------------

        n_generation: int
            number of events to be generated

        **batch_size_gen: int, default: 500
            batch size at which noise samples are passed through the generator

        **set_min_pt: bool, default: True
            if True, sets p_t coordinates of generated events to minimum value found in training set
        
        **order_by_pt: bool, default: False
            if True, orders particles by p_t within jets

        **normalise_data: bool, default: True
            if True, normalises generated events to mean & std of training set; if False, output is
            direct generator output (approx. normalised to mean 0, std norm_sigma, defined in object
            instantiation, for each feature)

        **center_gen: bool, default: True
            if True, centers the eta- and phi-coordinates of generated events
        """

        batch_size_gen = kwargs.get("batch_size_gen", 500)
        set_min_pt = kwargs.get("set_min_pt", True)
        order_by_pt = kwargs.get("order_by_pt", False)
        normalise_data = kwargs.get("normalise_data", True)
        center_gen = kwargs.get("center_gen", True)

        n_tot_generation = int(1.01*n_generation)

        generated_events = evaluation.generation_loop(self.generator, self.n_points, self.kde, 
                    batch_size = batch_size_gen, n_tot_generation = n_tot_generation,
                    dim_global = self.dim_global, dim_particle = self.dim_particle, rng = self.rng,
                    order_by_pt = order_by_pt, set_min_pt = set_min_pt, 
                    min_pt = self.train_set_mins[0], center_gen = center_gen, 
                    normalise_data = normalise_data, means = self.train_set_means, 
                    stds = self.train_set_stds, norm_sigma = self.norm_sigma, device = self.device)
        
        #if too many samples got killed in the kde sampling
        if len(generated_events) < n_generation:
            #this will always be sufficiently large s.t. n_generation events get past the kde
            #sampling
            n_tot_generation = int(1.05*n_generation)
            generated_events = evaluation.generation_loop(self.generator, self.n_points, self.kde, 
                    batch_size = batch_size_gen, n_tot_generation = n_tot_generation,
                    dim_global = self.dim_global, dim_particle = self.dim_particle, rng = self.rng,
                    order_by_pt = order_by_pt, set_min_pt = set_min_pt, 
                    min_pt = self.train_set_mins[0], center_gen = center_gen, 
                    normalise_data = normalise_data, means = self.train_set_means, 
                    stds = self.train_set_stds, norm_sigma = self.norm_sigma, device = self.device)
            
        generated_events = generated_events[:n_generation]

        return generated_events




