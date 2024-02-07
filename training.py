"""This script performs the training of a model to be specified.
Due to the sheer amount of variables it defines global variables in order to
make the function calls more concise.
"""

import logging
import time
import tqdm

import numpy as np
import torch
import jetnet
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
#random number generator used throughout the script for shuffling
rng          = np.random.default_rng(1)
#used to normalise input data to this std
norm_sigma   = 5
#total number of events generated for each evaluation assessment
#n_tot_generation = 300000
n_tot_generation = 3000 #just for now
#number of comparison runs for each Wasserstein validation step
#make sure runs is <= the number of times the length of validation/test sets fit into n_tot_generation
runs = 10
#number of epochs to be performed
num_epochs = 3



logging.basicConfig(format='%(asctime)s[%(levelname)s] %(funcName)s: %(message)s',
                          datefmt='%d/%m/%Y %I:%M:%S %p',
                          filename='logbook.log',
                          level=logging.DEBUG,
                          filemode='w')


logger = logging.getLogger("main")





def training(dataset):
    """This function performs the training of a specified model.
    As optimizers for both generator and discriminator Adam is used.
    """


    setup_training()


    #start the training loop
    start = time.time()
    iteration_counter = 0


    #use tqdm in order to display a progress bar
    iterator = tqdm.tqdm(dataloader, total = int(num_iter_per_ep*num_epochs - 1))
    #training loop
    for batch in iterator:
        iteration_counter += 1

        #validation loop
        if iteration_counter % num_iter_per_ep == 0:
            logger.info("Epoch %i done", epoch_counter + 1)

            validation_loop()

            epoch_counter += 1
            if epoch_counter == num_epochs: #breaking condition
                logger.info("All %d epochs done, training finished", num_epochs)
                logger.info("Best epoch was epoch %d with a Wasserstein distance of %.2f", best_epoch, test_w_distance)
                iterator.close()
                break

        global data
        data = batch.to(device)

        #Discriminator training
        discriminator_training()

        #generator training
        generator_training()

    return test_w_distance, test_w_dists_p, test_fpnd, best_epoch




def setup_training():

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global real_label
    global fake_label
    real_label = 1
    fake_label = 0
    #load the dataset
    dataset = data_proc.get_dataset(dataset_name)
    #split into sets according to splits = [0.7, 0.15, 0.15]
    global train_set
    global val_set
    global test_set
    train_set, val_set, test_set = data_proc.split_dataset(dataset, rng = rng)

    #load the precomputed kde for this dataset
    global kde
    kde = data_proc.get_kde(dataset_name)

    #get the properties needed for normalisation
    global train_set_means
    global train_set_stds
    global train_set_mins
    global train_set_maxs
    train_set_means, train_set_stds, train_set_mins, train_set_maxs = data_proc.dataset_properties(train_set)


    #initialise the models
    global generator
    global discriminator
    generator = models.Generator(n_points, input_size_p = 3, input_size_g = 10,
                 hid_size_p = 128, hid_size_g = 10, hid_size_g_in = 128,
                 num_epic_layers = 6)
    discriminator = models.Discriminator(n_points, input_size_p = 3, input_size_g = 10,
                 hid_size_p = 128, hid_size_g = 10, hid_size_g_in = 128,
                 num_epic_layers = 3)

    #initialise optimisers
    global optimizer_G
    global optimizer_D
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr_G,
                                        betas = (beta_1, 0.999), eps = 1e-14)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr_D,
                                        betas = (beta_1, 0.999), eps = 1e-14)

    #put models in training mode
    generator.train()
    discriminator.train()

    #normalise the training set
    train_set = data_proc.normalise_dataset(train_set, train_set_means, train_set_stds, norm_sigma = norm_sigma)

    #use custom class to prepare dataset
    dataset = data_proc.PreparedDataset(train_set, batch_size = batch_size, rng = rng)
    global num_iter_per_ep
    num_iter_per_ep = dataset.num_iter_per_ep()

    #specify batch_size to None to ensure dataloader will employ __iter__ method
    #as defined in PreparedDataset
    global dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = None)

    global epoch_counter
    epoch_counter = 0

    logger.info("used device is %s", device)
    logger.info("Model and data initialised")
    logger.info("Training will take %d iterations per epoch", num_iter_per_ep)


    global data

    #variables that will be assigned later in validation loop
    global best_w_dist
    global best_epoch
    global test_w_distance
    global test_w_dists_p
    global test_fpnd


def validation_loop():

    w_distance = evaluation.compute_wasserstein_distance(generator, val_set, kde,
                    batch_size = batch_size, n_tot_generation = n_tot_generation,
                    dim_global = dim_global, dim_particle = dim_particle,
                    rng = rng, set_min_pt = True, min_pt = train_set_mins[0],
                    runs = runs, device = device)

    #
    if epoch_counter == 0: #initialise running variables after first epoch
        best_w_dist = w_distance
        best_epoch = epoch_counter + 1
        #get Wasserstein distance for the test set
        test_w_distance = evaluation.compute_wasserstein_distance(generator, test_set, kde,
                        batch_size = batch_size, n_tot_generation = n_tot_generation,
                        dim_global = dim_global, dim_particle = dim_particle,
                        rng = rng, set_min_pt = True, min_pt = train_set_mins[0],
                        runs = runs, device = device)

        test_w_dists_p, test_fpnd = evaluation.evaluation_means(generator,
                            test_set, kde, calc_fpnd = True, calc_w_dist_p = True,
                            dataname = dataset_name, batch_size = batch_size,
                            n_tot_generation = n_tot_generation,
                            dim_global = dim_global, dim_particle = dim_particle,
                            rng = rng, set_min_pt = True, min_pt = train_set_mins[0],
                            runs = runs, device = device)

        utils.save_model(generator, discriminator, optimizer_G, optimizer_D,
                        file_name = dataset_name)
        logger.info("first epoch done, model saved")
        logger.info("Wasserstein distance is %.2f", test_w_distance)

    else: #from second epoch on, do this
        if w_distance < best_w_dist: # -> better model found
            best_w_dist = w_distance
            best_epoch = epoch_counter + 1

            test_w_distance = evaluation.compute_wasserstein_distance(generator, test_set, kde,
                            batch_size = batch_size, n_tot_generation = n_tot_generation,
                            dim_global = dim_global, dim_particle = dim_particle,
                            rng = rng, set_min_pt = True, min_pt = train_set_mins[0],
                            runs = runs, device = device)

            test_w_dists_p, test_fpnd = evaluation.evaluation_means(generator,
                                test_set, kde, calc_fpnd = True, calc_w_dist_p = True,
                                dataname = dataset_name, batch_size = batch_size,
                                n_tot_generation = n_tot_generation,
                                dim_global = dim_global, dim_particle = dim_particle,
                                rng = rng, set_min_pt = True, min_pt = train_set_mins[0],
                                runs = runs, device = device)

            utils.save_model(generator, discriminator, optimizer_G, optimizer_D,
                            file_name = dataset_name)

            logger.info("Better model found and saved after %i epochs", int(epoch_counter+1))
            logger.info("Its Wasserstein distance is %.2f", test_w_distance)




def discriminator_training():

    discriminator.train()
    generator.eval()

    discriminator.zero_grad()
    optimizer_D.zero_grad()

    noise_global, noise_particle = data_proc.get_noise(n_points, batch_size = batch_size,
                                    dim_global = dim_global, dim_particle = dim_particle,
                                    rng = rng, device = device)

    gen_out = generator(noise_particle, noise_global)
    #normalise output
    gen_out = data_proc.normalise_dataset(gen_out, means = train_set_means,
                                        stds = train_set_stds, norm_sigma = norm_sigma)
    #outputs of the Discriminator
    discr_out_real = discriminator(data)
    discr_out_fake = discriminator(gen_out)

    #loss is the least-squares-GAN loss
    discr_loss = 0.5 * (torch.mean((discr_out_real - real_label)**2) + torch.mean((discr_out_fake - fake_label)**2))
    #compute gradients, perform update
    discr_loss.backward()
    optimizer_D.step()


def generator_training():

    discriminator.eval()
    generator.train()
    generator.zero_grad()
    optimizer_G.zero_grad()

    noise_global, noise_particle = data_proc.get_noise(n_points, batch_size = batch_size,
                                    dim_global = dim_global, dim_particle = dim_particle,
                                    rng = rng, device = device)

    gen_out = generator(noise_particle, noise_global)
    #normalise output
    gen_out = data_proc.normalise_dataset(gen_out, means = train_set_means,
                                                stds = train_set_stds, norm_sigma = norm_sigma)
    #output of discriminator
    discr_out = discriminator(gen_out)
    #loss: real_label, because generator wants to fool discriminator
    gen_loss = 0.5 * torch.mean((discr_out - real_label)**2)
    #gradient and update
    gen_loss.backward()
    optimizer_G.step()
