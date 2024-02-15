"""This script evaluates the performance of a given network.
"""

import os
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from epicgan import utils, data_proc, models, evaluation, performance_metrics



#default values

#which dataset to use out of gluon30, quark30, top30, gluon150, quark150, top150
#dataset_name = "quark30"
#the number of points (particles) per jet; either 30 or 150
#n_points     = 30
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
runs         = 10
#number of epochs to be performed
num_epochs   = 20
#whether to set p_t coordinates of generated events to specified minimum value
set_min_pt   = True
#whether to normalise generated events to mean & std of training set
inv_normalise_data = True
#whether to order particles by p_t in validation loop
order_by_pt = True
#whether to center the eta- and phi-coordinates of generated events
center_gen   = True


def evaluate_performance(dataset_name, n_points, make_plots = True):
    """
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_folder = "./logbooks"
    logfile_name = "logbook_evaluation_" + dataset_name + ".log"

    logging.basicConfig(format = '%(asctime)s[%(levelname)s] %(funcName)s: %(message)s',
                              datefmt = '%d/%m/%Y %I:%M:%S %p',
                              filename = os.path.join(log_folder, logfile_name),
                              level = logging.DEBUG,
                              filemode = 'w')

    logger = logging.getLogger("main")

    #load the required model
    generator = models.Generator(n_points, input_size_p = dim_particle,
                 input_size_g = dim_global, hid_size_p = 128, hid_size_g = 10,
                 hid_size_g_in = 128, num_epic_layers = num_epic_layers_gen)
    discriminator = models.Discriminator(n_points, input_size_p = dim_particle,
                 input_size_g = dim_global, hid_size_p = 128, hid_size_g = 10,
                 hid_size_g_in = 128, num_epic_layers = num_epic_layers_dis)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr = lr_G,
                                        betas = (beta_1, 0.999), eps = 1e-14)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = lr_D,
                                        betas = (beta_1, 0.999), eps = 1e-14)


    generator, discriminator, optimizer_g, optimizer_d = utils.load_model(generator, discriminator, optimizer_g, optimizer_d, file_name = dataset_name, device = device)

    generator.eval()
    discriminator.eval()


    #get the dataset
    dataset = data_proc.get_dataset(dataset_name)
    train_set, val_set, test_set = data_proc.split_dataset(dataset, rng = rng)

    kde = data_proc.get_kde(dataset_name)
    train_set_means, train_set_stds, train_set_mins, train_set_maxs = data_proc.dataset_properties(train_set)


    generated_events = evaluation.generation_loop(generator, n_points, kde, batch_size = batch_size,
                        n_tot_generation = n_tot_generation, dim_global = dim_global,
                        dim_particle = dim_particle, rng = rng, order_by_pt = order_by_pt,
                        set_min_pt = set_min_pt, min_pt = train_set_mins[0], center_gen = center_gen,
                        inv_normalise_data = inv_normalise_data, inv_means = train_set_means,
                        inv_stds = train_set_stds, inv_norm_sigma = norm_sigma, device = device)

    if order_by_pt:
        #is performed in generation_loop for fake samples
        test_set = utils.order_array_pt(test_set)


    len_test_set = len(test_set)

    w_mass_mean, w_mass_std = performance_metrics.wasserstein_mass(test_set,
                                generated_events, num_samples = len_test_set, num_batches = 10,
                                return_std = True, rng = rng)

    w_coords_mean, w_coords_std = performance_metrics.wasserstein_coords(test_set,
                    generated_events, exclude_zeros = True, num_samples = len_test_set,
                    num_batches = 10, avg_over_features = True, return_std = True,
                    rng = rng)

    w_efp_mean, w_efp_std = performance_metrics.wasserstein_efps(test_set,
                    generated_events, num_samples = len_test_set, num_batches = 10,
                    avg_over_efps = True, return_std = True, rng = rng)

    if n_points == 30:
        fpnd_mean, fpnd_std = performance_metrics.fpnd_score(generated_events, dataname = dataset_name,
                            num_samples = len_test_set, num_batches = 3, return_std = True)

    logger.info("Evaluation done")

    logger.info("""W_mass: %.5f; std %.5f\n
                   W_coords: %.5f; std %.5f\n
                   W_efp: %.5f; std %.5f\n
                """, w_mass_mean, w_mass_std, w_coords_mean, w_coords_std, w_efp_mean, w_efp_std)

    if n_points == 30:
        logger.info("FPND: %.5f; std %.5f", fpnd_mean, fpnd_std)

    result_dict = {}
    result_dict["w_mass_mean"] = w_mass_mean
    result_dict["w_mass_std"]  = w_mass_std
    result_dict["w_coords_mean"] = w_coords_mean
    result_dict["w_coords_std"]  = w_coords_std
    result_dict["w_efp_mean"] = w_efp_mean
    result_dict["w_efp_std"]  = w_efp_std

    if n_points == 30:
        result_dict["fpnd_mean"] = fpnd_mean
        result_dict["fpnd_std"]  = fpnd_std

    if make_plots:
        fig = plot_overview(generator, n_points, test_set, dataset_name,
                            generated_events = generated_events, logger = logger)

        return result_dict, fig

    return result_dict






##############  got this one from EPiC-Gan Github  #################
def plot_overview(generator, n_points, test_set, dataset_name, generated_events = None,
                    kde = None, batch_size = 128, n_tot_generation = 300000,
                    dim_global = 10, dim_particle = 3, rng = None, order_by_pt = True,
                    set_min_pt = True, min_pt = 0, center_gen = True,
                    inv_normalise_data = True, inv_means = np.zeros(3), inv_stds = np.ones(3),
                    inv_norm_sigma = 1, device = "cuda", logger = None):
    """
    """

    if generated_events is None:
        if kde is None:
            if logger is not None:
                logger.warning("""if you do not specify a set of generated events,
                               please specify a kde for the generation; this function
                               will return None""")
            else:
                print("plot_overview got neither generated_events nor kde for generation, returns None")
            return None

        gen_ary = evaluation.generation_loop(generator, n_points, kde, batch_size = batch_size, n_tot_generation = n_tot_generation,
                            dim_global = dim_global, dim_particle = dim_particle, rng = rng, order_by_pt = order_by_pt,
                            set_min_pt = set_min_pt, min_pt = min_pt, center_gen = center_gen,
                            inv_normalise_data = inv_normalise_data, inv_means = inv_means, inv_stds = inv_stds,
                            inv_norm_sigma = inv_norm_sigma, device = device)
        gen_ary = gen_ary[:len(test_set)]

        if order_by_pt:
            data_ary = utils.order_array_pt(test_set)

    else:
        data_ary = test_set
        if order_by_pt:
            data_ary = utils.order_array_pt(data_ary)

        gen_ary  = generated_events[:len(test_set)]


    data_test_ary = data_ary
    fake_test_ary = gen_ary

    # data variables
    data_ms = utils.jet_masses(data_test_ary)
    data_pts = utils.jet_pts(data_test_ary)
    data_mults = utils.jet_multiplicities(data_test_ary)
    # fake variables
    fake_ms = utils.jet_masses(fake_test_ary)
    fake_pts = utils.jet_pts(fake_test_ary)
    fake_mults = utils.jet_multiplicities(fake_test_ary)
    # p4s
    data_test_ary_p4s = utils.torch_p4s_from_ptyphi(torch.tensor(data_test_ary)).numpy()
    fake_test_ary_p4s = utils.torch_p4s_from_ptyphi(torch.tensor(fake_test_ary)).numpy()


    # overview plots
    density = 0
    #n_points_cut = data_test_ary.shape[1]
    generator_name = "EPiC-GAN"
    #dataset_name = 'JetNet'+str(n_points_cut)
    color_list = ['grey', 'crimson', 'royalblue']
    fig = plt.figure(figsize=(18, 16), facecolor='white')
    gs = GridSpec(3,3)


    # particle pt
    ax = fig.add_subplot(gs[0])
    x_min, x_max = np.array([data_test_ary[:,:,0].flatten().min(), fake_test_ary[:,:,0].flatten().min()]).min(), np.array([data_test_ary[:,:,0].flatten().max(), fake_test_ary[:,:,0].flatten().max()]).max()
    x_min, x_max = 0, 1.
    hist1 = ax.hist(data_test_ary[:,:,0][data_test_ary[:,:,0] != 0].flatten(), bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])#, range=[-100,600])
    hist2 = ax.hist(fake_test_ary[:,:,0][fake_test_ary[:,:,0] != 0].flatten(), bins=100, label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])#, range=[-100,600])
    ax.legend(loc='upper right', fontsize=28,edgecolor='none')
    ax.set_xlabel(r'relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
    ax.set_ylabel('particles', fontsize=24)
    ax.set_yscale('log')
    ax.tick_params(labelsize=20)
    #plt.xscale('log')

    # particle rap
    ax = fig.add_subplot(gs[1])
    x_min, x_max = np.array([data_test_ary[:,:,1].flatten().min(), fake_test_ary[:,:,1].flatten().min()]).min(), np.array([data_test_ary[:,:,1].flatten().max(), fake_test_ary[:,:,1].flatten().max()]).max()
    x_min, x_max = -1.6,1.2
    hist1 = ax.hist(data_test_ary[:,:,1][data_test_ary[:,:,1] != 0].flatten(), bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])
    hist2 = ax.hist(fake_test_ary[:,:,1][fake_test_ary[:,:,1] != 0].flatten(), bins=hist1[1], label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])
    ax.set_xlabel(r'particle pseudorapidity $\eta^\mathrm{rel}$', fontsize=24)
    ax.set_ylabel('particles', fontsize=24)
    ax.set_yscale('log')
    ax.set_xticks(np.linspace(x_min, x_max, 5))
    ax.tick_params(labelsize=20)

    # particle phi
    ax = fig.add_subplot(gs[2])
    x_min, x_max = np.array([data_test_ary[:,:,2].flatten().min(), fake_test_ary[:,:,2].flatten().min()]).min(), np.array([data_test_ary[:,:,2].flatten().max(), fake_test_ary[:,:,2].flatten().max()]).max()
    x_min, x_max = -.5,.5
    hist1 = ax.hist(data_test_ary[:,:,2][data_test_ary[:,:,2] != 0].flatten(), bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])
    hist2 = ax.hist(fake_test_ary[:,:,2][fake_test_ary[:,:,2] != 0].flatten(), bins=hist1[1], label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])
    ax.set_xlabel(r'particle angle $\phi^\mathrm{rel}$', fontsize=24)
    ax.set_ylabel('particles', fontsize=24)
    ax.set_yscale('log')
    ax.set_ylim(1,)
    ax.tick_params(labelsize=20)


    n_points = [0,4,19]
    axes = [3,4,5]
    for j in range(3):
        i = n_points[j]
        ax = fig.add_subplot(gs[axes[j]])
        x_min, x_max = np.array([data_test_ary[:,i,0].flatten().min(), fake_test_ary[:,i,0].flatten().min()]).min(), np.array([data_test_ary[:,i,0].flatten().max(), fake_test_ary[:,i,0].flatten().max()]).max()
        hist1 = ax.hist(data_test_ary[:,i,0][data_test_ary_p4s[:,i,0] != 0].flatten(), bins=100, label=dataset_name, histtype='stepfilled', alpha=0.5, density=density, range=[x_min,x_max], color=color_list[0])
        hist2 = ax.hist(fake_test_ary[:,i,0][fake_test_ary_p4s[:,i,0] != 0].flatten(), bins=100, label=generator_name, histtype='step', density=density, range=[x_min,x_max], lw=4, color=color_list[1])#, range=[-100,600])
        if i == 0:
            ax.set_xlabel('{}'.format(i+1)+r'$^\mathrm{st}$ relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
        elif i == 1:
            ax.set_xlabel('{}'.format(i+1)+r'$^\mathrm{nd}$ relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
        elif i == 2:
            ax.set_xlabel('{}'.format(i+1)+r'$^\mathrm{rd}$ relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
        else:
            ax.set_xlabel('{}'.format(i+1)+r'$^\mathrm{th}$ relative particle $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
        ax.set_ylabel('particles', fontsize=24)
        ax.set_yscale('log')
        ax.set_xticks(ax.get_xticks()[1:][::2])   # hide every second x_tick, starting from the second
        ax.tick_params(labelsize=20)


    # jet mults
    ax = fig.add_subplot(gs[6])
    x_min, x_max = np.array([data_mults.min(), fake_mults.min()]).min(), np.array([data_mults.max(), fake_mults.max()]).max()
    b=x_max-x_min+1
    hist1 = ax.hist(data_mults, bins=b, label=dataset_name, histtype='stepfilled', alpha=0.5, range=[x_min,x_max], color=color_list[0])
    hist2 = ax.hist(fake_mults, bins=b, label=generator_name, histtype='step', lw=4, range=[x_min,x_max], color=color_list[1])
    ax.set_xlabel('particle multiplicity', fontsize=24)
    ax.set_yscale('log')
    ax.set_ylabel('jets', fontsize=24)
    ax.tick_params(labelsize=20)

    # jet mass
    ax = fig.add_subplot(gs[7])
    x_min, x_max = np.array([data_ms.min(), fake_ms.min()]).min(), np.array([data_ms.max(), fake_ms.max()]).max()
    x_min, x_max = 0, 0.3
    hist1 = ax.hist(data_ms, bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])
    hist2 = ax.hist(fake_ms, bins=100, label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])
    ax.set_xlabel('relative jet mass', fontsize=24)
    ax.set_ylabel('jets', fontsize=24)
    ax.set_yscale('log')
    ax.tick_params(labelsize=20)

    #jet pt
    ax = fig.add_subplot(gs[8])
    x_min, x_max = np.array([data_pts.min(), fake_pts.min()]).min(), np.array([data_pts.max(), fake_pts.max()]).max()
    x_min, x_max = 0.6, 1.2
    hist1 = ax.hist(data_pts, bins=100, label=dataset_name, histtype='stepfilled', range=[x_min,x_max], alpha=0.5, color=color_list[0])#, range=[-10,100])
    hist2 = ax.hist(fake_pts, bins=100, label=generator_name, histtype='step', range=[x_min,x_max], lw=4, color=color_list[1])
    ax.set_xlabel(r'relative jet $p_\mathrm{T}^\mathrm{rel}$', fontsize=24)
    ax.set_ylabel('jets', fontsize=24)
    ax.set_ylim(1,)
    ax.set_yscale('log')
    ax.tick_params(labelsize=20)

    plt.tight_layout()

    return fig
