"""This script defines a function that evaluates performance of a trained network.
Make sure that the hard-coded parameters that define the network architecture match
the model you want to load.
Due to the sheer amount of variables the ones that are usually not changed
are implemented as kwargs.
"""

import os
import sys
import logging
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from epicgan import utils, data_proc, models, evaluation, performance_metrics





def evaluate_performance(dataset_name, model_name, n_points, make_plots = True, save_plots = True, save_result_dict = False, save_file_name = None, rng = None, **kwargs):
    """Function that evaluates a (trained) network. Has an option to make the plots
    that are in the original EPiC-GAN paper and save them to a .png-file.
    When running the evaluation, ensure the place from where you run the training
    has the following folders and contents:
    1. The specified dataset is stored in folder 'JetNet_datasets' in '.hdf5'-format
    2. There is a folder 'saved_models' where the model you want to load is stored
    3. There is a folder 'logbooks', the logfile will be stored here

    Arguments:
    -----------

    dataset_name:
        specifies the dataset

    model_name: str
        model specification; must exist in folder 'saved_models'; will also be
        added as suffix to the logfile-name

    n_points: int
        number of particles per jet, either 30 or 150

    make_plots: bool, default: True
        if True, the plots that are in the original EPiC-GAN paper will be made

    save_plots: bool, default: True
        if True, plots will be saved to folder saved_plots with name save_file_name
        as a .png-file.

    save_result_dict: bool, default: False
        if True, dictionary containing results is stored to a .pkl-file with
        name dataset_name + "_evaluation_" + save_file_name in folder saved_models.

    save_file_name: str, default: None
        filename of the plots that will be saved, if make_plots is True;
        if None, plots will not be saved

    rng: np.random.Generator, default: None
        random number generator used for shuffling data; if equal to None, no
        shuffling will be performed

    **model_folder: str, default: "saved_models"
        folder, where the model to be loaded is stored

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

    **n_tot_generation: int, default: 300000
        number of samples generated for computing each evaluation score

    **runs: int, default: 10
        number of comparison runs for each evaluation score
        make sure n_tot_generation/runs is larger than the length of the test set

    **batch_size_gen: int, default: 500
        batch size at which noise samples are passed through the generator

    **set_min_pt: bool, default: True
        If True, sets p_t coordinates of generated events to minimum value found in training set.

    **order_by_pt: bool, default: True
        If True, orders particles by p_t in generated events and test data.

    **inv_normalise_data: bool, default: True
        If True, normalises generated events to mean & std of training set

    **center_gen: bool, default: True
        If True, centers the eta- and phi-coordinates of generated events


    Returns
    ----------

    result_dict: dict
    """

    model_folder = kwargs.get("model_folder", "saved_models")

    dim_particle = kwargs.get("dim_particle", 3)

    dim_global = kwargs.get("dim_global", 10)

    num_epic_layers_gen = kwargs.get("num_epic_layers_gen", 6)

    num_epic_layers_dis = kwargs.get("num_epic_layers_dis", 3)

    norm_sigma = kwargs.get("norm_sigma", 5)

    n_tot_generation = kwargs.get("n_tot_generation", 300000)

    runs = kwargs.get("runs", 10)

    batch_size_gen = kwargs.get("batch_size_gen", 500)

    set_min_pt = kwargs.get("set_min_pt", True)

    order_by_pt = kwargs.get("order_by_pt", True)

    inv_normalise_data = kwargs.get("inv_normalise_data", True)

    center_gen = kwargs.get("center_gen", True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_folder = "./logbooks"
    try:
        logfile_name = "logbook_evaluation_" + model_name + ".log"
    except TypeError:
        print("model_name must be a string")
        sys.exit()

    logging.basicConfig(format = '%(asctime)s[%(levelname)s] %(funcName)s: %(message)s',
                              datefmt = '%d/%m/%Y %I:%M:%S %p',
                              filename = os.path.join(log_folder, logfile_name),
                              level = logging.INFO,
                              filemode = 'w')

    logger = logging.getLogger("main")

    #load the required model
    generator = models.Generator(input_size_p = dim_particle,
                 input_size_g = dim_global, hid_size_p = 128, hid_size_g = dim_global,
                 hid_size_g_in = 128, num_epic_layers = num_epic_layers_gen)
    discriminator = models.Discriminator(input_size_p = dim_particle,
                 hid_size_p = 128, hid_size_g = dim_global, hid_size_g_in = 128, 
                 num_epic_layers = num_epic_layers_dis)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_g = torch.optim.Adam(generator.parameters())
    optimizer_d = torch.optim.Adam(discriminator.parameters())

    try:
        generator, discriminator, optimizer_g, optimizer_d = utils.load_model(generator, discriminator, optimizer_g, optimizer_d, file_name = model_name, 
                                                                              folder = model_folder, device = device)
    except FileNotFoundError as e:
        logger.exception(e)
        logger.critical("could not find a file named %s in saved_models", model_name)

        sys.exit()

    generator.eval()
    discriminator.eval()


    #get the dataset
    dataset = data_proc.get_dataset(dataset_name)
    train_set, _, test_set = data_proc.split_dataset(dataset, rng = rng)

    kde = data_proc.get_kde(dataset_name)
    train_set_means, train_set_stds, train_set_mins, _ = data_proc.dataset_properties(train_set)


    generated_events = evaluation.generation_loop(generator, n_points, kde, batch_size = batch_size_gen,
                        n_tot_generation = n_tot_generation, dim_global = dim_global,
                        dim_particle = dim_particle, rng = rng, order_by_pt = order_by_pt,
                        set_min_pt = set_min_pt, min_pt = train_set_mins[0], center_gen = center_gen,
                        inv_normalise_data = inv_normalise_data, inv_means = train_set_means,
                        inv_stds = train_set_stds, inv_norm_sigma = norm_sigma, device = device)

    if order_by_pt:
        #is performed already in generation_loop for fake samples
        test_set = utils.order_array_pt(test_set)


    len_test_set = len(test_set)

    w_mass_mean, w_mass_std = performance_metrics.wasserstein_mass(test_set,
                                generated_events, num_samples = len_test_set, runs = runs,
                                return_std = True, rng = rng)

    w_coords_mean, w_coords_std = performance_metrics.wasserstein_coords(test_set,
                    generated_events, exclude_zeros = True, num_samples = len_test_set,
                    runs = runs, avg_over_features = True, return_std = True,
                    rng = rng)

    w_efp_mean, w_efp_std = performance_metrics.wasserstein_efps(test_set,
                    generated_events, num_samples = len_test_set, runs = runs,
                    avg_over_efps = True, return_std = True, rng = rng)

    #commented lines calculate FPND score; Python version <= 3.10, torch-cluster required
    #if n_points == 30:
    #    fpnd_mean, fpnd_std = performance_metrics.fpnd_score(generated_events, dataname = dataset_name,
    #                        num_samples = len_test_set, num_batches = 3, return_std = True)

    logger.info("Evaluation done")

    logger.info("""W_mass: %.5f; std %.5f\n
                   W_coords: %.5f; std %.5f\n
                   W_efp: %.5f; std %.5f\n
                """, w_mass_mean, w_mass_std, w_coords_mean, w_coords_std, w_efp_mean, w_efp_std)

    #if n_points == 30:
    #    logger.info("FPND: %.5f; std %.5f", fpnd_mean, fpnd_std)

    result_dict = {}
    result_dict["w_mass_mean"] = w_mass_mean
    result_dict["w_mass_std"]  = w_mass_std
    result_dict["w_coords_mean"] = w_coords_mean
    result_dict["w_coords_std"]  = w_coords_std
    result_dict["w_efp_mean"] = w_efp_mean
    result_dict["w_efp_std"]  = w_efp_std

    #if n_points == 30:
    #    result_dict["fpnd_mean"] = fpnd_mean
    #    result_dict["fpnd_std"]  = fpnd_std

    if save_result_dict:
        folder = "saved_models"
        try:
            path = os.path.join(folder, dataset_name + "_evaluation_" + save_file_name + ".pkl")
        except TypeError as e:
            logger.exception(e)
            logger.warning("""please specify a string on where to save the results,
                            if you want to save them; result dictionary will be returned,
                            but not saved""")
        f = open(path, "wb")
        pickle.dump(result_dict, f)
        f.close()

    if make_plots:
        folder = "./saved_plots"
        fig = plot_overview(test_set, dataset_name, generated_events)

        if save_plots:
            try:
                plt.savefig(os.path.join(folder, save_file_name + ".png"))
            except TypeError as e:
                logger.exception(e)
                logger.warning("""please specify a string on where to save the plots,
                                if you want to save them; figure will be returned,
                                but not saved""")

        return result_dict, fig

    return result_dict






##############  got the code for the plots in this function from EPiC-GAN Github  #################
def plot_overview(test_set, dataset_name, generated_events = None, generator = None, n_points = None,
                    kde = None, batch_size_gen = 500, n_tot_generation = 300000,
                    dim_global = 10, dim_particle = 3, rng = None, order_by_pt = True,
                    set_min_pt = True, min_pt = 0, center_gen = True,
                    inv_normalise_data = True, inv_means = np.zeros(3), inv_stds = np.ones(3),
                    inv_norm_sigma = 1, device = "cuda"):
    """This function makes the 9 plots that are given for each dataset in the original EPiC-GAN paper.
    It does NOT save them.

    Arguments
    -------------

    test_set: np.array
        set of samples from original dataset to use for evaluation

    datset_name: str
        name used in labels of plots;

    generated_events: np.array, default: None
        generated events to use for evaluation; IF SPECIFIED, ALL OF THE FOLLOWING
        ARGUMENTS ARE REDUNDANT

    generator: epicgan.models.Generator
        generator network with which to perform the event generation; needs to be specified
        when generated_events is None

    n_points: int
        number of particles per jet, 30 or 150; needs to be specified when
        generated_events is None

    kde: scipy.stats.gaussian_kde
        kernel density estimation of n_eff for the dataset

    batch_size_gen: int, default: 500
        batch size at which noise samples are passed through the generator

    n_tot_generation: int, default: 300000
        number of points that will be sampled from kde

    dim_global: int, default: 10
        dimension of the space of global variables in the network

    dim_particle: int, default: 3
        dimension of the space of particle features

    rng: np.random.Generator, default: None
        random number generator used for kde-resampling; please note that
        resampling of kde needs a random number generator, it will default to
        np.random, if None is given

    order_by_pt: bool, default: True
        if True, the particles within each generated fake jet will be ordered by
        p_t in descending order

    set_min_pt: bool, default: True
        if True, all generated particles will be enforced to have p_t of at least
        the value specified in min_pt

    min_pt: int or float, default: 0
        minimum value to which all p_t below will be set, if set_min_pt is True

    center_gen: bool, default: True
        if True, the eta- and phi-coordinates of the generated events will be
        centered

    inv_normalise_data: bool, default: True
        if True, the generated events will be renormalised to have the statistical
        properties of the training set

    inv_means: list or np.array, default: np.zeros(3)
        mean value for each particle feature to which to renormalise the generated events

    inv_stds: list or np.array, default: np.ones(3)
        standard deviation value for each particle feature to which to renormalise
        the generated events

    inv_norm_sigma: float or int, default: 1
        std-value to which the real input data was normalised

    device: str, default: "cuda"
        device to which to send variables

    Returns
    ------------

    fig: matplotlib.figure
        figure containing the 9 plots
    """

    logger = logging.getLogger("main")

    if generated_events is None:
        if kde is None:
            logger.warning("""if you do not specify a set of generated events,
                               please specify a kde for the generation; this function
                               will return None""")
            return None

        gen_ary = evaluation.generation_loop(generator, n_points, kde, batch_size = batch_size_gen, n_tot_generation = n_tot_generation,
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
