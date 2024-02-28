"""Defines a function that generates events.
"""

import sys
import os

import torch
from epicgan import utils, models, evaluation, data_proc


def generate_events(dataset_name, model_name, n_generation, n_points, rng = None, **kwargs):
    """Generate n_generation events that mimick specified dataset.\n
    For using this function, please make sure to carefully specify all
    kwargs that differed from default in the training.

    Arguments
    -----------

    dataset_name: str
        dataset that is supposed to be mimicked by generated events

    model_name: str
        specifies the model whose generator to be used for generation

    n_generation: int
        number of events to be generated

    n_points: int
        particle multiplicity, 30 or 150

    rng: np.random.Generator, default: None
        random number generator used for shuffling created events; if equal to None, no shuffling 
        will be performed

    **dim_particle: int, default: 3
        dimension of the particle space, default 3 are [p_t, eta, phi]

    **dim_global: int, default: 10
        dimension of the global variable space within the networks

    **num_epic_layers_gen: int, default: 6
        number of EPiC-layers in the generator

    **folder: str, default: "saved_models"
        folder where the model is stored

    **device: str, default: "cuda" if available, else "cpu"
        device where to load the model and generate

    **batch_size_gen: int, default: 500
        batch size at which noise samples are passed through the generator

    **norm_sigma: float, default: 5
        std that the generated data is assumed to mimick; should equal norm_sigma that was used
        when training the model

    **set_min_pt: bool, default: True
        if True, sets p_t coordinates of generated events to minimum value found in whole dataset
    
    **order_by_pt: bool, default: False
        if True, orders particles by p_t within jets

    **normalise_data: bool, default: True
        if True, normalises generated events to mean & std of whole dataset; if False, output is
        direct generator output (approx. normalised to mean 0, std norm_sigma used in training for 
        each feature)

    **center_gen: bool, default: True
        if True, centers the eta- and phi-coordinates of generated events
    """

    dim_particle = kwargs.get("dim_particle", 3)
    dim_global = kwargs.get("dim_global", 10)
    num_epic_layers_gen = kwargs.get("num_epic_layers_gen", 6)
    folder = kwargs.get("folder", "saved_models")

    batch_size_gen = kwargs.get("batch_size_gen", 500)
    norm_sigma = kwargs.get("norm_sigma", 5.)
    set_min_pt = kwargs.get("set_min_pt", True)
    order_by_pt = kwargs.get("order_by_pt", False)
    normalise_data = kwargs.get("normalise_data", True)
    center_gen = kwargs.get("center_gen", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = kwargs.get("device", device)

    generator = models.Generator(input_size_p = dim_particle, input_size_g = dim_global, 
                                 hid_size_p = 128, hid_size_g = dim_global, hid_size_g_in = 128, 
                                 num_epic_layers = num_epic_layers_gen)

    
    try:
        generator = utils.load_generator(generator, model_name, folder = folder, device = device)
    except FileNotFoundError:
        print("cannot the model you specified")
        sys.exit()

    try:
        kde = data_proc.get_kde(dataset_name)
    except FileNotFoundError:
        print("could not find KDE, will compute it myself")
        try:
            kde = data_proc.compute_kde(dataset_name)
        except FileNotFoundError:
            print("the dataset you specified could not be found")
            sys.exit()

    if normalise_data or set_min_pt:
        dataset = data_proc.get_dataset(dataset_name)
        means, stds, mins, _ = data_proc.dataset_properties(dataset)
    else:
        means, stds, mins = None, None, None


    n_tot_generation = int(1.01*n_generation)

    generated_events = evaluation.generation_loop(generator, n_points, kde, 
                batch_size = batch_size_gen, n_tot_generation = n_tot_generation,
                dim_global = dim_global, dim_particle = dim_particle, rng = rng,
                order_by_pt = order_by_pt, set_min_pt = set_min_pt, 
                min_pt = mins[0], center_gen = center_gen, 
                normalise_data = normalise_data, means = means, 
                stds = stds, norm_sigma = norm_sigma, device = device)
    
    #if too many samples got killed in the kde sampling
    if len(generated_events) < n_generation:
        #this will always be sufficiently large s.t. n_generation events get past the kde
        #sampling
        n_tot_generation = int(1.05*n_generation)
        generated_events = evaluation.generation_loop(generator, n_points, kde, 
                batch_size = batch_size_gen, n_tot_generation = n_tot_generation,
                dim_global = dim_global, dim_particle = dim_particle, rng = rng,
                order_by_pt = order_by_pt, set_min_pt = set_min_pt, 
                min_pt = mins[0], center_gen = center_gen, 
                normalise_data = normalise_data, means = means, 
                stds = stds, norm_sigma = norm_sigma, device = device)
        
    generated_events = generated_events[:n_generation]

    return generated_events



    
