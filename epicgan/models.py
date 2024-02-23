"""Here, we define the models used for training. 
The EPiC-GAN-layer is defined and then implemented into the generator and
discriminator architectures. The models are defined according to the
EPiC-GAN paper https://arxiv.org/abs/2301.08128
"""


import torch
from torch import nn
from torch.nn.functional import leaky_relu
from torch.nn.utils.parametrizations import weight_norm



class EpicGanLayer(nn.Module):
    """This class refers to a generic EPiC-GAN-layer as defined in the paper
    with variable input, output and hidden size.
    """

    def __init__(self, hid_size_p, hid_size_g, hid_size_g_in):
        """No default values given, because this constructor is only to be called from
        the constructors of the generator and discriminator.

        Arguments
        -----------

        hid_size_p: int
            hidden dimension of the particle space

        hid_size_g: int
            hidden dimension of the global space

        hid_size_g_in: in
            dimension of the hidden (aggregation) layer of the global space

        """
        super(EpicGanLayer, self).__init__()

        self.hid_size_p = hid_size_p
        self.hid_size_g = hid_size_g
        self.hid_size_g_in = hid_size_g_in    #latent dimension between the two layers of Phi_in_g

        #layers
        self.aggreg = weight_norm(nn.Linear(int(2*self.hid_size_p + self.hid_size_g),
                                  self.hid_size_g_in))
        self.g_out = weight_norm(nn.Linear(self.hid_size_g_in, self.hid_size_g))

        self.distrib = weight_norm(nn.Linear(self.hid_size_g + self.hid_size_p, self.hid_size_p))
        self.p_out = weight_norm(nn.Linear(self.hid_size_p, self.hid_size_p))

    def forward(self, p_data, g_data):
        """Forward pass.

        Arguments
        ----------

        p_data: torch.Tensor
            input data for particle space of shape [batch_size, n_eff, num_particle_features]

        g_data: torch.Tensor
            input data for global space of shape [batch_size, input_size_g]


        Returns
        ----------

        p_interm: torch.Tensor
            output data for particle space of shape [batch_size, n_eff, num_particle_features]

        g_interm: torch.Tensor
            output data for global space of shape [batch_size, input_size_g]
        """
        n_eff = int(p_data.shape[1])

        #do not overwrite p_data or g_data, they are needed for residual connections
        p_mean = torch.mean(p_data, 1)  #calculates mean over n_eff
        p_sum = torch.sum(p_data, 1)    #calculates sum over n_eff

        g_interm = leaky_relu(self.aggreg(torch.cat([p_mean, p_sum, g_data], dim = 1)))
        g_interm = leaky_relu(self.g_out(g_interm) + g_data)

        #recreate the needed dimension for particle features
        #use torch.Tensor.view(shape) to reintroduce the 2nd dimension (number of particles)

        p_interm = (g_interm.view([-1, 1, self.hid_size_g])).repeat([1, n_eff, 1])
        p_interm = leaky_relu(self.distrib(torch.cat([p_interm, p_data], dim = 2)))
        p_interm = leaky_relu(self.p_out(p_interm) + p_data)

        return p_interm, g_interm







class Generator(nn.Module):
    """Defines the generator of the GAN.
    Note that the two hidden sizes (p and g_in) are the same in the default
    structure (both 128).
    """

    def __init__(self, input_size_p = 3, input_size_g = 10,
                 hid_size_p = 128, hid_size_g = 10, hid_size_g_in = 128,
                 num_epic_layers = 6):
        """Class constructor.

        Arguments
        ------------

        input_size_p: int, default: 3
            dimension of the input particle space; equals number of particle features;
            equals also the dimension of the output particle space
        
        input_size_g: int, default: 10
            dimension of the input global space

        hid_size_p: int, default: 128
            hidden dimension of the particle space in the network

        hid_size_g: int, default: 10
            hidden dimension of the global space in the network

        hid_size_g_in: int, default: 128
            dimension of the hidden layer in the initial processing of the global space

        num_epic_layers: int, default: 6
            number of EPiC-layers stacked in series 
        """

        super(Generator, self).__init__()

        self.input_size_p    = input_size_p
        self.input_size_g    = input_size_g
        self.hid_size_p      = hid_size_p
        self.hid_size_g      = hid_size_g
        self.hid_size_g_in   = hid_size_g_in     #only used in the encoder Phi_g_in
        self.num_epic_layers = num_epic_layers

        #layers
        self.p_in   = weight_norm(nn.Linear(self.input_size_p, self.hid_size_p))
        self.p_out  = weight_norm(nn.Linear(self.hid_size_p, self.input_size_p))

        self.g_in_1 = weight_norm(nn.Linear(self.input_size_g, self.hid_size_g_in))
        self.g_in_2 = weight_norm(nn.Linear(self.hid_size_g_in, self.hid_size_g))

        self.epic_layers = nn.ModuleList()
        for _ in range(self.num_epic_layers):
            self.epic_layers.append(EpicGanLayer(hid_size_p = self.hid_size_p,
                                    hid_size_g = self.hid_size_g,
                                    hid_size_g_in = self.hid_size_g_in))



    def forward(self, p_random_data, g_random_data):
        """Forward pass.

        Arguments
        ----------

        p_random_data: torch.Tensor
            input noise for particle space of shape [batch_size, n_eff, num_particle_features]

        g_random_data: torch.Tensor
            input noise for global space of shape [batch_size, input_size_g]


        Returns
        ----------

        p_interm: torch.Tensor
            generated events of shape [batch_size, n_eff, num_particle_features]
        """
        p_interm = leaky_relu(self.p_in(p_random_data))

        g_interm = leaky_relu(self.g_in_1(g_random_data))
        g_interm = leaky_relu(self.g_in_2(g_interm))

        p_first_epic_input = p_interm  #needed for residuals
        g_first_epic_input = g_interm  #needed for residuals

        for j in range(self.num_epic_layers):
            p_interm, g_interm = self.epic_layers[j](p_interm, g_interm)
            p_interm = p_interm + p_first_epic_input
            g_interm = g_interm + g_first_epic_input

        p_interm = self.p_out(p_interm)

        return p_interm




class Discriminator(nn.Module):
    """Defines the discriminator of the GAN. 
    """

    def __init__(self, input_size_p = 3, hid_size_p = 128,
                 hid_size_g = 10, hid_size_g_in = 128, num_epic_layers = 3):
        """
        Arguments 
        -----------

        input_size_p: int, default: 3
            dimension of the input particle space; equals number of particle features;

        hid_size_p: int, default: 128
            hidden dimension of the particle space in the network

        hid_size_g: int, default: 10
            hidden dimension of the global space in the network

        hid_size_g_in: int, default: 128
            dimension of the hidden layer in the initialisation of the global space

        num_epic_layers: int, default: 3
            number of EPiC-layers stacked in series
        """
        super(Discriminator, self).__init__()

        self.input_size_p    = input_size_p
        self.hid_size_p      = hid_size_p
        self.hid_size_g      = hid_size_g
        self.hid_size_g_in   = hid_size_g_in     #only used in the encoder Phi_g_in
        self.num_epic_layers = num_epic_layers

        self.p_in_1 = weight_norm(nn.Linear(self.input_size_p, self.hid_size_p))
        self.p_in_2 = weight_norm(nn.Linear(self.hid_size_p, self.hid_size_p))

        self.g_in_1 = weight_norm(nn.Linear(int(2*self.hid_size_p), self.hid_size_g_in))
        self.g_in_2 = weight_norm(nn.Linear(self.hid_size_g_in, self.hid_size_g))

        self.g_out_1 = weight_norm(nn.Linear(int(2*self.hid_size_p), self.hid_size_p))
        self.g_out_2 = weight_norm(nn.Linear(self.hid_size_p, self.hid_size_p))
        #3rd layer gives scalar output that decides real/fake
        self.g_out_3 = weight_norm(nn.Linear(self.hid_size_p, 1))

        self.epic_layers = nn.ModuleList()
        for _ in range(self.num_epic_layers):
            self.epic_layers.append(EpicGanLayer(hid_size_p = self.hid_size_p,
                                    hid_size_g = self.hid_size_g,
                                    hid_size_g_in = self.hid_size_g_in))


    def forward(self, p_data):
        """Forward pass.

        Arguments
        ----------

        p_data: torch.Tensor
            input data, either real or generated of shape [batch_size, n_eff, num_particle_features]


        Returns
        ----------

        out: torch.Tensor
            assigned label for each event, shape [batch_size]        
        """
        p_interm = leaky_relu(self.p_in_1(p_data))
        p_interm = leaky_relu(self.p_in_2(p_interm) + p_interm)

        p_mean = torch.mean(p_interm, 1)  #calculates mean over n_points
        p_sum  = torch.sum(p_interm, 1)    #calculates sum over n_points

        g_interm = leaky_relu(self.g_in_1(torch.cat([p_mean, p_sum], dim = 1)))
        g_interm = leaky_relu(self.g_in_2(g_interm))

        #activate the commented lines to include residual connection from input
        #of first epic layer to output of each of them
        #p_first_epic_input = p_interm    #needed for residual connections
        #g_first_epic_input = g_interm    #needed for residual connections

        for j in range(self.num_epic_layers):
            p_interm, g_interm = self.epic_layers[j](p_interm, g_interm)
            #p_interm = p_interm + p_first_epic_input
            #g_interm = g_interm + g_first_epic_input

        p_mean = torch.mean(p_interm, 1)  #calculates mean over n_points
        p_sum  = torch.sum(p_interm, 1)    #calculates sum over n_points

        p_interm = leaky_relu(self.g_out_1(torch.cat([p_mean, p_sum], dim = 1)))
        p_interm = leaky_relu(self.g_out_2(p_interm))
        out      = leaky_relu(self.g_out_3(p_interm))

        return out
