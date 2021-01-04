# encoding: utf-8
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: cluster_process.py
@Time: 2020-03-14 15:45
@Desc: cluster_process.py
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from vae.utils import init_weights
    from vae.config import DEVICE
    from hyperspherical_vae.distributions import VonMisesFisher
    from hyperspherical_vae.distributions import HypersphericalUniform

except ImportError as e:
    print(e)
    raise ImportError


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.shape
        )


class Generator(nn.Module):

    def __init__(self, latent_dim=50, output_channels=176, hidden_size=128, hidden_layer_depth=1, verbose=False):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.hidden_layer_depth = hidden_layer_depth
        self.hidden_size = hidden_size
        self.verbose = verbose

        self.RNN = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.linear = nn.Linear(latent_dim, output_channels)
        self.out = nn.Linear(hidden_size, 1)

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):

        x = self.linear(x)
        x = x.unsqueeze(2)
        x = self.RNN(x)[0]
        x = self.out(x)
        return x


class Encoder(nn.Module):

    def __init__(self, input_channels=176, output_channels=64, hidden_size=64, verbose=False):
        super(Encoder, self).__init__()

        self.output_channels = output_channels
        self.input_channels = input_channels
        self.verbose = verbose

        self.RNN = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.mu = nn.Linear(input_channels * hidden_size, self.output_channels)
        self.kappa = nn.Linear(input_channels * hidden_size, 1)

        init_weights(self)

        if self.verbose:
            print(self.model)

    def reparameterize(self, z_mean, z_kappa):

        q_z = VonMisesFisher(z_mean, z_kappa)
        p_z = HypersphericalUniform(z_mean.size(1) - 1, device=DEVICE)

        return q_z, p_z

    def forward(self, x):

        output, hidden = self.RNN(x)
        # h_end = output[:, :, -2]
        h_end = torch.flatten(output, 1, 2)
        mu = self.mu(h_end)
        mu = mu / torch.norm(mu, dim=1, keepdim=True)
        kappa = F.softplus(self.kappa(h_end)) + 1

        q_z, p_z = self.reparameterize(mu, kappa)
        z = q_z.rsample()
        return z, mu, kappa, q_z, p_z
