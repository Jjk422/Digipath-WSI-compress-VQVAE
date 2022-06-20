from LossFunction import LossFunction
from Network import Network
import torch
import torch.nn as nn
import math


class TemplateVAE(Network):
    """
    VariationalAutoEncoder template
    """
    def __init__(self, input_dimension, loss_function=LossFunction.binary_cross_entropy, number_of_latent_variables=128):
        super(TemplateVAE, self).__init__()
        self.input_dimension = input_dimension
        self.loss_func = loss_function
        self.number_of_latent_variables = number_of_latent_variables
        self.latent_space = "Latent space not assigned in module"

    def set_batch_data(self, images, _):
        self.batch_data["images"] = images.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        encoded_mu, encoded_sigma = self.encode(x)
        reparameterised = self.reparameterise(encoded_mu, encoded_sigma)
        self.latent_space = reparameterised
        decoded_images = self.decode(reparameterised)
        return decoded_images, encoded_mu, encoded_sigma

    def loss_function(self, network_outputs):
        reconstructed_images, mu, logvar = network_outputs

        error = self.loss_func(reconstructed_images, self.batch_data["images"])

        kullback_leibler_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        print(f"err: {error} || DKL: {kullback_leibler_divergence}")

        # TODO: Remove if not needed
        divergence_change_rate = 0.1 # Importance factor (when should the divergence be chosen over the error between images and reconstructions)
        # divergence_weighting = (1 - (1/(self.epoch + 1 * change_rate)))
        # divergence_weighting = 1/math.exp(error/divergence_change_rate)
        divergence_weighting = 1/math.exp((kullback_leibler_divergence)/error)
        # divergence_weighting = 1/math.exp(error/(kullback_leibler_divergence * divergence_change_rate))

        # return error + kullback_leibler_divergence
        return error + (divergence_weighting * kullback_leibler_divergence)
