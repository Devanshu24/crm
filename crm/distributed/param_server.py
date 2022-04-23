import numpy as np
import ray
import torch

from crm.core import Network


@ray.remote
class ParameterServer(object):
    def __init__(self, lr, num_neurons, adj_list, custom_activations=None):
        self.network = Network(num_neurons, adj_list, custom_activations)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.network.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.network.get_weights()

    def get_weights(self):
        return self.network.get_weights()
