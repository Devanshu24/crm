import ray
import torch.nn.functional as F

from crm.core import Network


@ray.remote
class DataWorker(object):
    def __init__(
        self, X_train, y_train, num_neurons, adj_list, custom_activations=None
    ):
        self.network = Network(num_neurons, adj_list, custom_activations)
        self.X_train = X_train
        self.y_train = y_train
        self.data_iterator = iter(zip(X_train, y_train))

    def compute_gradients(self, weights):
        self.network.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(zip(self.X_train, self.y_train))
            data, target = next(self.data_iterator)
        self.network.reset()
        # print(data)
        output = self.network.forward(data).reshape(1, -1)
        # print(output)
        # print(target)
        loss = F.cross_entropy(output, target.reshape(1))
        # print(loss)
        loss.backward()
        return self.network.get_gradients()
