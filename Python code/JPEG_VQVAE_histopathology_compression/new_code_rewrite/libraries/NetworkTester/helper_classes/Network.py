import torch.nn as nn


class Network(nn.Module):
    """
    Main network class, all other network classes inherit from this
    """
    def __init__(self):
        super(Network, self).__init__()
        self.batch_data = {"images": None, "classifications": None}
        self.device = "cpu"
        self.epoch = 0

    def set_device(self, device):
        self.device = device

    def set_batch_data(self, images, classifications, **kwargs):
        self.batch_data["images"] = images.to(self.device)
        self.batch_data["classifications"] = classifications.to(self.device)
        for key, value in kwargs.items():
            self.batch_data[key] = value

    def set_epoch(self, epoch):
        self.epoch = epoch

    # def train(self):
    #     raise NotImplementedError("Please implement a custom train function in the network class")
    #
    # def test(self):
    #     raise NotImplementedError("Please implement a custom test function in the network class")

    def calculate_metrics(self, batch_images, metric_controller):
        raise NotImplementedError("Please implement a custom calculate_metrics function in the network class for additional metric calculations")

    def run_batch(self):
        raise NotImplementedError("Please implement a custom run function in the network class for batch processing")

    def loss_function(self, network_outputs):
        raise NotImplementedError("Please implement a loss_function function in the network class for loss calculating")