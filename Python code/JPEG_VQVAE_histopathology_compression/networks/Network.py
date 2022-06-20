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

    def set_batch_data(self, images, classifications):
        self.batch_data["images"] = images.to(self.device)
        self.batch_data["classifications"] = classifications.to(self.device)

    def set_epoch(self, epoch):
        self.epoch = epoch

    # def train(self):
    #     print("Please implement custom train function in network class")
    #     exit(0)

    # def test(self):
    #     print("Please implement custom test function in network class")
    #     exit(0)
