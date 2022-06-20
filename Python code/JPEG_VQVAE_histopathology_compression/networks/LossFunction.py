import torch
import torch.nn.functional as F

class LossFunction():
    @staticmethod
    def binary_cross_entropy(reconstructed_images, images):
        return torch.nn.functional.binary_cross_entropy(
            reconstructed_images,
            images,
            reduction='sum'
        )

    # # Old backup no division by size
    # @staticmethod
    # def mean_squared_error(reconstructed_images, images):
    #     return torch.sum(
    #         (reconstructed_images - images) ** 2)

    @staticmethod
    def mean_squared_error(reconstructed_images, images):
        return F.mse_loss(reconstructed_images, images, size_average=False)#.div(images.size(0))

        # return F.mse_loss(reconstructed_images, images, size_average=False).div(images.size(0))