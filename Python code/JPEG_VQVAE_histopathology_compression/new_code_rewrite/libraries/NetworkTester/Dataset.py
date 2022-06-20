import os

# import natsort as natsort
import torch
import torchvision
from PIL import Image


class Dataset():
    # Template class for datasets
    # All inheriting dataset classes should override the load_dataset
    def __init__(self, data_storage_directory="", transform=None, target_transform=None, test_set_transform=None, download=False,
                 batch_size=(128, 128), normalisation=[None, None], dataset_name=None):
        self.data_storage_directory = data_storage_directory
        self.target_transform = target_transform
        self.image_dim = None
        self.transform = transform if transform is not None else torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        self.download = download
        self.normalisation = normalisation
        self.batch_size = batch_size
        self.dataset_name = self.__class__.__name__ if dataset_name is None else dataset_name
        self.test_set_transform = transform if test_set_transform is None else test_set_transform

        # self.data_storage_directory = data_storage_directory
        # self.training_dataset = self.load_dataset(data_storage_directory, for_training=True, transform=transform, target_transform=target_transform, download=download)
        # self.testing_dataset = self.load_dataset(data_storage_directory, for_training=False, transform=transform, target_transform=target_transform, download=download)
        self.training_data_loader = self.load_data_loader(data_storage_directory=self.data_storage_directory,
                                                          for_training=True, batch_size=batch_size[0],
                                                          shuffle=True, transform=self.transform, download=download)
        self.testing_data_loader = self.load_data_loader(data_storage_directory=self.data_storage_directory,
                                                         for_training=False, batch_size=batch_size[1],
                                                         shuffle=False, transform=self.test_set_transform, download=download)

    def load_dataset(self, root_path, for_training=True, transform=None, target_transform=None, download=False):
        # This method should be overridden in inherited class methods
        # return torchvision.datasets.MNIST(root_path, train=for_training, transform=transform,
        #                                   target_transform=self.target_transform, download=download)
        raise NotImplementedError("load_dataset function not overridden in sub Dataset() child or Dataset() is being called directly")
        # return None

    def load_data_loader(self, data_storage_directory="data_storage_directory/", for_training=True, batch_size=128,
                         shuffle=True, transform=None, download=False):
        return torch.utils.data.DataLoader(
            self.load_dataset(self.data_storage_directory,
                              for_training=for_training,
                              download=download,
                              transform=transform
                              ),
            batch_size=batch_size,
            shuffle=shuffle
        )
        # return torch.utils.data.DataLoader(
        #     torchvision.datasets.MNIST('data_storage_directory/',
        #                                train=for_training,
        #                                download=download,
        #                                transform=transform
        #                                ),
        #     batch_size=batch_size,
        #     shuffle=shuffle
        # )

    def get_train_loader(self):
        return self.training_data_loader

    def get_train_loader_iter(self):
        return iter(self.get_train_loader())

    def get_test_loader(self):
        return self.testing_data_loader

    def get_test_loader_iter(self):
        return iter(self.get_test_loader())

    def get_image_dim(self):
        return self.image_dim

    def set_normalisation(self, normalisation):
        self.normalisation = normalisation


class MNIST(Dataset):
    def __init__(self, data_storage_directory="", transform=None, target_transform=None, download=False, batch_size=(128, 128)):
        super(MNIST, self).__init__(data_storage_directory, transform, target_transform, None, download, batch_size)
        self.image_dim = (28, 28, 1)

    def load_dataset(self, root_path, for_training=True, transform=None, target_transform=None, download=False):
        # This method should be overridden in inherited class methods
        return torchvision.datasets.MNIST(root_path, train=for_training, transform=transform,
                                          target_transform=target_transform, download=download)


class CIFAR10(Dataset):
    def __init__(self, data_storage_directory="", transform=None, target_transform=None, download=False, batch_size=(128, 128)):
        super(CIFAR10, self).__init__(data_storage_directory, transform, target_transform, None, download, batch_size)
        self.image_dim = (32, 32, 3)

    def load_dataset(self, root_path, for_training=True, transform=None, target_transform=None, download=False):
        # This method should be overridden in inherited class methods
        return torchvision.datasets.CIFAR10(root_path, train=for_training, transform=transform,
                                            target_transform=target_transform, download=download)


class CIFAR100(Dataset):
    def __init__(self, data_storage_directory="", transform=None, target_transform=None, download=False, batch_size=(128, 128)):
        super(CIFAR100, self).__init__(data_storage_directory, transform, target_transform, None, download, batch_size)
        # Automate image dimension selection from transform
        self.image_dim = (32, 32, 3)

    def load_dataset(self, root_path, for_training=True, transform=None, target_transform=None, download=False):
        # This method should be overridden in inherited class methods
        return torchvision.datasets.CIFAR100(root_path, train=for_training, transform=transform,
                                             target_transform=target_transform, download=download)


class Camelyon2016(Dataset):
    def __init__(self, patch_dimension, data_storage_directory="", transform=None, target_transform=None, test_set_transform=None, download=False,
                 batch_size=(4, 4)):
        self.image_dim = patch_dimension

        self.dataset_name = "Camelyon_2016"
        self.patch_features={
            "tile_size": 512,
            # "overlap": 12
            "overlap": 0
        }

        self.data_storage_directory = f"{data_storage_directory}/{self.dataset_name}/{self.patch_features['tile_size']}"

        super(Camelyon2016, self).__init__(self.data_storage_directory, transform, target_transform, test_set_transform, download,
                                           batch_size=batch_size, dataset_name=self.dataset_name)

    def load_dataset(self, threshold_patches_dir, for_training=True, transform=None, target_transform=None,
                     download=False,
                     ):
        # This method should be overridden in inherited class methods

        # camelyon_dataset = torchvision.datasets.ImageFolder(
        #     f"{threshold_patches_dir}/{self.dataset_name}/{patch_features['tile_size']}", transform=transform)
        # return torch.utils.data.DataLoader(camelyon_dataset, batch_size=self.batch_size, shuffle=False)

        # return torchvision.datasets.ImageFolder(f"{threshold_patches_dir}/{self.dataset_name}/{patch_features['tile_size']}", transform=transform)
        return torchvision.datasets.ImageFolder(self.data_storage_directory, transform=transform)

class LeedsUnidentifiedAlexSlides(Dataset):
    def __init__(self, patch_dimension, data_storage_directory="", transform=None, target_transform=None, test_set_transform=None, download=False,
                 batch_size=(4, 4)):
        self.image_dim = patch_dimension

        self.dataset_name = "Leeds_Unidentified_Alex_Slides"
        self.patch_features={
            "tile_size": 512,
            "overlap": 12
        }

        self.data_storage_directory = f"{data_storage_directory}/{self.dataset_name}/{self.patch_features['tile_size']}"

        super(LeedsUnidentifiedAlexSlides, self).__init__(self.data_storage_directory, transform, target_transform, test_set_transform, download,
                                           batch_size=batch_size, dataset_name=self.dataset_name)

    def load_dataset(self, data_storage_directory, for_training=True, transform=None, target_transform=None,
                     download=False,
                     ):
        # This method should be overridden in inherited class methods

        # camelyon_dataset = torchvision.datasets.ImageFolder(
        #     f"{threshold_patches_dir}/{self.dataset_name}/{patch_features['tile_size']}", transform=transform)
        # return torch.utils.data.DataLoader(camelyon_dataset, batch_size=self.batch_size, shuffle=False)

        return torchvision.datasets.ImageFolder(data_storage_directory, transform=transform)

# # TODO: Need to fix this dataset class
# class BreastCancerCells(Dataset):
#     # Dataset from https://bioimage.ucsb.edu/research/bio-segmentation
#     # Currently stored in G:\Datasets\Cancer image dataset\bisque-20200630.152458\Breast Cancer Cells
#     def __init__(self, data_storage_directory="", transform=None, target_transform=None, download=False):
#         super(BreastCancerCells, self).__init__(data_storage_directory, transform, target_transform, download)
#         self.image_dim = (896, 768, 3)
#
#         self.data_storage_directory = data_storage_directory
#         self.transform = transform
#         all_imgs = [file for file in os.listdir(data_storage_directory) if file.endswith(".tif")]
#         # TODO: Rewrite this, it is turning into spagetti code
#         self.filename_list_all = natsort.natsorted(all_imgs)
#         self.filename_list_train = self.filename_list_all[:-int(len(self.filename_list_all)*0.2)]
#         self.filename_list_test = self.filename_list_all[-int(len(self.filename_list_all)*0.2):]
#         self.filename_list = []
#
#
#     def load_dataset(self, root_path, for_training=True, transform=None, target_transform=None, download=False):
#         # This method should be overridden in inherited class methods
#         if for_training:
#             self.filename_list = self.filename_list_train
#         else:
#             self.filename_list = self.filename_list_test
#         return self
#
#     def __len__(self):
#         return len(self.filename_list)
#
#     def __getitem__(self, idx):
#         img_loc = os.path.join(self.data_storage_directory, self.filename_list[idx])
#         image = Image.open(img_loc).convert("RGB")
#         tensor_image = self.transform(image)
#         return tensor_image

class UCLA(Dataset):
    def __init__(self, data_storage_directory="", transform=None, target_transform=None, test_set_transform=None, download=False,
                 batch_size=(4, 4)):
        self.image_dim = (876, 768, 3)

        self.dataset_name = "UCLA"

        super(UCLA, self).__init__(data_storage_directory, transform, target_transform, test_set_transform, download,
                                           batch_size=batch_size, dataset_name=self.dataset_name)

    def load_dataset(self, data_dir, for_training=True, transform=None, target_transform=None,
                     download=False,
                     ):
        # This method should be overridden in inherited class methods

        return torchvision.datasets.ImageFolder(f"{data_dir}", transform=transform)