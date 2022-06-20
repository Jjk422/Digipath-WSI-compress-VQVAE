# import numpy as np
from sys import exit

import torch
import torchvision
import argparse

# Custom classes
from helper_classes import ImageDisplay
from new_code_rewrite.libraries.NetworkTester import Dataset
from helper_classes.NetworkController import NetworkController
from helper_classes.LossFunction import LossFunction
from network_library.ConvVAE import ConvVAE
from network_library.VariationalAutoEncoder import VariationalAutoEncoder

from network_library.DecayVAE import DecayVAE
from network_library.GoogleNetVAE import GoogleNetVAE
from network_library.github_architectures.Kuc2477VAE import Kuc2477VAE


def get_command_line_args():
    """
    Get command line arguments from std input
    :return: Argument parser object
    """
    parser = argparse.ArgumentParser(description='Train Networks.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--dataset', type=str, default="MNIST", help='Dataset to use')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--train', action='store_true', help='Train and save a model')
    parser.add_argument('--test', action='store_true', help='Load and test a model')

    return parser.parse_args()


def run(args, network_control):
    if args.train:
        network_control.train(epochs=args.epochs)

        network_control.save(f"{network_control.root_directory}/finished_model_{args.epochs}.pt")
    if args.test:
        network_control.load(f"{network_control.root_directory}/finished_model_{args.epochs}.pt")

    if not (args.train or args.test):
        exit(0)

    network_control.write_latent_space_to_file()

    images, classifications = next(dataset.get_test_loader_iter())

    ImageDisplay.Image.display_image_subplot(12, 12, images, no_colour_channels=(dataset.image_dim[2] == 1),
                                             cmap="gray", remove_axes=True,
                                             save_path=f"{network_control.root_directory}/origional_images_test_sample"
                                             )

    reconstructed_images, _, _ = network_control.predict(images, collapse_channels=False)
    reconstructed_images = reconstructed_images.detach().cpu()
    ImageDisplay.Image.display_image_subplot(12, 12,
                                             reconstructed_images.reshape(-1, dataset.image_dim[2],
                                                                          dataset.image_dim[0],
                                                                          dataset.image_dim[1]),
                                             no_colour_channels=(dataset.image_dim[2] == 1), cmap="gray",
                                             remove_axes=True,
                                             save_path=f"{network_control.root_directory}/reconstucted_images_test_sample")

    try:
        distribution_sample = torch.normal(0, 1, size=(128, network_control.network.number_of_latent_variables))

        decoded_sample_images = network_control.decode_sample(distribution_sample.to(network_control.network.device))
        decoded_sample_images = decoded_sample_images.detach().cpu()

        ImageDisplay.Image.display_image_subplot(12, 12,
                                                 decoded_sample_images.reshape(-1, dataset.image_dim[2],
                                                                               dataset.image_dim[0],
                                                                               dataset.image_dim[1]),
                                                 no_colour_channels=(dataset.image_dim[2] == 1), cmap="gray",
                                                 remove_axes=True,
                                                 save_path=f"{network_control.root_directory}/decoded_sample_images"
                                                 )
    except Exception as exception:
        with open(f"{network_control.root_directory}/decoded_sample_images_failed.txt", "w+") as file:
            file.write(str(exception))

if __name__ == "__main__":
    args = get_command_line_args()

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    if args.dataset == "MNIST":
        dataset = Dataset.MNIST(data_storage_directory="data_storage_directory", download=True, transform=transform)
    elif args.dataset == "CIFAR10":
        dataset = Dataset.CIFAR10(data_storage_directory="data_storage_directory", download=True,
                                  transform=transform)
    elif args.dataset == "CIFAR100":
        dataset = Dataset.CIFAR100(data_storage_directory="data_storage_directory", download=True,
                                   transform=transform)
    else:
        print(f"Dataset [{args.dataset}] was not a valid dataset")
        exit(0)

    print(f"--- Current dataset: {dataset.__class__.__name__} ---")

    datasets = [
        Dataset.MNIST(data_storage_directory="data_storage_directory", download=True, transform=transform),
        Dataset.CIFAR10(data_storage_directory="data_storage_directory", download=True, transform=transform),
        Dataset.CIFAR100(data_storage_directory="data_storage_directory", download=True, transform=transform)
    ]

    for dataset in datasets:
        dataset_name = dataset.__class__.__name__

        network_controllers = [
            NetworkController(
                ConvVAE(dataset.image_dim, loss_function=LossFunction.mean_squared_error),
                dataset,
                device=args.device,
                root_directory=f"models/ConvVAE_{dataset_name}"
            ),
            NetworkController(
                Kuc2477VAE(dataset.image_dim, loss_function=LossFunction.mean_squared_error),
                dataset,
                device=args.device,
                root_directory=f"models/githubArchitecture_Kuc2477VAE_{dataset_name}"
            ),
            NetworkController(
                GoogleNetVAE(dataset.image_dim, loss_function=LossFunction.mean_squared_error),
                dataset,
                device=args.device,
                root_directory=f"models/GoogleNetVAE_{dataset_name}"
            ),
            # NetworkController(
            #     GoogleNetVAE(dataset.image_dim, loss_function=LossFunction.mean_squared_error),
            #     Dataset.CIFAR100(
            #         data_storage_directory="data_storage_directory",
            #         download=True,
            #         transform=torchvision.transforms.Compose([
            #             torchvision.transforms.ToTensor(),
            #             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #         ])
            #     ),
            #     device=args.device,
            #     root_directory=f"models/GoogleNetVAE_googleNet_normalised_dataset_{dataset_name}"
            # ),
            NetworkController(
                DecayVAE(dataset.image_dim, loss_function=LossFunction.mean_squared_error),
                dataset,
                device=args.device,
                root_directory=f"models/DecayVAE_{dataset_name}"
            ),
            NetworkController(
                VariationalAutoEncoder(dataset.image_dim, loss_function=LossFunction.mean_squared_error),
                dataset,
                device=args.device,
                root_directory=f"models/VariationalAutoEncoder_{dataset_name}"
            ),
            NetworkController(
                ConvVAE(dataset.image_dim, loss_function=LossFunction.mean_squared_error, number_of_latent_variables=8),
                dataset,
                device=args.device,
                root_directory="models/ConvVAE_eight_latent_variables",
            ),w
        ]

        for network_controller in network_controllers:
            run(args, network_controller)