# import numpy as np
from sys import exit

import torch
import torchvision
import argparse

# Custom classes
from helper_classes import ImageDisplay
from libraries.NetworkTester import Dataset
from helper_classes.LatentSpace import LatentSpace
from helper_classes.LossFunction import LossFunction
from helper_classes.NetworkController import NetworkController
from network_library.ConvVAE import ConvVAE


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

# TODO: Rewrite and integrate extraction code into framework or VQVAE2 module
# This is based on the code found at: https://github.com/rosinality/vq-vae-2-pytorch/blob/master/extract_code.py
def extract(loader, model, device):
    code_rows_array = []
    for batch_index, batch in enumerate(loader):
        img = batch[0].to(device)

        _, _, _, id_t, id_b = model.encode(img)
        id_t = id_t.detach().cpu().numpy()
        id_b = id_b.detach().cpu().numpy()

        for row in zip(id_t, id_b):
            code_rows_array.append(row)

    return code_rows_array

if __name__ == "__main__":
    args = get_command_line_args()

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # TODO: Fix or remove [Normalisation code]
    # # Normalisation of images, [(mean), (standard_deviation)]
    normalisation = [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0)]
    # # For no normalisation do None, None
    # normalisation = [None, None]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(normalisation[0], normalisation[1])
    ])

    if args.dataset == "MNIST":
        dataset = Dataset.MNIST(data_storage_directory="data_storage_directory", download=True,
                                transform=transform)
    elif args.dataset == "CIFAR10":
        dataset = Dataset.CIFAR10(data_storage_directory="data_storage_directory", download=True,
                                  transform=transform)
    elif args.dataset == "CIFAR100":
        dataset = Dataset.CIFAR100(data_storage_directory="data_storage_directory", download=True,
                                   transform=transform)
    # TODO: Fix BreastCancer slides image class
    # elif args.dataset == "BreastCancer":
    #     dataset = Dataset.BreastCancerCells(
    #         data_storage_directory="G:/Datasets/Cancer image dataset/bisque-20200630.152458/Breast Cancer Cells",
    #         transform=transform)
    else:
        print(f"Dataset [{args.dataset}] was not a valid dataset")
        exit(0)

    # TODO: Fix or remove [Normalisation code]
    # dataset.set_normalisation(normalisation)

    print(f"--- Current dataset: {dataset.__class__.__name__} ---")

    network_control = NetworkController(
        ConvVAE(dataset.image_dim, loss_func=LossFunction.mean_squared_error, number_of_latent_variables=128),
        dataset,
        device=args.device,
        root_directory="models/ConvVAE",
    )

    # network_control = NetworkController(
    #     VQVAE(dataset.image_dim[2]),
    #     dataset,
    #     device=args.device,
    #     root_directory="models/VQVAE"
    # )

    if args.train:
        import time

        start_time = time.time()

        network_control.train(epochs=args.epochs)

        print(f"--- Trained in: {(time.time() - start_time)} seconds ---")

        with open(f"{network_control.root_directory}/training_time.txt", "w+") as file:
            file.write(f"{time.time() - start_time} seconds")

        network_control.save(f"{network_control.root_directory}/finished_model_{network_control.network.epoch + 1}.pt")
    if args.test:
        network_control.load(f"{network_control.root_directory}/finished_model_{args.epochs}.pt")
        # network_control.load("D:/PhD_Project_testing/AI-Project-Framework/models/VQVAE2_CIFAR100/finished_model_5.pt")


    if not (args.train or args.test):
        exit(0)

    network_control.write_latent_space_to_file()

    images, classifications = next(dataset.get_test_loader_iter())

    ImageDisplay.Image.display_image_subplot(12, 12, images, no_colour_channels=(dataset.image_dim[2] == 1),
                                             cmap="gray", remove_axes=True,
                                             save_path=f"{network_control.root_directory}/origional_images_test_sample",
                                             normalised_mean_and_std=dataset.normalisation
                                             )

    predicted_output = network_control.predict(images, collapse_channels=False)
    reconstructed_images = predicted_output[0].detach().cpu()
    ImageDisplay.Image.display_image_subplot(12, 12,
                                             reconstructed_images.reshape(-1, dataset.image_dim[2],
                                                                          dataset.image_dim[0],
                                                                          dataset.image_dim[1]),
                                             no_colour_channels=(dataset.image_dim[2] == 1), cmap="gray",
                                             remove_axes=True,
                                             save_path=f"{network_control.root_directory}/reconstucted_images_test_sample",
                                             normalised_mean_and_std=dataset.normalisation
                                             )

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
                                                 save_path=f"{network_control.root_directory}/decoded_sample_images",
                                                 normalised_mean_and_std=dataset.normalisation
                                                 )
    except Exception as exception:
        with open(f"{network_control.root_directory}/decoded_sample_images_failed.txt", "w+") as file:
            file.write(str(exception))


    # # Run one batch of the test set to get the mean and logvar values for plotting the latent space
    # images, classifications = next(dataset.get_test_loader_iter())
    # images = images.to(network_control.network.device)
    # mean, logvar = network_control.network.encode(images)
    # z = network_control.network.reparameterise(mean, logvar)
    # mean_np = mean.cpu().detach().numpy()
    # logvar = logvar.cpu().detach().numpy()
    #
    # # Display latent space
    # latent_space = LatentSpace(mean, labels=classifications)
    # latent_space.save(f"{network_control.root_directory}/latent_space_graph.png")
    # latent_space.display()


    latent_space = LatentSpace(extract(loader=dataset.get_test_loader(), model=network_control.network, device=network_control.network.device))