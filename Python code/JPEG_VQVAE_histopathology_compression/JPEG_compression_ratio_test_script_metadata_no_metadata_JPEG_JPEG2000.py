import InitialiseEnvironment

import random

currentdir, parentdir, date_today, date_time_today, date_time = InitialiseEnvironment.Initialise(create_files=True, copy_library_code=False, directory_to_copy_library_code_to="/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history", main_filename=__file__)

import torchvision

from pathlib import Path

from new_code_rewrite.libraries.NetworkTester.Dataset import Camelyon2016
from new_code_rewrite.libraries.NetworkTester.helper_classes.NetworkController import NetworkController
from new_code_rewrite.libraries.Metrics.Logger import Logger

from networks.VQVAE import VQVAE

# default_config = {
#     "dataset": MNIST,
#     "dataset_kwargs": {"data_storage_directory": "X:/Datasets/MNIST", "download": True, "transform": "torchvision.transforms.ToTensor()"},
#     "model": DecayVAE,
#     "model_kwargs": {"starting_image_shape": (28,28,1), "number_of_latent_features": 3},
#     "num_latent_features": 3,
#     "epochs": 10
# }

class RandomiseColourChannels(object):
    """Randomise all colour channels in the tensor."""
    def __call__(self, image):
        channel_to_zero = random.randint(0, 2)
        image[:][:][channel_to_zero] *= random.uniform(0, 2)
        return image

class RandomiseColourChannelsChooseDist(object):
    def __init__(self, distribution_function=random.uniform(0, 2)):
        self.distribution_function = distribution_function

    """Randomise all colour channels in the tensor."""
    def __call__(self, image):
        # channel_to_zero = random.randint(0, 2)
        for channel_index in range(len(image[0][0])):
            image[:][:][channel_index] *= self.distribution_function[channel_index]
            return image

configurations = [
#    {
#        "id": "simple_1_epoch_tester_config",
#        "dataset": Camelyon2016,
#        "dataset_kwargs": {
#            "patch_dimension": (512,512,3),
#            "data_storage_directory": "/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/Datasets/Camelyon 2016 subset/thresholded_patches",
#            "transform": torchvision.transforms.Compose([
#                torchvision.transforms.ToTensor(),
#                torchvision.transforms.Resize((512, 512)),
#		RandomiseColourChannels(),
#            ]),
#            "batch_size": (4, 4)
#        },
#        "model": VQVAE,
#        "model_kwargs": {
#            "in_channel": 3,
#            "channel": 128,
#            "n_res_block": 16,
#            "n_res_channel": 32,
#            "embed_dim": 2,
#            "n_embed": 512,
#            "decay": 0.99,
#            # "loss_function": LossFunction.mean_squared_error
#        },
#        "num_latent_features": 2,
#        "epochs": 1
#    },


#    {
#        "id": "embed_dim_2_no_colour_aug",
#        "dataset": Camelyon2016,
#        "dataset_kwargs": {
#            "patch_dimension": (512,512,3),
#            "data_storage_directory": "/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/Datasets/Camelyon 2016 subset/thresholded_patches",
#            "transform": torchvision.transforms.Compose([
#                torchvision.transforms.ToTensor(),
#                torchvision.transforms.Resize((512, 512)),
#                # RandomiseColourChannels(),
#                # RandomiseColourChannelsChooseDist(distribution_function=torch.randn(3, requires_grad=False)),
#            ]),
#            "batch_size": (4, 4)
#        },
#        "model": VQVAE,
#        "model_kwargs": {
#            "in_channel": 3,
#            "channel": 128,
#            "n_res_block": 16,
#            "n_res_channel": 32,
#            "embed_dim": 2,
#            "n_embed": 512,
#            "decay": 0.99,
#            # "loss_function": LossFunction.mean_squared_error
#        },
#        "num_latent_features": 2,
#        "epochs": 2000
#    },

#   {
#       "id": "embed_dim_2_randomised_colour_augmentation",
#       "dataset": Camelyon2016,
#       "dataset_kwargs": {
#           "patch_dimension": (512,512,3),
#           "data_storage_directory": "/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/Datasets/Camelyon 2016 subset/thresholded_patches",
#           "transform": torchvision.transforms.Compose([
#               torchvision.transforms.ToTensor(),
#               torchvision.transforms.Resize((512, 512)),
#	        RandomiseColourChannels(),
#        	# RandomiseColourChannelsChooseDist(distribution_function=torch.randn(3, requires_grad=False)),
#            ]),
#            "batch_size": (4, 4)
#        },
#        "model": VQVAE,
#        "model_kwargs": {
#            "in_channel": 3,
#            "channel": 128,
#            "n_res_block": 16,
#            "n_res_channel": 32,
#            "embed_dim": 2,
#            "n_embed": 512,
#            "decay": 0.99,
#            # "loss_function": LossFunction.mean_squared_error
#        },
#        "num_latent_features": 2,
#        "epochs": 2000
#    },

    {
        "id": "embed_dim_4_no_colour_aug",
        "dataset": Camelyon2016,
        "dataset_kwargs": {
            "patch_dimension": (512,512,3),
            "data_storage_directory": "/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/Datasets/Camelyon 2016 subset/thresholded_patches",
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512)),
                # RandomiseColourChannels(),
                # RandomiseColourChannelsChooseDist(distribution_function=torch.randn(3, requires_grad=False)),
            ]),
            "batch_size": (4, 4)
        },
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 4,
            "n_embed": 512,
            "decay": 0.99,
            # "loss_function": LossFunction.mean_squared_error
        },
        "num_latent_features": 4,
        "epochs": 2000
    },

    {
        "id": "embed_dim_4_randomised_colour_augmentation",
        "dataset": Camelyon2016,
        "dataset_kwargs": {
            "patch_dimension": (512,512,3),
            "data_storage_directory": "/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/Datasets/Camelyon 2016 subset/thresholded_patches",
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512)),
                # RandomiseColourChannels(),
                # RandomiseColourChannelsChooseDist(distribution_function=torch.randn(3, requires_grad=False)),
            ]),
            "batch_size": (4, 4)
        },
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 4,
            "n_embed": 512,
            "decay": 0.99,
            # "loss_function": LossFunction.mean_squared_error
        },
        "num_latent_features": 4,
        "epochs": 2000
    },

    {
        "id": "embed_dim_6_no_colour_aug",
        "dataset": Camelyon2016,
        "dataset_kwargs": {
            "patch_dimension": (512,512,3),
            "data_storage_directory": "/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/Datasets/Camelyon 2016 subset/thresholded_patches",
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512)),
                # RandomiseColourChannels(),
                # RandomiseColourChannelsChooseDist(distribution_function=torch.randn(3, requires_grad=False)),
            ]),
            "batch_size": (4, 4)
        },
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 6,
            "n_embed": 512,
            "decay": 0.99,
            # "loss_function": LossFunction.mean_squared_error
        },
        "num_latent_features": 6,
        "epochs": 2000
    },

    {
        "id": "embed_dim_6_randomised_colour_augmentation",
        "dataset": Camelyon2016,
        "dataset_kwargs": {
            "patch_dimension": (512,512,3),
            "data_storage_directory": "/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/Datasets/Camelyon 2016 subset/thresholded_patches",
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512)),
                RandomiseColourChannels(),
                # RandomiseColourChannelsChooseDist(distribution_function=torch.randn(3, requires_grad=False)),
            ]),
            "batch_size": (4, 4)
        },
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 6,
            "n_embed": 512,
            "decay": 0.99,
            # "loss_function": LossFunction.mean_squared_error
        },
        "num_latent_features": 6,
        "epochs": 2000
    },

    {
        "id": "embed_dim_8_no_colour_aug",
        "dataset": Camelyon2016,
        "dataset_kwargs": {
            "patch_dimension": (512,512,3),
            "data_storage_directory": "/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/Datasets/Camelyon 2016 subset/thresholded_patches",
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512)),
            ]),
            "batch_size": (4, 4)
        },
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 8,
            "n_embed": 512,
            "decay": 0.99,
            # "loss_function": LossFunction.mean_squared_error
        },
        "num_latent_features": 8,
        "epochs": 2000
    },

    {
        "id": "embed_dim_8_randomised_colour_augmentation",
        "dataset": Camelyon2016,
        "dataset_kwargs": {
            "patch_dimension": (512,512,3),
            "data_storage_directory": "/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/Datasets/Camelyon 2016 subset/thresholded_patches",
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((512, 512)),
                RandomiseColourChannels(),
                # RandomiseColourChannelsChooseDist(distribution_function=torch.randn(3, requires_grad=False)),
            ]),
            "batch_size": (4, 4)
        },
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 8,
            "n_embed": 512,
            "decay": 0.99,
            # "loss_function": LossFunction.mean_squared_error
        },
        "num_latent_features": 8,
        "epochs": 2000
    },
]

def initialise_config(config_dict):
    kwarg_keys = [i for i in config_dict.keys() if "_kwargs" in i]

    for kwarg_key in kwarg_keys:
        property_key = kwarg_key.replace("_kwargs", "")
        config_dict[property_key] = config_dict[property_key](**config_dict[kwarg_key])

for config in configurations:
    # log_dir_path = f"models/tensorboard_logs/{config['model'].__name__}-{config['epochs']}_Epochs-{config['dataset'].__class__.__name__}-{config['num_latent_features']}_latent_features"
    # log_dir_path = f"models/{date_today}/{date_time_today}/models/tensorboard_logs/{config['model'].__name__}-{config['epochs']}_Epo-{config['dataset'].__name__}-{config['num_latent_features']}_lat_feat"
    root_dir = f"/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history/{date_today}/{date_time_today}"
    log_dir_path = f"{root_dir}/models/tensorboard_logs/{config['model'].__class__.__name__}-{config['id']}-{config['epochs']}_Epochs-{config['dataset'].__class__.__name__}-{'-'.join([str(key) + '_' + str(value) for key, value in config['model_kwargs'].items()])}"
    Path(log_dir_path).mkdir(parents=True, exist_ok=True)
    # os.makedirs(log_dir_path, exist_ok=True)

    # config["logger"] = Logger(config["model"], log_directory=f"models/tensorboard_logs/{config['model'].__class__.__name__}__{config['epochs']}_Epochs__{config['dataset'].__class__.__name__}__{config['num_latent_features']}_latent_features")
    initialise_config(config)

    print(f"---- Running configuration for config id [{config['id']}] at datetime {date_time_today} -----")
    print(f"Log dir path set to {log_dir_path}")
    logger = Logger(config["model"], log_directory=log_dir_path)

    network_controller = NetworkController(
        config["model"],
        config["dataset"],
        device="cuda",
        things_to_log=["scalars", "images"],
        logger=logger,
        display_progress_bar=True,
        root_directory = root_dir,
    )
    network_controller.train(epochs = config["epochs"])

    network_controller.save(f"{root_dir}/models/{config['model'].__class__.__name__}-{config['id']}-{config['epochs']}_Epochs-{config['dataset'].__class__.__name__}-{'-'.join([str(key) + '_' + str(value) for key, value in config['model_kwargs'].items()])}.pt")
    print(f"Finished Training model {config['model'].__class__.__name__}-{config['id']} on dataset {config['dataset'].__class__.__name__} for {config['epochs']} epochs")
