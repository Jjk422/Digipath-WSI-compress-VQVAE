import InitialiseEnvironment

currentdir, parentdir, date_today, date_time_today, date_time = InitialiseEnvironment.Initialise(
	conda_requirements_directory_path="/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history",
	create_files=True,
	copy_library_code=True,
	library_code_location="/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression",
	directory_to_copy_library_code_to="/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history",
	main_filename=__file__
)

import torchvision

# import os
from pathlib import Path

from new_code_rewrite.libraries.NetworkTester.Dataset import Camelyon2016, LeedsUnidentifiedAlexSlides, UCLA
from new_code_rewrite.libraries.NetworkTester.helper_classes.NetworkController import NetworkController
from new_code_rewrite.libraries.Metrics.Logger import Logger

# from new_code_rewrite.libraries.

from networks.VQVAE import VQVAE

configurations = [
    {
        "id": "embed_dim_2--randomised_colour_augmentation",
        "network_saved_path": f"/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history/2022-03-29/2022_03_29-15_17_19/models/VQVAE-embed_dim_2_randomised_colour_augmentation-2000_Epochs-Camelyon2016-in_channel_3-channel_128-n_res_block_16-n_res_channel_32-embed_dim_2-n_embed_512-decay_0.99.pt",
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 2,
            "n_embed": 512,
            "decay": 0.99,
            # "loss_function": LossFunction.mean_squared_error
        },
        "num_latent_features": 2,
        "epochs": 2000
    },

    {
        "id": "embed_dim_4--randomised_colour_augmentation",
        "network_saved_path": f"/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history/2022-03-29/2022_03_29-15_17_19/models/VQVAE-embed_dim_4_randomised_colour_augmentation-2000_Epochs-Camelyon2016-in_channel_3-channel_128-n_res_block_16-n_res_channel_32-embed_dim_4-n_embed_512-decay_0.99.pt",
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 4,
            "n_embed": 512,
            "decay": 0.99,
        },
    },

    #{
    #    "id": "embed_dim_6--randomised_colour_augmentation",
    #    "network_saved_path": f"/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history/2022-03-29/2022_03_29-15_17_19/models/",
    #    "model": VQVAE,
    #    "model_kwargs": {
    #        "in_channel": 3,
    #        "channel": 128,
    #        "n_res_block": 16,
    #        "n_res_channel": 32,
    #        "embed_dim": 6,
    #        "n_embed": 512,
    #        "decay": 0.99,
    #    },
    #},

    {
        "id": "embed_dim_2--no_data_augmentation",
        "network_saved_path": f"/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history/2022-03-29/2022_03_29-15_17_19/models/VQVAE-embed_dim_2_no_colour_aug-2000_Epochs-Camelyon2016-in_channel_3-channel_128-n_res_block_16-n_res_channel_32-embed_dim_2-n_embed_512-decay_0.99.pt",
        # "dataset": {
        #     "testing_datasets": {
        #         "Camelyon2016 subset": Camelyon2016,
        #         "UCLA": UCLA,
        #         "Internal validation set": LeedsUnidentifiedAlexSlides,
        #         },
        #     },
        # "dataset_kwargs": {
        #     "testing_datasets": {
        #         "Camelyon2016 subset": {
        #             "patch_dimension": (512,512,3),
        #             "data_storage_directory": "X:/Datasets/Camelyon/thresholded_patches",
        #             "transform": torchvision.transforms.Compose([
        #                 torchvision.transforms.ToTensor(),
        #                 torchvision.transforms.Resize((512, 512))
        #             ]),
        #             "batch_size": (4, 4)
        #         },
        #         "UCLA": {
        #             "data_storage_directory": "X:/Datasets/UCLA/Slides",
        #             "transform": torchvision.transforms.Compose([
        #                 torchvision.transforms.ToTensor(),
        #                 torchvision.transforms.Resize((512, 512))
        #             ]),
        #             "batch_size": (4, 4)
        #         },
        #         "Internal validation set": {
        #             "data_storage_directory": "X:/Datasets/LeedsUnidentifiedAlexSlides/Thresholded_patches",
        #             "transform": torchvision.transforms.Compose([
        #                 torchvision.transforms.ToTensor(),
        #                 torchvision.transforms.Resize((512, 512))
        #             ]),
        #             "batch_size": (4, 4)
        #         },
        #     },
        # },
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 2,
            "n_embed": 512,
            "decay": 0.99,
            # "loss_function": LossFunction.mean_squared_error
        },
        "num_latent_features": 2,
        "epochs": 2000
    },

    {
        "id": "embed_dim_4--no_data_augmentation",
        "network_saved_path": f"/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history/2022-03-29/2022_03_29-15_17_19/models/VQVAE-embed_dim_4_no_colour_aug-2000_Epochs-Camelyon2016-in_channel_3-channel_128-n_res_block_16-n_res_channel_32-embed_dim_4-n_embed_512-decay_0.99.pt",
        "model": VQVAE,
        "model_kwargs": {
            "in_channel": 3,
            "channel": 128,
            "n_res_block": 16,
            "n_res_channel": 32,
            "embed_dim": 4,
            "n_embed": 512,
            "decay": 0.99,
        },
    },

    #{
    #    "id": "embed_dim_6--no_data_augmentation",
    #    "network_saved_path": f"/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history/2022-03-29/2022_03_29-15_17_19/models/",
    #    "model": VQVAE,
    #    "model_kwargs": {
    #        "in_channel": 3,
    #        "channel": 128,
    #        "n_res_block": 16,
    #        "n_res_channel": 32,
    #        "embed_dim": 6,
    #        "n_embed": 512,
    #        "decay": 0.99,
    #    },
    #},
]

def initialise_config(config_dict):
    kwarg_keys = [i for i in config_dict.keys() if "_kwargs" in i]

    for kwarg_key in kwarg_keys:
        property_key = kwarg_key.replace("_kwargs", "")
        config_dict[property_key] = config_dict[property_key](**config_dict[kwarg_key])

for config in configurations:
    root_dir = f"D:/Users/SteamLink/Desktop/PhD_nobackup_local_testing/2022_03_16-00_13_00_models/testing/{date_today}/{date_time_today}"
    log_dir_path = f"{root_dir}/models/tensorboard_logs/{config['model'].__class__.__name__}-{config['id']}"
    Path(log_dir_path).mkdir(parents=True, exist_ok=True)
    initialise_config(config)

    logger = Logger(config["model"], log_directory=log_dir_path)

    network_controller = NetworkController(
        config["model"],
        Camelyon2016((512, 512, 3), data_storage_directory="X:/Datasets/Camelyon/thresholded_patches", transform=torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()])),
        device="cuda",
        things_to_log=["scalars"],
        # things_to_log=["scalars", "images"],
        logger=logger,
        display_progress_bar=True,
        root_directory = root_dir,
    )

    network_controller.load(config['network_saved_path'])
    network_controller.network.eval()
    print(f"Finished loading model {config['model'].__class__.__name__} from path {config['network_saved_path']}")# on dataset {config['dataset'].__class__.__name__}")

    # for dataset_index, dataset in enumerate(config.testing_datasets):
    #     network_controller.test(dataset=dataset(**config.dataset_kwargs[dataset_index]))
    #     print(f"Finished Testing model {config['model'].__class__.__name__} on dataset {dataset.__class__.__name__}")

    network_controller.test(dataset=Camelyon2016((512, 512, 3), data_storage_directory="X:/Datasets/Camelyon/thresholded_patches", transform=torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()])))
    print(f"Finished Testing model {config['model'].__class__.__name__} on dataset Camelyon2016 subset")

    # print(next(network_controller.dataset.get_test_loader_iter())[0].shape)
    # quant_t, quant_b, *_ = network_controller.network.encode(next(network_controller.dataset.get_test_loader_iter())[0].to("cuda"))
    #
    # print(quant_t.shape)
    # print(quant_b.shape)

    network_controller.test(dataset=UCLA("X:/Datasets/UCLA/Slides", transform=torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()])))
    print(f"Finished Testing model {config['model'].__class__.__name__} on dataset UCLA")

    network_controller.test(dataset=LeedsUnidentifiedAlexSlides((512,512,3), "X:/Datasets/LeedsUnidentifiedAlexSlides/Thresholded_patches", transform=torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()])))
    print(f"Finished Testing model {config['model'].__class__.__name__} on dataset internal validation set")


        # print(network_controller.network)
