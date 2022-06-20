# import numpy as np

import torch
import torchvision
import argparse

# Custom classes
from libraries.NetworkTester import Dataset
from helper_classes.NetworkController import NetworkController
from network_library.VQVAECustom import VQVAECustom
from network_library.BasicCNN import BasicCNN

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

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# from PIL import Image
# import tifffile

from PIL import Image
# im = Image.open('a_image.tif')

import matplotlib.pyplot as plt
# I = plt.imread("G:/Datasets/PhD_datasets/Camelyon/drive-download-20200717T095305Z-004/normal_001.tif")

# test = tifffile.imread("G:/Datasets/PhD_datasets/Camelyon/drive-download-20200717T095305Z-004/normal_001.tif")

# test = Image.open("G:/Datasets/PhD_datasets/Camelyon/drive-download-20200717T095305Z-004/normal_001.tif")
# test = Image("G:/Datasets/PhD_datasets/Camelyon/drive-download-20200717T095305Z-004/normal_001.tif").getdata()
# testdata = test.getdata()
# print(testdata[0:10][0:10])

dataset = Dataset.CIFAR100(data_storage_directory="data_storage_directory", download=True,
                           transform=transform)

network_control_VQVAE = NetworkController(
    VQVAECustom(dataset.image_dim[2]),
    dataset,
    device="cpu",
    root_directory="models/VQVAE_CNN/VQVAE"
)

network_control_VQVAE.train(epochs=5)
# network_control_VQVAE.save(f"{network_control_VQVAE.root_directory}/finished_model_{network_control_VQVAE.network.epoch + 1}.pt")
network_control_VQVAE.load("D:/PhD_Project_testing/AI-Project-Framework/models/VQVAE_CNN/VQVAE_CIFAR100/finished_model_5.pt")

network_control_CNN = NetworkController(
    BasicCNN(dataset.image_dim),
    dataset,
    device="cuda",
    root_directory="models/VQVAE_CNN/BasicCNN"
)

