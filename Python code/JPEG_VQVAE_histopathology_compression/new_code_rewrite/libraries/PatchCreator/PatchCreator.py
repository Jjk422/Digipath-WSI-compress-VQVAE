# import Constants

import openslide
import openslide.deepzoom

import matplotlib.pyplot as plt
# from skimage import data, filters
from PIL import Image
import numpy as np

import torch
import torchvision
import os

"""
PatchCreator.py
====================================
Creates patches from a given SVS file or other histopathology slide scanner format.
"""
class PatchCreator():
    """
    PatchCreator does not have multithreaded support.
    """
    def __init__(
            self,
            dataset_name: str = "No Dataset Name",
            tile_size: int = 512,
            overlap: int = 0,
            deepzoom_level: int = 15
    ) -> None:
        """
        Initialise PatchCreator module.

        :param dataset_name: Name of the dataset
        :param tile_size: Size for the dataset tiles
        :param overlap: Overlap between each tile on the slide
        :param deepzoom_level: Level of deepzoom within the openslide library
        """
        self.metadata = {
            "dataset_name": "No Dataset Name",
        }

        self.patch_features = {
            "tile_size": tile_size,
            "overlap": overlap,
            "deepzoom_level": deepzoom_level,
        }

    def set_metadata(self, key: str, value: object) -> None:
        """
        Set metadata key to value.

        :param key: Metadata key name
        :param value: Metadata value
        """
        self.metadata[key] = value

    def set_dataset_name(self, name: object) -> None:
        """
        Set the name of the dataset to be used within patch generation.

        :param name: Dataset name
        :rtype: None
        """
        self.set_metadata("dataset_name", name)

    def set_output_patch_root_directory(self, directory_path: str) -> None:
        """
        Set the output patch root directory to be used within patch generation.

        :param directory_path: Path of the output root directory patch
        :rtype: None
        """
        self.set_metadata("output_patch_root_directory", directory_path)

    # Internal usage method used within the create_output_patch_directory_structure method
    def __set_output_patch_directory(self, directory_path: str) -> None:
        """
        Set the output patch directory. Private internal class method.

        :param directory_path: Path of the output patch directory
        :rtype: None
        """
        self.set_metadata("output_patch_directory", directory_path)

    def get_output_patch_root_directory(self) -> str:
        """
        Get the current output patch root directory.

        :rtype: str
        """
        return self.metadata['output_patch_root_directory']

    def get_output_patch_directory(self) -> str:
        """
        Get the current output patch directory.

        :rtype: str
        """
        return self.metadata['output_patch_directory']

    def get_current_output_directories(self) -> None:
        """
        Print the current output directory

        :rtype: None
        """
        print("--- Current output_directories ---")
        print(f"Root_directory: {self.get_output_patch_root_directory()}")
        print(f"Patch_directory: {self.get_output_patch_directory()}")

    def create_output_patch_directory_structure(self, output_patch_root_directory: str,
                                                class_name: str = "undefined_class") -> None:
        """
        Create the output directory structure used within patch creation.

        :param output_patch_root_directory: Output patch root directory.
        :param class_name: Name for the dataset class
        :rtype: None
        """
        self.set_output_patch_root_directory(output_patch_root_directory)
        self.__set_output_patch_directory(
            f"{output_patch_root_directory}/{self.metadata['dataset_name']}/{self.patch_features['tile_size']}/{class_name}")

        # class_name = "Normal"
        # image_type = "Training"
        #         output_patch_directory = f"{patches_dir}/{data_set_name}/{patch_features['tile_size']}/{class_name}"
        #         root_directory_full_images = f"{slides_dir}"
        #         os.makedirs(output_patch_root_directory, exist_ok=True)
        os.makedirs(
            f"{self.metadata['output_patch_root_directory']}/{self.metadata['dataset_name']}/{self.patch_features['tile_size']}/{class_name}",
            exist_ok=True)

    @staticmethod
    def get_slide_thumbnail(slide_path: str, thumbnail_size: tuple = (256, 256)) -> Image:
        """
        Get the slide thumbnail for a WSI slide.

        :param slide_path:
        :param thumbnail_size:
        :return:
        :rtype: object
        """
        return openslide.OpenSlide(slide_path).get_thumbnail(thumbnail_size)

    @staticmethod
    def view_slide_thumbnail(slide_path: str, thumbnail_size: tuple = (256, 256)) -> None:
        """
        View the slide thumbnail for a WSI slide.

        :param slide_path:
        :param thumbnail_size:
        :rtype: None
        """
        plt.imshow(PatchCreator.get_slide_thumbnail(slide_path, thumbnail_size))
        plt.show()

    def generate_patches_from_slide(self, slide_path: str, show_slide_thumbnail: bool = True) -> None:
        """
        Generate patches from slide

        :param slide_path:
        :param show_slide_thumbnail:
        :rtype: None
        """
        print(f"Getting patches for slide: {slide_path}")
        slide_name = slide_path.split("/")[-1].split(".")[0]

        slide = openslide.OpenSlide(slide_path)
        deepzoom = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=self.patch_features['tile_size'],
                                                        overlap=self.patch_features["overlap"])
        plt.imshow(slide.get_thumbnail((256, 256)))
        plt.show()

        print(f"deepzoom levels: {deepzoom.level_tiles}")
        print(f"deepzoom level dimensions: {deepzoom.level_dimensions}")

        level_dimensions = deepzoom.level_tiles[self.patch_features["deepzoom_level"]]
        max_tile_size = (0, 0)

        for column in range(level_dimensions[0]):
            for row in range(level_dimensions[1]):
                plt.axis("off")

                deepzoom.get_tile(self.patch_features["deepzoom_level"], (column, row)).save(
                    f"{self.metadata['output_patch_directory']}/{slide_name}_{level_dimensions}_{column}_{row}.png")
            print("=", end="")
            # figure.show()
        print("")

        # plt.imshow(slide.get_thumbnail((256, 256)))
        # slide.get_thumbnail((8192, 8192))

    # def generate_patches_from_slide_directory(slide_image_directory):
    #     for slide_path in os.listdir(root_directory_full_images):
    #         # slide_path = "Camelyon/Full images/Normal/normal_001.tif"
    #         generate_patches_from_slide(slide_path)

    def generate(self, root_dir_path: str, slide_path: str, dataset_name: str = "Undefined") -> None:
        """
        Generate patches from a WSI slide. Main method.

        :param root_dir_path:
        :param slide_path:
        :param dataset_name:
        :rtype: object
        """
        self.set_dataset_name(dataset_name)
        self.create_output_patch_directory_structure(root_dir_path)
        self.get_current_output_directories()
        self.generate_patches_from_slide(slide_path)

#     %%time
#     greyscale_transform = torchvision.transforms.Compose(
#         [
#          torchvision.transforms.Grayscale(num_output_channels=1)
#         ]
#     )

#     root_directory = f"{patches_dir}/{data_set_name}/{patch_features['tile_size']}/{class_name}"

#     patches_above_threshold = []
#     # tile_size = 278
#     threshold = 0.2

#     patches_accepted = 0

#     # Code based on code at https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html
#     for index, path in enumerate([file for file in os.listdir(root_directory) if file.split(".")[-1] == "png"]):
#       image = Image.open(f"{root_directory}/{path}")
#       greyscale_image = greyscale_transform(image)

#       edges = filters.sobel(greyscale_image)

#       low = 0.1
#       high = 0.35

#       low_threshold = (edges > low).astype(int)
#       high_threshold = (edges > high).astype(int)
#       hysteresis_threshold = filters.apply_hysteresis_threshold(edges, low, high)

#       hysteresis_threshold = high_threshold + hysteresis_threshold

#       if np.count_nonzero(hysteresis_threshold) >= (patch_features["tile_size"]*patch_features["tile_size"]) * threshold:
#         patches_above_threshold.append((path, image))
#         patches_accepted += 1

#     %%time
#     os.makedirs(f"{threshold_patches_dir}/{data_set_name}/{patch_features['tile_size']}/{class_name}", exist_ok=True)

#     for image_info in patches_above_threshold:
#       image_info[1].save(f"{threshold_patches_dir}/{data_set_name}/{patch_features['tile_size']}/{class_name}/{image_info[0]}")
#       # plt.imshow(image)
#       # plt.show()
#     print(len(patches_above_threshold))
#     print([i[0] for i in patches_above_threshold])
