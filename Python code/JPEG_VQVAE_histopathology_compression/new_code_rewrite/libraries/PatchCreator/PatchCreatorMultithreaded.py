# import Constants

import openslide
import openslide.deepzoom

import matplotlib.pyplot as plt
from skimage import filters
from PIL import Image
import numpy as np

import torch
import torchvision
import os
from new_code_rewrite.libraries.Utils.ThreadManager import ThreadManagerTQDM
from new_code_rewrite.libraries.Utils.ProgressBar import ProgressBar

class PatchCreator():
    def __init__(
            self,
            dataset_name="No Dataset Name",
            tile_size=512,
            overlap=0,
            deepzoom_level=15,
            progress_bar=None,
            display_progress_bar=True,
    ) -> None:
        """

        :rtype: None
        """
        self.metadata = {
            "dataset_name": "No Dataset Name",
        }

        self.patch_features = {
            "tile_size": tile_size,
            "overlap": overlap,
            "deepzoom_level": deepzoom_level,
        }

        self.display_progress_bar = display_progress_bar
        self.progress_bar = progress_bar if progress_bar is not None else ProgressBar

    def set_metadata(self, key, value):
        self.metadata[key] = value

    def set_dataset_name(self, name):
        self.set_metadata("dataset_name", name)

    def set_class_name(self, name):
        self.set_metadata("class_name", name)

    def set_output_patch_root_directory(self, directory_path):
        self.set_metadata("output_patch_root_directory", directory_path)

    # Internal usage method used within the create_output_patch_directory_structure method
    def __set_output_patch_directory(self, directory_path):
        self.set_metadata("output_patch_directory", directory_path)

    def __set_output_thresholded_patch_directory(self, directory_path):
        self.set_metadata("output_thresholded_patch_directory", directory_path)

    def get_dataset_name(self):
        return self.metadata["dataset_name"]

    def get_class_name(self):
        return self.metadata["class_name"]

    def get_output_patch_root_directory(self):
        return self.metadata['output_patch_root_directory']

    def get_output_patch_directory(self):
        return self.metadata['output_patch_directory']

    def get_output_thresholded_patch_directory(self):
        return self.metadata['output_thresholded_patch_directory']

    def get_current_output_directories(self):
        print("--- Current output_directories ---")
        print(f"Root_directory: {self.get_output_patch_root_directory()}")
        print(f"Patch_directory: {self.get_output_patch_directory()}")
        try:
            print(f"Thresholded_patch_directory: {self.get_output_thresholded_patch_directory()}")
        except KeyError:
            pass

    def create_output_patch_directory_structure(self, output_patch_root_directory, class_name="undefined_class"):
        self.set_output_patch_root_directory(output_patch_root_directory)
        self.__set_output_patch_directory(
            f"{output_patch_root_directory}/{self.metadata['dataset_name']}/{self.patch_features['tile_size']}/{class_name}")
        # self.__set_output_thresholded_patch_directory(None)

        # class_name = "Normal"
        # image_type = "Training"
        #         output_patch_directory = f"{patches_dir}/{data_set_name}/{patch_features['tile_size']}/{class_name}"
        #         root_directory_full_images = f"{slides_dir}"
        #         os.makedirs(output_patch_root_directory, exist_ok=True)
        os.makedirs(
            f"{self.metadata['output_patch_root_directory']}/{self.metadata['dataset_name']}/{self.patch_features['tile_size']}/{class_name}",
            exist_ok=True)

    @staticmethod
    def get_slide_thumbnail(slide_path, thumbnail_size=(256, 256)):
        return openslide.OpenSlide(slide_path).get_thumbnail(thumbnail_size)

    @staticmethod
    def view_slide_thumbnail(slide_path, thumbnail_size=(256, 256)):
        plt.imshow(PatchCreator.get_slide_thumbnail(slide_path, thumbnail_size))
        plt.show()

    def extract_and_save_tile(self, argument_iterator):
        (column, row) = argument_iterator

        self.deepzoom.get_tile(self.patch_features["deepzoom_level"], (column, row)).save(
            f"{self.metadata['output_patch_directory']}/{self.slide_name}_{self.level_dimensions}_{column}_{row}.png")


    def generate_patches_from_slide(self, slide_path, show_slide_thumbnail=True):
        print(f"Getting patches for slide: {slide_path}")
        self.slide_name = slide_path.split("/")[-1].split(".")[0]

        slide = openslide.OpenSlide(slide_path)
        self.deepzoom = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=self.patch_features['tile_size'],
                                                        overlap=self.patch_features["overlap"])

        if show_slide_thumbnail:
            plt.imshow(slide.get_thumbnail((256, 256)))
            plt.show()

        print(f"deepzoom levels: {self.deepzoom.level_tiles}")
        print(f"deepzoom level dimensions: {self.deepzoom.level_dimensions}")

        self.level_dimensions = self.deepzoom.level_tiles[self.patch_features["deepzoom_level"]]
        max_tile_size = (0, 0)

        plt.axis("off")

        argument_iterator = [(column, row) for row in range(self.level_dimensions[1]) for column in range(self.level_dimensions[0])]

        ThreadManagerTQDM(self.extract_and_save_tile, argument_iterator, "Extracting and saving patch tiles")

        # plt.imshow(slide.get_thumbnail((256, 256)))
        # slide.get_thumbnail((8192, 8192))

    # def generate_patches_from_slide_directory(slide_image_directory):
    #     for slide_path in os.listdir(root_directory_full_images):
    #         # slide_path = "Camelyon/Full images/Normal/normal_001.tif"
    #         generate_patches_from_slide(slide_path)

    def generate(self, slide_path, patch_output_directory, dataset_name="Undefined", class_name="undefined_class", show_slide_thumbnail=False):
        self.set_dataset_name(dataset_name)
        self.create_output_patch_directory_structure(patch_output_directory, class_name=class_name)
        self.get_current_output_directories()
        self.generate_patches_from_slide(slide_path, show_slide_thumbnail=show_slide_thumbnail)

    # def threshold_patches(self):


    def threshold(self, patch_directory_path, threshold_patches_directory_path, threshold=0.1, dataset_name="Undefined", class_name="undefined_class"):
        self.set_dataset_name(dataset_name)
        self.set_class_name(class_name)
        self.__set_output_patch_directory(f"{patch_directory_path}/{self.get_dataset_name()}/{self.patch_features['tile_size']}/{self.get_class_name()}")
        self.__set_output_thresholded_patch_directory(f"{threshold_patches_directory_path}/{dataset_name}/{self.patch_features['tile_size']}/{class_name}")
        self.__threshold = threshold

        self.greyscale_transform = torchvision.transforms.Compose(
                [
                 torchvision.transforms.Grayscale(num_output_channels=1)
                ]
            )

        # root_directory = f"{patch_directory_path}/{self.get_dataset_name()}/{self.patch_features['tile_size']}/{self.get_class_name()}"

        # patches_above_threshold = []

        # patches_accepted = 0

        png_filenames = [file for file in os.listdir(self.get_output_patch_directory()) if file.split(".")[-1] == "png"]

        patches_above_threshold = ThreadManagerTQDM(self.threshold_image, png_filenames, "Thresholding images")

        os.makedirs(self.get_output_thresholded_patch_directory(), exist_ok=True)

        argument_iterator = [patch for patch in patches_above_threshold if patch is not None]
        patches_accepted = len(argument_iterator)
        ThreadManagerTQDM(self.save_image, argument_iterator, "Saving thresholded patches")
        return patches_accepted

        # with ProgressBar(total_value=len(png_filenames), description="Processing png files") as progress_bar:
            # Code based on code at https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html
            # for index, path in enumerate(png_filenames):

            # for image_info in patches_above_threshold:
            #   image_info[1].save(f"{threshold_patches_directory_path}/{dataset_name}/{self.patch_features['tile_size']}/{class_name}/{image_info[0]}")

            # progress_bar.update()

    def threshold_image(self, argument_iterator):
        image_path = argument_iterator
        image = Image.open(f"{self.get_output_patch_directory()}/{image_path}")
        greyscale_image = self.greyscale_transform(image)

        edges = filters.sobel(greyscale_image)

        low = 0.1
        high = 0.35

        low_threshold = (edges > low).astype(int)
        high_threshold = (edges > high).astype(int)
        hysteresis_threshold = filters.apply_hysteresis_threshold(edges, low, high)

        hysteresis_threshold = high_threshold + hysteresis_threshold

        if np.count_nonzero(hysteresis_threshold) >= (self.patch_features["tile_size"]*self.patch_features["tile_size"]) * self.__threshold:
            return (image_path, image)
        else:
            return None

    def save_image(self, argument_iterator):
        image_path, image = argument_iterator
        image.save(f"{self.get_output_thresholded_patch_directory()}/{image_path}")

    # def threshold_multithreaded_tqdm(self, argument_iterator):
    #     ThreadManagerTQDM(save_image, argument_iterator)

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
