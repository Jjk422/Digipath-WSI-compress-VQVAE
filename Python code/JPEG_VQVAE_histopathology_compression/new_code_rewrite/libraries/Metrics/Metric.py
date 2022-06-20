import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms
from skimage.metrics import structural_similarity as ssim

class Metric:
    def image_diff(image1, image2, ignore_channels=True, pytorch_input_format=False, display_plot_differences=False,
                   display_difference_metrics=True):
        image_difference = image1 - image2 if pytorch_input_format else image1.permute(2, 0, 1) - image2.permute(2, 0,
                                                                                                                 1)

        for index, channel_diff in enumerate(image_difference):
            if ignore_channels:
                cmap = "gray"
            else:
                print(f"- Difference map for {['Red', 'Green', 'Blue', 'Alpha'][index]} channel -")
                cmap = "gray"

            if display_plot_differences:
                plt.imshow(channel_diff, cmap=cmap)
                plt.show()
                # Thresholder from https://stackoverflow.com/questions/36719997/threshold-in-2d-numpy-array/36720130
                image_diff_threshold = ((channel_diff > 0.1) * channel_diff)
                plt.imshow(image_diff_threshold, cmap="gray", vmin=0, vmax=255)
                plt.show()

            if display_difference_metrics:
                print(
                    f"Max: {channel_diff.max()} || Sum: {channel_diff.sum()} || Average (Mean): {channel_diff.mean()}")

            if ignore_channels:
                break

    @staticmethod
    def __calculate_accuracy(original_image, reconstructed_image, use_threshold_limit=False, difference_multiplier=0.1, min_possible_pixel_value=0, max_possible_pixel_value=255):
        if use_threshold_limit is False:
            pixels_within_threshold_limit = (torch.eq(original_image, reconstructed_image))
        else:
            pixels_within_threshold_limit = torch.logical_and(
                torch.gt(reconstructed_image, original_image - ((max_possible_pixel_value - min_possible_pixel_value) * difference_multiplier)),
                torch.lt(reconstructed_image, original_image + ((max_possible_pixel_value - min_possible_pixel_value) * difference_multiplier))
            )

        return torch.div(torch.sum(pixels_within_threshold_limit), original_image.numel())

    @staticmethod
    def calculate_accuracy_tensors(input_values, use_threshold_limit=False, difference_multiplier=0.1, min_possible_pixel_value=0, max_possible_pixel_value=255):
        return Metric.__calculate_accuracy(
            input_values["original_image"],
            input_values["reconstructed_image"],
            use_threshold_limit=use_threshold_limit,
            difference_multiplier=difference_multiplier,
            min_possible_pixel_value=min_possible_pixel_value,
            max_possible_pixel_value=max_possible_pixel_value
        ).item()

    # TODO: Sort out metrics so can either select between minimum and maximum pixel values
    @staticmethod
    def calculate_accuracy_tensors_no_percentage_threshold(input_values, other_args):
        return Metric.__calculate_accuracy(
            input_values["original_image"],
            input_values["reconstructed_image"],
            use_threshold_limit=False,
            difference_multiplier=None,
            min_possible_pixel_value=other_args["min_possible_pixel_value"],
            max_possible_pixel_value=other_args["max_possible_pixel_value"]
        ).item()

    @staticmethod
    def calculate_accuracy_tensors_1_percent_threshold(input_values, other_args):
        return Metric.__calculate_accuracy(
            input_values["original_image"],
            input_values["reconstructed_image"],
            use_threshold_limit=True,
            difference_multiplier=0.01,
            min_possible_pixel_value=other_args["min_possible_pixel_value"],
            max_possible_pixel_value=other_args["max_possible_pixel_value"]
        ).item()

    @staticmethod
    def calculate_accuracy_tensors_5_percent_threshold(input_values, other_args):
        return Metric.__calculate_accuracy(
            input_values["original_image"],
            input_values["reconstructed_image"],
            use_threshold_limit=True,
            difference_multiplier=0.05,
            min_possible_pixel_value=other_args["min_possible_pixel_value"],
            max_possible_pixel_value=other_args["max_possible_pixel_value"]
        ).item()

    @staticmethod
    def calculate_accuracy_tensors_10_percent_threshold(input_values, other_args):
        return Metric.__calculate_accuracy(
            input_values["original_image"],
            input_values["reconstructed_image"],
            use_threshold_limit=True,
            difference_multiplier=0.1,
            min_possible_pixel_value=other_args["min_possible_pixel_value"],
            max_possible_pixel_value=other_args["max_possible_pixel_value"]
        ).item()

    """ Calculate the compression ratio metric [uncompressed_size/compressed_size]"""
    @staticmethod
    def __calculate_compression_ratio(original_image_size: float, compressed_image_size: float):
        return original_image_size / compressed_image_size

    # TODO: possibly rewrite this into a recursive version that can be called from the above function automatically
    @staticmethod
    def calculate_compression_ratio_tensors(input_values, other_args):
        return Metric.__calculate_compression_ratio(
            input_values["original_image"].numel(),
            input_values["compressed_image"].numel()
        )

    """ Calculate the space saving metric [1 - (compressed_size/uncompressed_size)]"""
    @staticmethod
    def __calculate_space_saving(original_image_size, compressed_image_size):
        return 1 - (compressed_image_size / original_image_size)

    # TODO: possibly rewrite this into a recursive version that can be called from the above function automatically
    @staticmethod
    def calculate_space_saving_ratio_tensors(input_values, other_args):
        return Metric.__calculate_space_saving(
            input_values["original_image"].numel(),
            input_values["compressed_image"].numel()
        )

    @staticmethod
    def __calculate_compression_accuracy_metric(original_image, reconstructed_image, compressed_image):
        return 1 + torch.mul(
            Metric.__calculate_space_saving(original_image.numel(), compressed_image.numel()),
            torch.sum(Metric.__calculate_accuracy(original_image, reconstructed_image, use_threshold_limit=0.1)),
        ).item()

    @staticmethod
    def calculate_compression_accuracy_metric_tensors(input_values, other_args):
        return Metric.__calculate_compression_accuracy_metric(
            input_values["original_image"],
            input_values["reconstructed_image"],
            input_values["compressed_image"]
        )

    @staticmethod
    def __calculate_compression_ratio_per_SSIM_metric(original_image, reconstructed_image, compressed_image):
        return torch.div(
            Metric.__calculate_compression_ratio(original_image.numel(), compressed_image.numel()),
            Metric.__calculate_structural_similarity(original_image, reconstructed_image)
        )

    @staticmethod
    def calculate_compression_ratio_per_SSIM_metric_tensors(input_values, other_args):
        return Metric.__calculate_compression_ratio_per_SSIM_metric(
            input_values["original_image"],
            input_values["reconstructed_image"],
            input_values["compressed_image"]
        )

    @staticmethod
    def __calculate_compression_ratio_per_PSNR_metric(original_image, reconstructed_image, compressed_image):
        return torch.div(
            Metric.__calculate_compression_ratio(original_image.numel(), compressed_image.numel()),
            Metric.___calculate_peak_signal_to_noise_ratio(original_image, reconstructed_image)
        )

    @staticmethod
    def calculate_compression_ratio_per_PSNR_metric_tensors(input_values, other_args):
        return Metric.__calculate_compression_ratio_per_PSNR_metric(
            input_values["original_image"],
            input_values["reconstructed_image"],
            input_values["compressed_image"]
        )

    @staticmethod
    def ___calculate_peak_signal_to_noise_ratio(original_image, reconstructed_image, max_possible_pixel_value=255):
        mean_squared_error = torch.mean((original_image - reconstructed_image) ** 2)
        return 20 * torch.log10(max_possible_pixel_value / torch.sqrt(mean_squared_error))

    @staticmethod
    def calculate_peak_signal_to_noise_ratio(input_values, other_args):
        return Metric.___calculate_peak_signal_to_noise_ratio(
            input_values["original_image"],
            input_values["reconstructed_image"],
            max_possible_pixel_value=other_args["max_possible_pixel_value"]
        )

    @staticmethod
    def __calculate_structural_similarity(original_image, reconstructed_image):
        return ssim(
            original_image.permute(1,2,0).cpu().numpy(),
            reconstructed_image.permute(1,2,0).cpu().numpy(),
            multichannel=True
        )

    @staticmethod
    def calculate_structural_similarity(input_values, other_args):
        return Metric.__calculate_structural_similarity(
            input_values["original_image"],
            input_values["reconstructed_image"],
        )

    @staticmethod
    def get_number_of_numerical_array_values_original_image(input_values, other_args):
        return input_values["original_image"].numel()

    @staticmethod
    def get_number_of_numerical_array_values_compressed_representation(input_values, other_args):
        return input_values["compressed_image"].numel()

class MetricController:
    def __init__(self,
                 metrics_to_apply={
                     "Exact pixel accuracy": Metric.calculate_accuracy_tensors_no_percentage_threshold,
                     "1% threshold pixel accuracy": Metric.calculate_accuracy_tensors_1_percent_threshold,
                     "5% threshold pixel accuracy": Metric.calculate_accuracy_tensors_5_percent_threshold,
                     "10% threshold pixel accuracy": Metric.calculate_accuracy_tensors_10_percent_threshold,
                     "Compression ratio": Metric.calculate_compression_ratio_tensors,
                     "Space saving ratio": Metric.calculate_space_saving_ratio_tensors,
                     "Compression vs accuracy metric": Metric.calculate_compression_accuracy_metric_tensors,
                     "Peak signal to noise ratio": Metric.calculate_peak_signal_to_noise_ratio,
                     "Structural similarity": Metric.calculate_structural_similarity,
                     "Original Image Size (num of numerical array values)": Metric.get_number_of_numerical_array_values_original_image,
                     "Compressed Image Size (num of numerical array values)": Metric.get_number_of_numerical_array_values_compressed_representation,
                     "Compression_ratio_per_SSIM": Metric.calculate_compression_ratio_per_SSIM_metric_tensors,
                     "Compression_ratio_per_PSNR": Metric.calculate_compression_ratio_per_PSNR_metric_tensors,
                 },
                 minimum_pixel_value = 0,
                 maximum_pixel_value = 255,
                 ):
        self.metrics_to_apply = metrics_to_apply
        self.other_args = {"min_possible_pixel_value": minimum_pixel_value, "max_possible_pixel_value": maximum_pixel_value}
        self.inputs = {}

    # def apply_metrics(self, inputs):
    #     [metric(inputs) for metric in self.metrics_to_apply]

    def apply_metrics(self):
        # print(self.inputs)
        return {metric_name: metric(self.inputs, self.other_args) for metric_name, metric in self.metrics_to_apply.items()}

    def apply_metrics_to_batch(self, original_images, reconstructed_images, compressed_images):
        results = {}

        for metric_name, metric in self.metrics_to_apply.items():
            cumulative_total, current_index = 0, 0
            minimum, maximum = None, None
            try:
                for i in range(len(original_images)):
                    current_index += 1
                    self.set_original_image(original_images[i])
                    self.set_reconstructed_image(reconstructed_images[i])
                    self.set_compressed_image(compressed_images[i])
                    metric_result = metric(self.inputs, self.other_args)
                    cumulative_total = cumulative_total + metric_result
                    minimum = min(result for result in [metric_result, minimum] if result is not None)
                    maximum = max(result for result in [metric_result, maximum] if result is not None)

                    # running_average = cumulative_total/(i + 1)
            # results[metric_name] = {"Total": cumulative_total, "Average": cumulative_total/len(original_images)}
            #     results[metric_name] = {"Average": cumulative_total/len(original_images), "Minimum": minimum, "Maximum": maximum}
                results[metric_name] = cumulative_total/len(original_images)
            except IndexError:
                # results[metric_name] = {"Average": cumulative_total/(i+1), "Minimum": minimum, "Maximum": maximum}
                results[metric_name] = cumulative_total/(i+1)
                # return results
        return results


    def set_input_value(self, key, value):
        self.inputs[key] = value

    def set_original_image(self, original_image):
        self.set_input_value("original_image", original_image)

    def set_reconstructed_image(self, reconstructed_image):
        self.set_input_value("reconstructed_image", reconstructed_image)

    def set_compressed_image(self, compressed_image):
        self.set_input_value("compressed_image", compressed_image)

    def __set_min_max_pixel_values(self, value_to_set, value):
        self.other_args[value_to_set] = value

    def set_minimum_pixel_value(self, value):
        self.__set_min_max_pixel_values("min_possible_pixel_value", value)

    def set_maximum_pixel_value(self, value):
        self.__set_min_max_pixel_values("max_possible_pixel_value", value)