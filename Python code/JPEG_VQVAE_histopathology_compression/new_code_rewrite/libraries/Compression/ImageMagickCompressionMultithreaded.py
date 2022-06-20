import os
import shutil
from PIL import Image
import torch.tensor
from torchvision.transforms import ToTensor

from threading import Thread

from new_code_rewrite.libraries.Metrics.Metric import MetricController
from new_code_rewrite.libraries.Utils.ThreadManager import ThreadManagerTQDM

class ImageMagickCompression:
    def __init__(self, logger=None, remove_all_images_after_compression=True, display_progress_bar=True):
        self.logger = logger
        self.metric_controller = MetricController(minimum_pixel_value = -1, maximum_pixel_value = 1)
        self.remove_all_images_after_compression = remove_all_images_after_compression
        self.display_progress_bar = True

    # def __del__(self):
    #     self.logger.close()

    @staticmethod
    def __parse_command_to_run(filename, original_images_directory_path, jpeg_compression_directory_path, percentage_quality, template_command_to_run):
        command = template_command_to_run.replace('{source_image_filepath}', f'"{original_images_directory_path}/{filename}"')
        command = command.replace('{percentage_quality}', percentage_quality)
        command = command.replace('{result_image_filepath}', f'"{jpeg_compression_directory_path}/{percentage_quality}%/jpeg_compressed_{percentage_quality}%_{filename}.jpeg"')
        return command

    @staticmethod
    def __get_images_to_compress_from_file(original_images_directory_path):
        return os.listdir(original_images_directory_path)

    @staticmethod
    def __open_image_as_tensor(filename):
        with Image.open(filename) as image:
            return ToTensor()(image)

    @staticmethod
    def __open_image_as_tensor_thread_method(filename, return_result_list, return_result_thread_index):
        return_result_list[return_result_thread_index] = ImageMagickCompression.__open_image_as_tensor(filename)

    @staticmethod
    def __copy_file(argument_tuple):
        original_image_file_path, output_image_file_path, transform = argument_tuple
        with Image.open(original_image_file_path, "r") as image:
            transform(image).save(output_image_file_path)

    def run_generic_compression_command(self, original_images_directory_path, jpeg_compression_directory_path, template_command, compression_percentages=None, tag="Compressed Images using generic Imagemagick jpeg compression", transform=None, maximum_logged_images_saved=16):
        percentages_to_test = [str(i) for i in range(0, 101, 5)] if compression_percentages is None else compression_percentages

        print(f"--- Compressing images using Imagemagick compression ---")
        print(f"Compression percentages to test: {compression_percentages}")
        for compression_percentage in percentages_to_test:
            print(f"Running compression with {compression_percentage}% compression")
            os.makedirs(f"{jpeg_compression_directory_path}/{compression_percentage}%", exist_ok=True)
            try:
                images = ImageMagickCompression.__get_images_to_compress_from_file(original_images_directory_path)

                if transform is not None:
                    os.makedirs(f"{jpeg_compression_directory_path}/{compression_percentage}%/tmp", exist_ok=True)

                    argument_iterator = [(f"{original_images_directory_path}/{image}", f"{jpeg_compression_directory_path}/{compression_percentage}%/tmp/{image}", transform) for image in images]
                    ThreadManagerTQDM(ImageMagickCompression.__copy_file, argument_iterator, "Creating ImageMagick commands", display_progress_bar=self.display_progress_bar)

                    # for index, image in enumerate(images):
                    #     transform(Image.open(f"{original_images_directory_path}/{image}", "r")).save(f"{jpeg_compression_directory_path}/{compression_percentage}%/tmp/{image}")
                    original_images_directory_path = f"{jpeg_compression_directory_path}/{compression_percentage}%/tmp"

                compression_commands_to_run = [ImageMagickCompression.__parse_command_to_run(filename, original_images_directory_path, jpeg_compression_directory_path, compression_percentage, template_command) for filename in images]
                print(f"Example command: {compression_commands_to_run[0]}")

                # TODO: Move threading into concurrent.futures ThreadPoolExtractor class
                ThreadManagerTQDM(os.system, compression_commands_to_run, "Running compression commands", display_progress_bar=self.display_progress_bar)

                # threads = []
                # for index, command in enumerate(compression_commands_to_run):
                #     threads.append(Thread(target=os.system, args=(command,)))
                #     threads[index].start()
                #
                # [thread.join() for thread in threads]

                if self.logger is not None:
                    print("Saving images to logger")
                    image_filenames_to_log = ImageMagickCompression.__get_images_to_compress_from_file(original_images_directory_path)[0:maximum_logged_images_saved:1]
                    # TODO: Move threading into concurrent.futures ThreadPoolExtractor class
                    threads = []
                    # tensorimages = [None] * len(image_filenames_to_log)
                    tensorimages = [None] * maximum_logged_images_saved if maximum_logged_images_saved < len(image_filenames_to_log) else [None] * len(image_filenames_to_log)
                    for index, filename in enumerate(image_filenames_to_log):
                        threads.append(Thread(target=ImageMagickCompression.__open_image_as_tensor_thread_method, args=(f"{jpeg_compression_directory_path}/{compression_percentage}%/jpeg_compressed_{compression_percentage}%_{filename}.jpeg", tensorimages, index)))
                        threads[index].start()

                    [thread.join() for thread in threads]

                    # Add tensorimages (images that have been converted to torch tensors) to the logger
                    self.logger.add_images(tensorimages, step=compression_percentage, tag=tag)

                    # # TODO: Add threading here
                    metric_results = {}
                    for index, image_filename in enumerate(image_filenames_to_log):
                        self.metric_controller.set_original_image(ImageMagickCompression.__open_image_as_tensor(f"{jpeg_compression_directory_path}/{compression_percentage}%/tmp/{image_filename}"))
                        # test = Image.open(f"{jpeg_compression_directory_path}/{compression_percentage}%/jpeg_compressed_{compression_percentage}%_{image_filename}.jpeg")
                        self.metric_controller.set_compressed_image(ToTensor()(Image.open(f"{jpeg_compression_directory_path}/{compression_percentage}%/jpeg_compressed_{compression_percentage}%_{image_filename}.jpeg")))
                        self.metric_controller.set_reconstructed_image(tensorimages[index])
                        # metric_results = self.metric_controller.apply_metrics()
                        metric_results.update(self.metric_controller.apply_metrics())

                    for metric_name, metric_result in metric_results.items():
                        self.logger.add_scalar(f"JPEG 2000 Compression Average Metrics/{metric_name}", sum(metric_result)/len(metric_result) if isinstance(metric_result, list) else metric_result, compression_percentage)

                print()
            except Exception as exc:
                # TODO: This really should print to sys.stderr
                # print(str(exc))
                print(f'"{exc.__class__.__name__}" exception occured, quitting current compression task')
                shutil.rmtree(f"{jpeg_compression_directory_path}")
                return None

                # for index, filename in enumerate(image_filenames_to_log):
                #     self.metric_controller.set_original_image(ImageMagickCompression.__open_image_as_tensor(image_filenames_to_log))
                #     with image.open(f"{jpeg_compression_directory_path}/{compression_percentage}%/jpeg_compressed_{compression_percentage}%_{filename}.jpeg"):
                #         self.metric_controller.set_compressed_image(tensorimages[0])
                #         self.metric_controller.set_reconstructed_image(tensorimages[0])


                # original_images_filenames = ImageMagickCompression.__get_images_to_compress_from_file(original_images_directory_path)
                # original_images_list = [ImageMagickCompression.__open_image_as_tensor(filename) for filename in original_images_filenames]
                # compressed_images_list = [ImageMagickCompression.__open_image_as_tensor(f"{jpeg_compression_directory_path}/{compression_percentage}%/jpeg_compressed_{compression_percentage}%_{filename}.jpeg") for filename in image_filenames_to_log]
                #
                #
                # metric_results = self.metric_controller.apply_metrics_to_batch(original_images_list, reconstructed_images, compressed_images)
                #
                # for metric_name, metric_result in metric_results.items():
                #     self.logger.add_scalar(f"Training Epoch Average Metrics - {metric_name}", metric_result, epoch)

        if self.remove_all_images_after_compression:
            # for compression_percentage in compression_percentages:
            shutil.rmtree(f"{jpeg_compression_directory_path}")
        else:
            # TODO: Move threading into concurrent.futures ThreadPoolExtractor class
            threads = []
            for index, compression_percentage in enumerate(compression_percentages):
                threads.append(Thread(target=shutil.rmtree, args=(f"{jpeg_compression_directory_path}/{compression_percentage}%/tmp",)))
                threads[index].start()

            [thread.join() for thread in threads]

    def run_chroma_channel_reduction_jpeg(self, original_images_directory_path, jpeg_compression_directory_path, compression_percentages=None, transform=None):
        command_to_run = "magick {source_image_filepath} -strip -interlace Plane -sampling-factor 4:2:0 -quality {percentage_quality}% {result_image_filepath}"

        self.run_generic_compression_command(original_images_directory_path, jpeg_compression_directory_path, command_to_run, tag="JPEG chroma channel reduction compression args: [-strip -interlace Plane -sampling-factor 4:2:0]", compression_percentages=compression_percentages, transform=transform)