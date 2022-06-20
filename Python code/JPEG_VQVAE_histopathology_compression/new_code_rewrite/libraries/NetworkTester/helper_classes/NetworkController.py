import os
import traceback

import torch
from torch import optim

from new_code_rewrite.libraries.NetworkTester.helper_classes import ImageDisplay
from new_code_rewrite.libraries.Metrics.Logger import Logger

from new_code_rewrite.libraries.Utils.ProgressBar import ProgressBar

from new_code_rewrite.libraries.Utils.ThreadManager import ThreadManager, ThreadManagerTQDM

from new_code_rewrite.libraries.Metrics.Metric import MetricController

class NetworkController():
    def __init__(self, network, dataset, secondary_network=None, device='cpu', optimiser=None, loss_report_interval=100, root_directory = "", previous_datetime=None, log_directory=None, display_progress_bar=True, logger=None, number_of_partially_trained_models_to_save=2, things_to_log=["images", "scalars", "embeddings"]):
        self.loss_report_interval = loss_report_interval
        self.network = network.to(device)
        self.secondary_network = secondary_network
        self.device = device
        self.network.set_device(self.device)
        self.things_to_log = things_to_log
        self.logger = Logger(self.network, log_directory=log_directory) if logger is None else logger
        # TODO: Parametrise the metric controller minimum and maximum pixel values
        self.metric_controller = MetricController(minimum_pixel_value = -1, maximum_pixel_value = 1)
        self.display_progress_bar = display_progress_bar
        self.number_of_partially_trained_models_to_save = number_of_partially_trained_models_to_save

        # self.datetime = time.strftime("%Y_%m_%d-%H_%M_%S") if previous_datetime is None else previous_datetime
        # self.root_directory = f"{root_directory}/{network.__class__.__name__}-{dataset.__class__.__name__}-{self.datetime}"
        self.root_directory = f"{root_directory}_{dataset.__class__.__name__}"

        os.makedirs(self.root_directory, exist_ok=True)
        os.makedirs(f"{self.root_directory}/partially_trained_models", exist_ok=True)

        # if not os.path.isdir(self.root_directory):
        #     os.mkdir(self.root_directory)

        # if not os.path.isdir(f"{self.root_directory}/partially_trained_models"):
        #     os.mkdir(f"{self.root_directory}/partially_trained_models")

        self.write_network_architecture_to_file()

        # Network training information
        self.loss = 0

        if optimiser is None:
            optimiser = optim.Adam(self.network.parameters())
        self.optimiser = optimiser
        self.dataset = dataset
        self.train_loader = self.dataset.get_train_loader()
        self.test_loader = self.dataset.get_test_loader()

    def train(self, epochs=10, use_old_log_generator=False, secondary_network_to_train=None):
        self.network.train()
        if secondary_network_to_train is not None:
            self.secondary_network = secondary_network_to_train

        # self.network = self.network.to(self.device)
        print(f"--- Current network: {self.network.__class__.__name__}{'' if secondary_network_to_train is None else f'-->{secondary_network_to_train.__class__.__name__}'} ---")
        if use_old_log_generator:
            print(f"Creating log file at {self.root_directory}/log_file.txt")
            open(f"{self.root_directory}/log_file.txt", "w+").close()
        print("Training network")
        # TODO: Find a better way of doing batch iteration documentation/recording for logging
        batch_iter = 0
        progress_bar = ProgressBar(epochs, description="Epoch progress", position=0, display_progress_bar=self.display_progress_bar)

        number_of_saved_partially_trained_models = 0
        for epoch in range(epochs):
            self.network.train()
            # print("=", end="")
            # This doesn't seem to be working so assigning directly
            self.network.set_epoch(epoch)
            total_loss = 0

            running_loss = 0
            progress_bar_batch = ProgressBar(len(self.train_loader), description="Batch progress", position=1, leave=False, display_progress_bar=self.display_progress_bar)
            # progress_bar_batch.reset()
            for batch_index, batch in enumerate(self.train_loader):
                epoch_loss = []

                images, classifications = batch
                images = images.to(self.device)
                classifications = classifications.to(self.device)
                # images.to(self.device)

                # Zero parameter gradients
                self.optimiser.zero_grad()

                if secondary_network_to_train is not None:
                    self.network.eval()
                    with torch.no_grad():
                        encoded_images = self.network.encode(images)
                    self.secondary_network.set_batch_data(images, classifications, encoded_images=encoded_images)
                    self.network.train()
                    loss = self.secondary_network.loss_function(self.secondary_network.run_batch())
                else:
                    self.network.set_batch_data(images, classifications)
                    loss = self.network.loss_function(self.network.run_batch())

                # reconstructed_images = self.network(images)

                # loss = self.network.loss_function(self.network.run_batch())

                # Backpropagate loss
                loss.backward()
                self.optimiser.step()

                self.loss = loss

                total_loss += self.loss.item()

                #
                running_loss += loss.item()
                if batch_index % self.loss_report_interval == 0:
                    if use_old_log_generator:
                        log_line = f"Epoch: [{epoch}/{epochs}] || Batch: {batch_index:03d} || Complete: {(batch_index / len(self.train_loader) * 100):4.1f}% || Loss: {loss.item()/len(images):.{3}f}"
                        print(log_line)
                        with open(f"{self.root_directory}/log_file.txt", "a") as file:
                            file.write(log_line)
                            file.write("\n")

                    if "scalars" in self.things_to_log:
                        self.logger.add_scalar("Training Batch/Running Loss", running_loss, batch_iter)
                        self.logger.add_scalar("Training Batch/Average Loss", loss.item()/len(images), batch_iter)
                        batch_iter += 1

                    running_loss = 0

                    # epoch_loss.append(loss.item()/len(images))

                progress_bar_batch.update(1)
            progress_bar_batch.close()

            with torch.no_grad():
                self.network.eval()

                if "images" in self.things_to_log:
                    self.logger.add_images(images, epoch, tag=f"Training dataset/{self.dataset.__class__.__name__}/Original images")

                    # TODO: Fix input parameter logging selection
                    self.logger.add_images(self.network(images), epoch, tag=f"Training dataset/{self.dataset.__class__.__name__}/Reconstructed images")

                if "scalars" in self.things_to_log:
                    # # TODO: rewrite the average loss function to take into account the average per dataset rather than just the average for the cifar10 dataset
                    # self.logger.add_loss_value(total_loss/(80000/128), "training_average_loss", epoch)

                    self.logger.add_scalar("Training Epoch/Average Loss", total_loss/len(self.train_loader), epoch)

                    self.logger.add_scalar("Training Epoch/Total Loss", total_loss, epoch)
                    # TODO: rewrite the average loss function to take into account the average per dataset rather than just the average for the cifar10 dataset
                    # self.logger.add_scalar("Training Epoch Average Loss", total_loss/(80000/128), epoch)
                    self.logger.add_scalar("Training Epoch/Last Loss", loss, epoch)

                    # TODO: Make this function work for any number of args
                    metric_results = self.network.calculate_metrics(images, self.metric_controller)

                    for metric_name, metric_result in metric_results.items():
                        if self.secondary_network is not None:
                            self.logger.add_scalar(f"{self.network.__class__.__name__}/{self.secondary_network.__class__.__name__}/Training Epoch/Average Metrics/{metric_name}", metric_result, epoch)
                        else:
                            self.logger.add_scalar(f"Training Metrics For Each Epoch/Average Metrics/{metric_name}", metric_result, epoch)

                    if "embeddings" in self.things_to_log:
                        try:
                            embeddings = self.network.calculate_embeddings(images)
                            for embedding_name, embedding in embeddings.items():
                                self.logger.add_embeddings(embedding, epoch, tag=f"training_embeddings/embedding-{embedding_name}", metadata=[], label_img=[], metadata_header=[])
                                # self.logger.add_embeddings(self.network.embedding_top.embedding_table.weight, epoch, tag="training_embeddings/top", metadata=[], label_img=[], metadata_header=[])
                                # self.logger.add_embeddings(self.network.embedding_bottom.embedding_table.weight, epoch, tag="training_embeddings/bottom", metadata=[], label_img=[], metadata_header=[])
                        except Exception as exception:
                            traceback.print_exc(exception)
                            print(f"Exception occurred when logging embeddings, looks like the network {self.network.__class__.__name__} may not support embedding logging.")

            # Check for nan values
            if running_loss != running_loss:
                print("nan loss reached")
                break

            # # Temp code to see what networks are doing need to change to framework compatible methods/classes
            # if self.network.epoch == 0:
            #     try:
            #         ImageDisplay.Image.display_image_subplot(12, 12, next(self.dataset.get_test_loader_iter())[0],
            #                                                  no_colour_channels=(self.dataset.image_dim[2] == 1),
            #                                                  cmap="gray",
            #                                                  remove_axes=True,
            #                                                  display=False,
            #                                                  save_path=f"{self.root_directory}/origional_images_test_sample"
            #                                                  )
            #     except TypeError:
            #         pass
            #
            # if self.network.epoch % 5 == 0:
            #     predicted_output = self.predict(next(self.dataset.get_test_loader_iter())[0], collapse_channels=False)
            #     reconstructed_images = predicted_output[0].detach().cpu()
            #
            #     try:
            #         ImageDisplay.Image.display_image_subplot(12, 12,
            #                                                  reconstructed_images.reshape(-1, self.dataset.image_dim[2],
            #                                                                               self.dataset.image_dim[0],
            #                                                                               self.dataset.image_dim[1]),
            #                                                  no_colour_channels=(self.dataset.image_dim[2] == 1),
            #                                                  cmap="gray",
            #                                                  remove_axes=True,
            #                                                  display=False,
            #                                                  save_path=f"{self.root_directory}/reconstucted_images_test_sample_training"
            #                                                  )
            #     except TypeError:
            #         pass

            number_of_saved_partially_trained_models += 1
            if self.secondary_network is not None:
                self.save(f"{self.root_directory}/partially_trained_models/partially_trained_model_{self.network.epoch}_epochs_{self.network.__class__.__name__}/{self.network.epoch}_{self.secondary_network.__class__.__name__}.pt")
            else:
                self.save(f"{self.root_directory}/partially_trained_models/partially_trained_model_{self.network.epoch}_epochs_{self.network.__class__.__name__}.pt")
            progress_bar.update(1)

            if number_of_saved_partially_trained_models > self.number_of_partially_trained_models_to_save:
                if self.secondary_network is not None:
                    os.remove(f"{self.root_directory}/partially_trained_models/partially_trained_model_{self.network.epoch - self.number_of_partially_trained_models_to_save}_epochs_{self.network.__class__.__name__}/{self.network.epoch}_{self.secondary_network.__class__.__name__}.pt")
                else:
                    os.remove(f"{self.root_directory}/partially_trained_models/partially_trained_model_{self.network.epoch - self.number_of_partially_trained_models_to_save}_epochs_{self.network.__class__.__name__}.pt")
                number_of_saved_partially_trained_models -= 1
        # print()
        if self.secondary_network is not None:
            self.save(f"{self.root_directory}/finished_model_{self.network.epoch}_epochs_{self.network.__class__.__name__}/{self.network.epoch}_{self.secondary_network.__class__.__name__}.pt")
        else:
            self.save(f"{self.root_directory}/finished_model_{self.network.epoch}_epochs_{self.network.__class__.__name__}.pt")
        progress_bar.close()



    # def __test_batch_thread_manager_tqdm(self, data):
    #     with torch.no_grad():
    #         batch_images, batch_index, dataset = data
    #         self.__test_batch_log_images(batch_images, batch_index, dataset)
    #         self.__test_batch_log_metrics(batch_images, batch_index, dataset)
    #         # return (self.__test_batch_log_images(batch_images, batch_index, dataset), self.__test_batch_log_metrics(batch_images, batch_index, dataset))

    def __test_batch_thread_manager_tqdm(self, argument_iterator, dataset, step_offset=0):
        with torch.no_grad():
            # for thread_index, (batch_images, network_image_output) in enumerate(ThreadManagerTQDM(self.__test_batch_log_images_iterator, argument_iterator, progress_bar_description="Batch image testing", display_progress_bar=self.display_progress_bar)):
            #     # saving_progress_bar = ProgressBar(total_value=len(batch_images), description="")
            #     batch_images = batch_images
            #     network_image_output = network_image_output
            #     self.logger.add_images(batch_images, batch_index[thread_index], tag=f"Testing dataset/{dataset[thread_index].__class__.__name__}/Original images")
            #     self.logger.add_images(self.network(batch_images), batch_index[thread_index], tag=f"Testing dataset:/{dataset[thread_index].__class__.__name__}/Reconstructed images")

            ### Add images to tensorboardX logger ###
            if "images" in self.things_to_log:
                with ProgressBar(total_value=len(argument_iterator), description="Logging test image results", position=0) as saving_results_progress_bar:
                    for index, (batch_images, output_batch_images) in enumerate(ThreadManagerTQDM(self.__test_batch_log_images_iterator, argument_iterator, progress_bar_description="Batch image testing", display_progress_bar=self.display_progress_bar, progress_bar_position=1)):
                        self.logger.add_images(batch_images, step_offset + index, tag=f"Testing dataset/{dataset.__class__.__name__}/Original images")
                        self.logger.add_images(output_batch_images, step_offset + index, tag=f"Testing dataset:/{dataset.__class__.__name__}/Reconstructed images")
                        # self.logger.add_images(batch_images, index, tag=f"Testing dataset/{dataset[index].__class__.__name__}/Original images")
                        # self.logger.add_images(output_batch_images, index, tag=f"Testing dataset:/{dataset[index].__class__.__name__}/Reconstructed images")
                        saving_results_progress_bar.update(1)

            # for index, (batch_images, output_batch_images) in enumerate(self.shared_list_results):
            #     self.logger.add_images(batch_images, index, tag=f"Testing dataset/{dataset[index].__class__.__name__}/Original images")
            #     self.logger.add_images(output_batch_images, index, tag=f"Testing dataset:/{dataset[index].__class__.__name__}/Reconstructed images")

            # argument_iterator = zip(batch_images, batch_index, dataset, [i for i in range(len(argument_iterator))])

            ### Add metrics to tensorboardX logger ###
            if "scalars" in self.things_to_log:
                with ProgressBar(total_value=len(argument_iterator), description="Logging test average metric results", position=0) as saving_results_progress_bar:
                # # {self.logger.add_scalar(f"Testing Average Metrics (Batch number)/{metric.keys()[thread_index]}/{dataset[thread_index].__class__.__name__}", metric.values()[thread_index], batch_index[thread_index]) for thread_index, metric in enumerate(ThreadManagerTQDM(self.__test_batch_log_metrics_iterator, argument_iterator, progress_bar_description="Batch metrics logging", display_progress_bar=self.display_progress_bar))}
                    for index, batch_results in enumerate(ThreadManagerTQDM(self.__test_batch_log_metrics_iterator, argument_iterator, progress_bar_description="Batch metrics testing", display_progress_bar=self.display_progress_bar, progress_bar_position=1)):
                        for metric_name, metric_result in batch_results.items():
                            self.logger.add_scalar(f"Testing Average Metrics (Batch number)/{metric_name}/{dataset.__class__.__name__}", metric_result, step_offset + index)
                            # self.logger.add_scalar(f"Testing Average Metrics (Batch number)/{metric_name}/{dataset[thread_index].__class__.__name__}", metric_result, batch_index[thread_index])
                        saving_results_progress_bar.update(1)

            # ThreadManagerTQDM(self.__test_batch_log_metrics_iterator, argument_iterator, progress_bar_description="Batch metrics testing", display_progress_bar=self.display_progress_bar)

            ### Add embeddings to tensorboardX logger ###
            if "embeddings" in self.things_to_log:
                try:
                    # with ProgressBar(total_value=len(argument_iterator), description="Logging embeddings", position=0) as saving_results_progress_bar:
                    #     self.logger.add_embeddings(self.network.embedding_top.embedding_table.weight, 0, tag="testing_embeddings/top", metadata=[], label_img=self.network.embedding_top.weight, metadata_header=[])
                    #     self.logger.add_embeddings(self.network.embedding_bottom.embedding_table.weight, 0, tag="testing_embeddings/bottom", metadata=[], label_img=self.network.embedding_bottom.weight, metadata_header=[])
                    #     saving_results_progress_bar.update()

                    with ProgressBar(total_value=len(argument_iterator), description="Logging embedding for each image", position=0) as saving_results_progress_bar:
                        for index, (batch_images, output_batch_images) in enumerate(ThreadManagerTQDM(self.__test_batch_log_embedding_iterator, argument_iterator, progress_bar_description="Batch image testing", display_progress_bar=self.display_progress_bar, progress_bar_position=1)):
                            embeddings = self.network.calculate_embeddings(batch_images)
                            for embedding_name, embedding in embeddings.items():
                                self.logger.add_embeddings(embedding, 0, tag=f"testing_embeddings/embedding-{embedding_name}", metadata=[], label_img=[], metadata_header=[])
                            saving_results_progress_bar.update()


                            # print("Here")
                            #  self.logger.add_embeddings(batch_images, step_offset + index, tag=f"Testing dataset/{dataset.__class__.__name__}/Original images")
                            # # self.logger.add_embeddings(encoded_map, step_offset + index, tag=f"Testing dataset:/{dataset.__class__.__name__}/Reconstructed images")
                            # saving_results_progress_bar.update(1)

                except Exception:
                    if self.secondary_network is None:
                        print(f"Exception occurred when logging embeddings, looks like the network {self.network.__class__.__name__} may not support embedding logging.")
                    else:
                        print(f"Exception occurred when logging embeddings, looks like the network {self.network.__class__.__name__}/{self.secondary_network.__class__.__name__} may not support embedding logging.")

        # return (self.__test_batch_log_images(batch_images, batch_index, dataset), self.__test_batch_log_metrics(batch_images, batch_index, dataset))

    def __test_batch_log_images_iterator(self, argument_iterator):
        with torch.no_grad():
            # batch_images, batch_index, dataset = argument_iterator
            batch_images = argument_iterator
            return self.__test_batch_log_images(batch_images)

    def __test_batch_log_metrics_iterator(self, argument_iterator):
        with torch.no_grad():
            # batch_images, batch_index, dataset = argument_iterator
            batch_images = argument_iterator
            return self.__test_batch_log_metrics(batch_images)

    def __test_batch_log_embeddings_iterator(self, argument_iterator):
        with torch.no_grad():
            # batch_images, batch_index, dataset = argument_iterator
            batch_images = argument_iterator
            return self.__test_batch_log_embeddings(batch_images)

    def __test_batch_log_images(self, batch_images):
        with torch.no_grad():
            batch_images = batch_images.to(self.device)
            # self.logger.add_images(batch_images, batch_index, tag=f"Testing dataset/{dataset.__class__.__name__}/Original images")
            # TODO: Fix this for different network output parameters
            output_batch_images = self.network(batch_images)
            output_batch_images = output_batch_images[0] if isinstance(output_batch_images, tuple) else output_batch_images
            # output_batch_images.detach().cpu()
            # self.logger.add_images(output_batch_images, batch_index, tag=f"Testing dataset:/{dataset.__class__.__name__}/Reconstructed images")
            return (batch_images.detach().cpu(), output_batch_images.detach().cpu())

    def __test_batch_log_metrics(self, batch_images):
        with torch.no_grad():
            # TODO: Replace encoding and decoding within the test function and any other metric functions
            batch_images = batch_images.to(self.device)

            metric_results = self.network.calculate_metrics(batch_images, self.metric_controller)

            # quant_t, quant_b, _, _, _ = self.network.encode(batch_images)

            # for metric_name, metric_result in metric_results.items():
            #     self.logger.add_scalar(f"Testing Average Metrics (Batch number)/{metric_name}/{dataset.__class__.__name__}", metric_result, batch_index)
            #     # self.logger.add_scalar(f"Testing Average Metrics (Batch number) - {metric_name}", metric_result, batch_index)

            return metric_results

    def __test_batch_log_embeddings(self, batch_images):
        with torch.no_grad():
            # TODO: Replace encoding and decoding within the test function and any other metric functions
            batch_images = batch_images.to(self.device)
            encoded_batch = self.network.encode(batch_images)

            return (batch_images.detach().cpu(), encoded_batch.detach().cpu())

    def test(self, dataset, max_number_of_test_images=2000, secondary_network = None):
        self.secondary_network = self.secondary_network or secondary_network

        print(f"--- Current network: {self.network.__class__.__name__}{'' if self.secondary_network is None else f'/{self.secondary_network.__class__.__name__}'}---")
        print(f"Current dataset: {dataset.__class__.__name__}")
        print("Testing network")
        self.network.eval()

        # with ThreadManager(max_workers=10) as thread_manager:
        #     print(thread_manager)
        #     for batch_index, (images, classification) in enumerate(dataset.get_test_loader()):
        #         thread_manager.add_thread(self.__test_batch_thread_manager_tqdm, (images, batch_index, dataset))


        # thread_manager = ThreadManager(max_workers=10)
        # for batch_index, (images, classification) in enumerate(dataset.get_test_loader()):
        #     thread_manager.add_thread(self.__test_batch_thread_manager_tqdm, (images, batch_index), dataset, self.things_to_log)
        #
        # thread_manager.finish()

        # # TODO: Code this better to select the first element or other elements in the dataset
        # image_iterator, *_ = zip(*dataset.get_test_loader())
        # self.__test_batch_thread_manager_tqdm(image_iterator, dataset)

        argument_iterator = [(images, batch_index, dataset) for batch_index, (images, classification) in enumerate(dataset.get_test_loader())]
        with torch.no_grad():
            test_loader = dataset.get_test_loader()
            number_of_test_functions_per_test_batch = 500
            total_number_of_test_function_batches = len(test_loader)/number_of_test_functions_per_test_batch
            with ProgressBar(total_value=total_number_of_test_function_batches, description="Processing test batches", display_progress_bar=self.display_progress_bar) as progress_bar:
                argument_iterator = []
                for index, (image, classifications) in enumerate(test_loader):
                    argument_iterator.append(image.clone().detach())
                    if (index % number_of_test_functions_per_test_batch) == number_of_test_functions_per_test_batch - 1:
                        self.__test_batch_thread_manager_tqdm(argument_iterator, dataset, step_offset=index)
                        argument_iterator.clear()
                        progress_bar.update()
                    if index > max_number_of_test_images:
                        break
                else:
                    self.__test_batch_thread_manager_tqdm(argument_iterator, dataset, step_offset=index)
                    progress_bar.update()

    def predict(self, images, collapse_channels=True):
        # TODO: Will currently only work for w*w dimensional images, need to allow for w*h dimensional images
        self.network.eval()
        with torch.no_grad():
            images = images.to(self.device)
            # print(images.shape)
            if collapse_channels:
                return self.network(images.reshape(-1, images.shape[-2], images.shape[-1]))
            else:
                return self.network(images)

    def decode_sample(self, distribution_sample):
        self.network.eval()
        with torch.no_grad():
            return self.network.decode(distribution_sample)

    def save(self, save_path):
        # torch.save({
        #     'epoch': self.epoch,
        #     'model_state_dict': self.network.state_dict(),
        #     'optimizer_state_dict': self.optimiser.state_dict(),
        #     'loss': self.loss,
        # }, save_path)

        # torch.save(self.network.state_dict(), save_path)

        # TODO: Do this in a better way so that the making of the directory is better
        os.makedirs("/".join(save_path.split("/")[:-1:]), exist_ok=True)

        torch.save(self.network, save_path)

    def load(self, load_path):
        # self.network.load_state_dict(torch.load(load_path))
        # self.network.eval()

        # self.network.set_device(self.device)
        # self.network = self.network.to(self.device)

        self.network = torch.load(load_path)
        self.network.set_device(self.device)
        self.network.to(self.device)
        self.network.eval()
        return self.network

    def write_latent_space_to_file(self):
        # Write latent space to file
        latent_space_file = f"{self.root_directory}/latent_space.txt"
        with open(latent_space_file, "w+") as file:
            try:
                for latent_variable in self.network.latent_space.data:
                    file.write(str(latent_variable.cpu().numpy()))
            except Exception as exception:
                file.write(str(exception))

    def write_network_architecture_to_file(self):
        with open(f"{self.root_directory}/network_architecture.txt", "w+") as file:
            file.write(str(self.network))