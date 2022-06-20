import matplotlib.pyplot as plt

class Image():
    @staticmethod
    def convert_image(image_tensor, no_colour_channels=False):
        if no_colour_channels:
            image_tensor = image_tensor[0]
        else:
            image_tensor = image_tensor.permute(1, 2, 0)
        return image_tensor

    @staticmethod
    def display_image(image_tensor, no_colour_channels=False, cmap=None, remove_axes=False, vmin=0, vmax=1):
        image_tensor = Image.convert_image(image_tensor, no_colour_channels=False)
        if remove_axes:
            plt.axis("off")
        plt.imshow(image_tensor, cmap=cmap, vmin=vmin, vmax=vmax)
        Image.show_image()

    @staticmethod
    def display_image_subplot(nrows, ncols, images, no_colour_channels=False, cmap=None, remove_axes=False, display=True, save_path=None, normalised_mean_and_std=[None, None]):
        figure = plt.figure()
        for index, image_tensor in enumerate(images):
            # TODO: Fix or remove [Normalisation code]
            # print(image_tensor.shape)
            # # TODO: Fix unnormalise code, this is inefficient to do it every time
            # if normalised_mean_and_std is not [None, None]:
            #     # TODO: Fix the normalisation selecting only the first of the values
            #     image_tensor = Image.unnormalise(image_tensor, normalised_mean_and_std[0], normalised_mean_and_std[1])
            # print(image_tensor.shape)

            image_tensor = Image.convert_image(image_tensor, no_colour_channels=no_colour_channels)
            subplot = figure.add_subplot(nrows, ncols, index + 1)
            if remove_axes:
                subplot.axis("off")

                subplot.imshow(image_tensor, cmap=cmap)

            if display:
                Image.show_image(figure)

        if save_path is not None:
            Image.save_image(save_path, figure)

    @staticmethod
    def show_image(figure=None):
        if figure is None:
            plt.show()
        else:
            figure.show()

    @staticmethod
    def save_image(save_path, figure=None):
        if figure is None:
            plt.imsave(save_path)
        else:
            figure.savefig(save_path)

    # TODO: Fix or remove [Normalisation code]
    @staticmethod
    def unnormalise(tensor, mean, standard_deviation):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (Tuple): Tuple of means used to normalise
            standard_deviation (Tuple) Tuple of standard devations used to normalise
        Returns:
            Tensor: Normalized image.
        """
        for tensor, mean, standard_deviation in zip(tensor, mean, standard_deviation):
            tensor.mul_(standard_deviation).add_(mean)
        return tensor
        # return (tensor * standard_deviation) + mean