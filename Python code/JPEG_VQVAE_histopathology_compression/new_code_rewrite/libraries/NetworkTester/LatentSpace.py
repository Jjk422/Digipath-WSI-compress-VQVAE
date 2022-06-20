import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class LatentSpace():
    # TODO: Simplify and re-implement this code based on code from https://medium.com/@ranasinghiitkgp/t-sne-visualization-of-high-dimension-mnist-dataset-48fb23d1bafd
    def __init__(self, data, labels=None):
        # Picking the top 1000 points as TSNE takes a lot of time for 15K points
        # data_1000 = images[0:1000, :].detach().cpu().numpy().reshape(128, -1)
        try:
            data_1000 = data.detach().cpu().numpy().reshape(128, -1)
        except:
            data_1000 = data
        # data_1000 = z.detach().cpu().numpy().reshape(128, -1)

        print(data_1000.shape)
        if labels is None:
            labels_1000 = np([i for i in range(len(data_1000))])
        else:
            labels_1000 = labels.detach().cpu().numpy()

        print(labels_1000)
        model = TSNE(n_components=2, random_state=0)
        # configuring the parameteres
        # the number of components = 2
        # default perplexity = 30
        # default learning rate = 200
        # default Maximum number of iterations for the optimization = 1000
        tsne_data = model.fit_transform(data_1000)

        # creating a new data frame which help us in ploting the result data
        tsne_data = np.vstack((tsne_data.T, labels_1000)).T
        tsne_df = pd.DataFrame(data=tsne_data, columns=("Dimension_1", "Dimension_2", "label"))
        # Plotting the result of tsne
        self.figure = sn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, "Dimension_1",
                                                                  "Dimension_2").add_legend()

    def save(self, save_path):
        # plt.show()
        # plt.imsave(f"{network_control.root_directory}/latent_space_graph.png")

        self.figure.savefig(save_path)

        # figure.imsave(f"{network_control.root_directory}/latent_space_graph.png")

    def display(self):
        # TODO: Figure out a way to make this work on just the figure
        plt.show()