import torchvision
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, model, log_directory=None):
        self.log_writer = SummaryWriter() if log_directory is None else SummaryWriter(log_directory)
        # print(f"Log writer {TODO: State time} created")
        self.model = model

    def __del__(self):
        self.log_writer.close()

    def add_images(self, images, step, tag="images"):
        images = images[0] if isinstance(images, tuple) else images
        grid = torchvision.utils.make_grid(images)
        self.log_writer.add_image(tag, grid, step)
        # TODO: Ensure commenting this out did not break anything
        # self.log_writer.add_graph(self.model, images)

    def create_graph(self, images):
        images = images[0] if isinstance(images, tuple) else images
        self.log_writer.add_graph(self.model, images)

    def add_scalar(self, name, value, global_step):
        self.log_writer.add_scalar(name, value, global_step)

    def add_scalars(self, name, values, global_step):
        self.log_writer.add_scalars(name, values, global_step)

    def add_loss_value(self, loss_value, train_or_test, global_step):
        if train_or_test == "train":
            self.add_scalar('Loss/train', loss_value, global_step)
        elif train_or_test == "test":
            self.add_scalar('Loss/test', loss_value, global_step)

    def add_accuracy_value(self, loss_value, train_or_test, global_step):
        if train_or_test == "train":
            self.add_scalar('Accuracy/train', loss_value, global_step)
        elif train_or_test == "test":
            self.add_scalar('Accuracy/test', loss_value, global_step)

    def add_embeddings(self, embeddings, step, tag="embeddings", metadata=None, label_img=None, metadata_header=None):
        # self.log_writer.add_embedding(embeddings, metadata = metadata, label_img = label_img, global_step=step, tag=tag, metadata_header=metadata_header)
        self.log_writer.add_embedding(embeddings, global_step=step, tag=tag)