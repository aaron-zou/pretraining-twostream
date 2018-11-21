from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split,
                 num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset,
                                              batch_size, shuffle,
                                              validation_split, num_workers)


class Hmdb51RgbDataset(Dataset):
    """
    HMDB51 image dataset (spatial stream).
    """
    pass


class Hmdb51FlowDataset(Dataset):
    """
    HMDB51 optical flow dataset (motion stream).
    """
    pass


class Hmdb51RgbDataloader(BaseDataLoader):
    """
    HMDB51 image dataloader (spatial stream).
    """
    pass


class Hmdb51FlowDataloader(BaseDataLoader):
    """
    HMDB51 optical flow dataloader (motion stream).
    """
    pass
