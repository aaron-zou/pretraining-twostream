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


class Ucf101RgbDataset(Dataset):
    """
    UCF101 image dataset (spatial stream).
    """

    def __init__(self, path, )


class Ucf101FlowDataset(Dataset):
    """
    UCF101 optical flow dataset (motion stream).
    """
    pass


class Ucf101RgbDataloader(BaseDataLoader):
    """
    UCF101 image dataloader (spatial stream).
    """
    pass


class Ucf101FlowDataloader(BaseDataLoader):
    """
    UCF101 optical flow dataloader (motion stream).
    """
    pass
