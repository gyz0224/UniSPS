"""Low-light dataset transforms and factory functions."""

from torchvision.transforms import Compose, RandomCrop, ToTensor

from datasets.lowlight import DatasetFromFolder, DatasetFromFolderEval


def transform1():
    return Compose([RandomCrop((256, 256)), ToTensor()])


def transform2():
    return Compose([ToTensor()])


def get_training_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform1())


def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())


__all__ = ["get_eval_set", "get_training_set", "transform1", "transform2"]
