import numpy as np
import os
import torch
import pickle
from torch.utils.data import Dataset
from glob import glob


class SliceDataset(Dataset):
    """
    Dataset containing slices of volumes
    """

    def __init__(self, ids, volumes, labelss, class_weightss, headers=None):
        super(SliceDataset, self).__init__()

        _, H, W = volumes[0].shape
        self.ids_extended = np.array([vol_id for vol_id, volume in zip(ids, volumes) for _ in range(len(volume))])
        imgs = torch.from_numpy(np.concatenate(volumes).astype(np.float32))
        self.imgs = imgs if len(imgs.size()) == 4 else torch.unsqueeze(imgs, dim=1)
        self.labelss = torch.from_numpy(np.concatenate(labelss).reshape((-1, H, W)))
        self.class_weightss = torch.from_numpy(np.concatenate(class_weightss).reshape((-1, H, W)))
        self.headers = headers

    def __getitem__(self, idx):
        vol_id = self.ids_extended[idx]
        img = self.imgs[idx]
        labels = self.labelss[idx]
        class_weights = self.class_weightss[idx]
        if self.headers:
            header = self.headers[idx]
            return vol_id, img, labels, class_weights, header
        else:
            return vol_id, img, labels, class_weights

    def __len__(self):
        return len(self.ids_extended)


class VolumeDataset(Dataset):
    """
    Dataset containing whole volumes
    """

    def __init__(self, ids, volumes, labelss, class_weightss, headers=None):
        super(VolumeDataset, self).__init__()

        self.ids = ids
        self.volumes = volumes
        self.labelss = labelss
        self.class_weightss = class_weightss
        self.headers = headers

    def __getitem__(self, idx):
        vol_id = self.ids[idx]
        volume = torch.from_numpy(self.volumes[idx])
        labels = torch.from_numpy(self.labelss[idx])
        class_weights = torch.from_numpy(self.class_weightss[idx])
        if self.headers:
            header = self.headers[idx]
            return vol_id, volume, labels, class_weights, header
        else:
            return vol_id, volume, labels, class_weights

    def __len__(self):
        return len(self.ids)


def get_dataset(data_dir, mode, sliced=True, header=False):
    """
    Loads data files and creates specified dataset
    :param data_dir: Directory inheriting pickled data files
    :param mode: 'train', 'val' or 'test' dataset
    :param sliced: If True creates dataset returning slices otherwise volumes
    :return: Specified Dataset
    """
    assert mode in ['train', 'val', 'test'], 'Value for "mode" must either be "train", "val" or "test"'

    data_dict = {}
    if 'nas' in data_dir:
        for fp in glob(os.path.join(data_dir, '*_' + mode)):
            with open(fp, 'rb') as f:
                data_dict[os.path.basename(fp).replace('_' + mode, '')] = pickle.load(f)
    else:
        for fp in glob(os.path.join(data_dir, '*_{}.pkl'.format(mode))):
            with open(fp, 'rb') as f:
                data_dict[os.path.basename(fp)[:-(len(mode) + 5)]] = pickle.load(f)

    if sliced:
        return SliceDataset(data_dict['ids'], data_dict['volumes'], data_dict['labels'], data_dict['class_weights'], data_dict['headers'] if header else None)
    else:
        return VolumeDataset(data_dict['ids'], data_dict['volumes'], data_dict['labels'], data_dict['class_weights'], data_dict['headers'] if header else None)
