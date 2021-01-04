# encoding: utf-8
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: cluster_process.py
@Time: 2020-03-14 15:45
@Desc: cluster_process.py
"""

try:
    import torch

    from nilearn import datasets
    from torch.utils.data import Dataset
except ImportError as e:
    print(e)
    raise ImportError


class ADHD(Dataset):

    def __init__(self, root, data, train=True, transform=None, download=False):
        super(ADHD, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.data = data

    def __getitem__(self, index):

        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        return img.unsqueeze(1)

    def __len__(self):

        return len(self.data)


DATASET_FN_DICT = {
    'adhd': ADHD,
}


dataset_list = DATASET_FN_DICT.keys()


def _get_dataset(dataset_name='adhd'):

    if dataset_name in DATASET_FN_DICT:
        return DATASET_FN_DICT[dataset_name]
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, dataset_list))


# get the loader of all datas
def get_dataloader(data, dataset_path='../datasets/brain',
                   dataset_name='adhd', train=True, batch_size=50):
    dataset = _get_dataset(dataset_name)

    loader = torch.utils.data.DataLoader(
        dataset(dataset_path, data, download=True, train=train, transform=lambda x: torch.tensor(x)),
        batch_size=batch_size,
        shuffle=False,
    )
    return loader


def get_adhd_data(data_dir='./datasets/brain', n_subjects=6):

    dataset = datasets.fetch_adhd(data_dir=data_dir, n_subjects=n_subjects)
    imgs = dataset.func

    return imgs