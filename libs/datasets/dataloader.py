import os
from torch.utils.data.dataloader import DataLoader


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


class sDataLoader(DataLoader):

    def get_stream(self):
        while True:
            for data in iter(self):
                yield data
