from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive

class Retinopathy(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, seed=0):
        super(Retinopathy, self).__init__()
        assert split in ['train', 'val', 'test']
        self.root = os.path.join(root, 'retinopathy')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform