from pathlib import Path
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive

class SUN397(Dataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.

    Args:
        root (string): Root directory of the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root1 directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"
    
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, seed=0):
        super(SUN397, self).__init__()
        assert split in ['train', 'val', 'test']
        self.root = Path(root) / "SUN397"
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            self._download(root)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(os.path.join(self.root, "ClassName.txt")) as f:
            self.classes = [c[3:].strip() for c in f]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self.idx_to_class = dict(zip(range(len(self.classes)), self.classes))
        self._image_files = list(self.root.rglob("sun_*.jpg"))

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self.root).parts[1:-1])] for path in self._image_files
        ]
        
        np.random.seed(seed=seed)
        random_indices = np.random.permutation(len(self._image_files))
        
        self.train_val_test_split = {"train": random_indices[:int(0.6 * len(random_indices))],
                                     "val": random_indices[int(0.6 * len(random_indices)):int(0.8 * len(random_indices))],
                                     "test": random_indices[int(0.8 * len(random_indices)):]}

        self.targets = []
        self.text_targets = []
        self.samples = []
        
        for index in self.train_val_test_split[self.split]:
            self.samples.append(self._image_files[index])
            self.targets.append(self._labels[index])
            self.text_targets.append(self.idx_to_class[self._labels[index]])
            
    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, self.samples[index])
        img = Image.open(img_path).convert('RGB')
        target = self.targets[index]
        text_target = self.text_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, text_target

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root))

    def _download(self, root) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=root, md5=self._DATASET_MD5)