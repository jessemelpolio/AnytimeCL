from torch.utils.data import Dataset
import os
import json
from PIL import Image
import numpy as np
from torchvision.datasets.folder import make_dataset, IMG_EXTENSIONS

class Resisc45(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, seed=0):
        super(Resisc45, self).__init__()
        assert split in ['train', 'val']
        self.root = os.path.join(root, 'resisc45')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
                
        self.categories = sorted(os.listdir(os.path.join(self.root, "images")))
        class_to_idx = {c: idx for idx, c in enumerate(self.categories)}
        self.instances = make_dataset(os.path.join(self.root, "images"), class_to_idx, IMG_EXTENSIONS)
        self.classes = [cat.replace("_", " ") for cat in self.categories]
        self.idx_to_class = {idx: c for idx, c in enumerate(self.classes)}
        self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}
        
        np.random.seed(seed=seed)
        random_indices = np.random.permutation(len(self.instances))
        
        self.train_val_test_split = {"train": random_indices[:int(0.7 * len(random_indices))],
                                     "val": random_indices[int(0.7 * len(random_indices)):]}
        self.samples = []
        self.targets = []
        self.text_targets = []

        for index in self.train_val_test_split[self.split]:
            self.samples.append(self.instances[index][0])
            self.targets.append(self.instances[index][1])
            self.text_targets.append(self.idx_to_class[self.instances[index][1]])

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, "images"))
        
    def __getitem__(self, index: int):
        img_path = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        target = self.targets[index]
        text_target = self.text_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, text_target