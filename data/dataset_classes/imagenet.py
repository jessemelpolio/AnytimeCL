from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive

class ImageNet(Dataset):
    def __init__(self, root, split='train', download=True, transform=None, target_transform=None):
        super(ImageNet, self).__init__()
        assert split in ['train', 'val', 'test']
        if root == '/data/owcl_data':
            self.root = os.path.join(os.path.dirname(root), 'imagenet')
        else:
            self.root = os.path.join(root, 'ILSVRC2012')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.number_to_classes = json.load(open('./data/dataset_classes/imagenet_classes.json', 'r'))     
        
        self.samples = []
        self.targets = []
        self.text_targets = []
        
        self.number = sorted(os.listdir(os.path.join(self.root, self.split)))
        
        self.idx_to_class = {idx: self.number_to_classes[self.number[idx]] for idx in range(len(self.number))}
        self.class_to_idx = {self.number_to_classes[self.number[idx]]: idx for idx in range(len(self.number))}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        for idx, num in enumerate(sorted(os.listdir(os.path.join(self.root, self.split)))):
            for img in sorted(os.listdir(os.path.join(self.root, self.split, num))):
                self.samples.append(os.path.join(self.split, num, img))
                self.targets.append(idx)
                self.text_targets.append(self.number_to_classes[num])

    def __len__(self) -> int:
        return len(self.samples)

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