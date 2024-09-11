from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive

class CLEVRCount(Dataset):
    """
    Provides CLEVR dataset.
    Currently, two tasks are supported:
        1. Predict number of objects. (X)
        2. Predict distnace to the closest object.
    """
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(CLEVRCount, self).__init__()
        assert split in ['train', 'val', 'test']
        self.root = os.path.join(os.path.dirname(root), 'clevr', 'CLEVR_v1.0')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        scenes_json = os.path.join(self.root, 'scenes', 'CLEVR_' + self.split + '_scenes.json')
        scenes_data = json.load(open(scenes_json, 'r'))
        
        self.samples = []
        self.targets = []
        self.text_targets = []

        for scene in scenes_data['scenes']:
            self.samples.append(scene['image_filename'])
            self.targets.append(len(scene['objects']) - 3)
            self.text_targets.append(str(len(scene['objects'])))
            
        self.classes = ['3', '4', '5', '6', '7', '8', '9', '10']
        self.idx_to_class = {idx: self.classes[idx] for idx in range(len(self.classes))}
        self.class_to_idx = {self.classes[idx]: idx for idx in range(len(self.classes))}
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, 'images', self.split, self.samples[index])
        img = Image.open(img_path).convert('RGB')
        target = self.targets[index]
        text_target = self.text_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, text_target


class CLEVRDist(Dataset):
    """
    Provides CLEVR dataset.
    Currently, two tasks are supported:
        1. Predict number of objects.
        2. Predict distnace to the closest object. (X)
    """
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(CLEVRDist, self).__init__()
        assert split in ['train', 'val', 'test']
        self.root = os.path.join(os.path.dirname(root), 'clevr', 'CLEVR_v1.0')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
        
        scenes_json = os.path.join(self.root, 'scenes', 'CLEVR_' + self.split + '_scenes.json')
        scenes_data = json.load(open(scenes_json, 'r'))
        
        self.samples = []
        self.targets = []
        self.text_targets = []

        for scene in scenes_data['scenes']:
            self.samples.append(scene['image_filename'])
            dist = [scene["objects"][i]["pixel_coords"][2] for i in range(len(scene["objects"]))]
            dist = np.min(dist)
            self.targets.append(np.max(np.where((self.thrs - dist) < 0)))
            self.text_targets.append(str(self.thrs[np.max(np.where((self.thrs - dist) < 0))]))
            
        self.classes = [str(thr) for thr in self.thrs]
        self.idx_to_class = {idx: self.classes[idx] for idx in range(len(self.classes))}
        self.class_to_idx = {self.classes[idx]: idx for idx in range(len(self.classes))}
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, 'images', self.split, self.samples[index])
        img = Image.open(img_path).convert('RGB')
        target = self.targets[index]
        text_target = self.text_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, text_target