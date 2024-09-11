from torch.utils.data import Dataset
import os
from PIL import Image


class Places365LT(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Places365LT, self).__init__()
        assert split in ['train', 'val', 'test', 'open']
        self.root = os.path.join(root, 'places365')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.annotation_file = os.path.join(self.root, 'Places_LT', 'Places_LT_' + self.split + '.txt')
        self.category_file = os.path.join(self.root, 'categories.txt')
        self.class_to_idx = {}

        with open(self.category_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(' ')
                category = line[0][3:]
                if len(category.split('/')) == 2:
                    category = '_'.join(category.split('/'))
                self.class_to_idx[category] = int(line[1])

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        self.targets = []
        self.text_targets = []
        self.samples = []

        with open(self.annotation_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(' ')
                self.samples.append(line[0])
                self.targets.append(int(line[1]))
                self.text_targets.append(self.idx_to_class[int(line[1])])

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.samples[index])
        img = Image.open(img_path).convert('RGB')
        target = self.targets[index]
        text_target = self.text_targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, text_target

    def __len__(self):
        return len(self.samples)
