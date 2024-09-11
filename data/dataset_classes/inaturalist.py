from torch.utils.data import Dataset
import os
from PIL import Image
import json


# loading this dataset is similar to COCO
class iNaturalist(Dataset):
    def __init__(self, root, split='train', download=True, transform=None, target_transform=None):
        super(iNaturalist, self).__init__()
        assert split in ['train', 'val', 'test']
        self.root = os.path.join(root, 'iNaturalist')
        self.split = split

        with open(os.path.join(self.root, 'categories.json'), 'r') as f:
            map_label = json.load(f)
        map_2018 = dict()
        for _map in map_label:
            map_2018[int(_map['id'])] = _map['name'].strip().lower()

        self.annotation_file = os.path.join(self.root, self.split + '2018.json')
        self.samples = []
        self.targets = []
        self.text_targets = []

        with open(self.annotation_file, 'r') as f:
            class_info = json.load(f)

            categories_2018 = [x['name'].strip().lower() for x in map_label]
            self.class_to_idx = {c: idx for idx, c in enumerate(categories_2018)}
            self.idx_to_class = {idx: c for idx, c in enumerate(categories_2018)}
            self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

            self.id2label = dict()
            for categorie in class_info['categories']:
                name = map_2018[int(categorie['name'])]
                self.id2label[int(categorie['id'])] = name.strip().lower()

            for image, annotation in zip(class_info['images'], class_info['annotations']):
                file_path = os.path.join(self.root, image['file_name'])
                self.samples.append(file_path)
                id_name = self.id2label[int(annotation['category_id'])]
                self.text_targets.append(id_name)
                target = self.class_to_idx[id_name]
                self.targets.append(target)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = self.samples[index]
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
