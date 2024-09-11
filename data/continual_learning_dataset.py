import torch
import numpy as np
import random
import bisect
import os
import copy
import json

from .dataset_utils import deal_with_dataset, single_dataset_build
from encode_features.compression_interface import restore_pca_compressed

# This class inherits mostly from ConcatDataset but is modified to allow for open world recognition
class ConcatWithTextDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        self.targets = []
        self.text_targets = []
        self.classes = []
        self.original_classes = []
        self.breakpoints = [0]
        self.name = "ConcatWithTextDataset"

        for n_d, d in enumerate(self.datasets):
            name = d.__name__ if hasattr(d, "__name__") else d.__class__.__name__
            assert not isinstance(
                d, torch.utils.data.IterableDataset
            ), "ConcatDataset does not support IterableDataset"
            assert hasattr(
                d, "class_to_idx"
            ), "Dataset should have class_to_idx attribute"
            assert isinstance(d.class_to_idx, dict), "class_to_idx should be a dict"
            assert hasattr(d, "targets"), "Dataset should have targets attribute"
            assert hasattr(
                d, "prompt_template"
            ), "Dataset should have prompt_template attribute"

            cum_max_class = len(self.classes)
            self.name += "_" + name
            prompt = random.choice(d.prompt_template)
            class_to_idx = {prompt.format(c): i for i, c in enumerate(d.classes)}
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            # TODO: check if these classes are already in the list
            self.original_classes += d.classes
            self.classes += [idx_to_class[i] for i in range(len(idx_to_class))]
            self.breakpoints.append(len(self.classes))
            # cum_max_class = max(self.targets) + 1 if len(self.targets) > 0 else 0
            # cum_max_class = len(self.classes) - len(d.classes)
            text_targets = [idx_to_class[t] for t in d.targets]
            data_targets = [t + cum_max_class for t in d.targets]
            self.targets += data_targets
            self.text_targets += text_targets

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.cumulative_sizes = self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return (
            self.datasets[dataset_idx][sample_idx][0],
            torch.tensor(self.targets[idx], dtype=torch.long),
            self.text_targets[idx],
        )

    def __len__(self):
        return self.cumulative_sizes[-1]


# Let us define that all datasets in this file should return 4 items in the __getitem__ function:
# 1. input image/features
# 2. target label
# 3. text target
# 4. path to the input if there is any (for loading convenience of storing huge amounts of intermediate features)

class IntermediateFeatureFolderDataset(torch.utils.data.Dataset):
    # each dataset is a single dataset of UnitedDataset
    def __init__(self, root, split, transform=None, target_transform=None):
        # key elements that should be extracted from a dataset are: classes, classes_with_prompt, targets, text_targets, class_to_idx
        # root1 is a folder that contains the HDF5 files ordered similarly to ImageFolder
        # split is either 'train' or 'test'
        self.root = root
        self.name = os.path.basename(root)
        self.split = split
        self.samples = []
        self.targets = []
        self.text_targets = []
        self.classes = []
        self.classes_with_prompt = []
        self.class_to_idx = {}
        self.transform = transform
        self.target_transform = target_transform

        # under the root1, there is a file called dataset.json that contains the mapping between the folder name and the class name
        # load the json file
        json_file = os.path.join(root, "dataset_info.json")
        print("Loading dataset info from", json_file)
        if not os.path.exists(json_file):
            print(
                f"dataset_info.json does not exist, now searching for {split}_dataset_info.json instead!"
            )
            json_file = os.path.join(root, f"{split}_dataset_info.json")
            assert os.path.exists(
                json_file
            ), f"{json_file} does not exist, please check the path"

        
        with open(json_file, "r") as f:
            dataset_dict = json.load(f)
        self.classes = list(dataset_dict["classes"])
        self.classes_with_prompt = list(dataset_dict["classes_with_prompt"])
        self.folder_to_class = dataset_dict["folder_to_class"]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        # dataset_dict is a dictionary that maps the folder name to the class name
        # for example, dataset_dict = {'0': 'class0', '1': 'class1', '2': 'class2'}

        for class_name in self.classes:
            class_folder_path = os.path.join(self.root, split, class_name)
            assert os.path.isdir(
                class_folder_path
            ), f"{class_folder_path} is not a folder"
            for file in os.listdir(class_folder_path):
                if not file.endswith(".npy") and not file.endswith(".npz"):
                    continue
                # file is a hdf5 file
                self.samples.append(os.path.join(class_folder_path, file))
                self.text_targets.append(
                    self.classes_with_prompt[
                        self.class_to_idx[self.folder_to_class[class_name]]
                    ]
                )
                self.targets.append(self.class_to_idx[self.folder_to_class[class_name]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # sample is a npy file
        sample = self.samples[index]
        image_features = np.load(sample)
        image_features = torch.from_numpy(image_features)
        if self.transform is not None:
            image_features = self.transform(image_features)
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (
            image_features,
            torch.tensor(target).long(),
            self.text_targets[index],
            [sample],
        )
        

class CompressedIntermediateFeatureFolderDataset(IntermediateFeatureFolderDataset):
    def __init__(self, args, root, split, transform=None, target_transform=None):
        super().__init__(root, split, transform, target_transform)
        self.args = args
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        sample = self.samples[index]
        pack_load = np.load(sample)
        compressed = pack_load["compressed"]
        principle = pack_load["principle"]
        means = pack_load["means"]

        if self.args.int_quantize:
            max_p = pack_load["max_p"]
            min_p = pack_load["min_p"]
            max_c = pack_load["max_c"]
            min_c = pack_load["min_c"]
            max_m = pack_load["max_m"]
            min_m = pack_load["min_m"]
            max_min_val = [max_p, min_p, max_c, min_c,max_m,min_m]

        else:
            max_min_val = None
        
        image_features = restore_pca_compressed(compressed, principle, means, self.args, max_min_val)
        if self.transform is not None:
            image_features = self.transform(image_features)
        
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (
            image_features,
            torch.tensor(target).long(),
            self.text_targets[index],
            [sample],
        )


class TwoIntermediateFeatureFolderDataset(torch.utils.data.Dataset):
    # each dataset is a single dataset of UnitedDataset
    def __init__(self, root1, root2, split, transform=None, target_transform=None, load_image_feature_from_root1=True):
        # key elements that should be extracted from a dataset are: classes, classes_with_prompt, targets, text_targets, class_to_idx
        # root1 is a folder that contains the HDF5 files ordered similarly to ImageFolder
        # root2 is a folder that contains the HDF5 files ordered similarly to ImageFolder
        # split is either 'train' or 'test'
        self.root1 = root1
        self.root2 = root2
        self.name = os.path.basename(root1)
        self.split = split
        self.samples = []
        self.samples2 = []
        self.targets = []
        self.text_targets = []
        self.classes = []
        self.classes_with_prompt = []
        self.class_to_idx = {}
        self.transform = transform
        self.target_transform = target_transform
        self.load_image_feature_from_root1 = load_image_feature_from_root1

        # under the root1, there is a file called dataset.json that contains the mapping between the folder name and the class name
        # load the json file
        # TODO root1 should be the CLIP feature folder
        json_file = os.path.join(root1, "dataset_info.json")
        print("Loading dataset info from", json_file)
        if not os.path.exists(json_file):
            print(
                f"dataset_info.json does not exist, now searching for {split}_dataset_info.json instead!"
            )
            json_file = os.path.join(root1, f"{split}_dataset_info.json")
            assert os.path.exists(
                json_file
            ), f"{json_file} does not exist, please check the path"

        with open(json_file, "r") as f:
            dataset_dict = json.load(f)
        self.classes = list(dataset_dict["classes"])
        self.classes_with_prompt = list(dataset_dict["classes_with_prompt"])
        self.folder_to_class = dataset_dict["folder_to_class"]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        # dataset_dict is a dictionary that maps the folder name to the class name
        # for example, dataset_dict = {'0': 'class0', '1': 'class1', '2': 'class2'}

        for class_name in self.classes:
            class_folder_path = os.path.join(self.root1, split, class_name)
            assert os.path.isdir(
                class_folder_path
            ), f"{class_folder_path} is not a folder"
            for file in os.listdir(class_folder_path):
                if not file.endswith(".npy"):
                    continue
                # file is a hdf5 file
                self.samples.append(os.path.join(class_folder_path, file))
                # print(os.path.join(class_folder_path, file), os.path.exists(os.path.join(class_folder_path, file)))
                self.samples2.append(
                    os.path.join(self.root2, split, class_name, file)
                )  # TODO root2 should be the DINO feature folder
                # print(os.path.join(self.root2, split, class_name, file), os.path.exists(os.path.join(self.root2, split, class_name, file)))

                assert os.path.exists(os.path.join(class_folder_path, file))
                assert os.path.exists(os.path.join(self.root2, split, class_name, file))

                self.text_targets.append(
                    self.classes_with_prompt[
                        self.class_to_idx[self.folder_to_class[class_name]]
                    ]
                )
                self.targets.append(self.class_to_idx[self.folder_to_class[class_name]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # sample is a npy file
        sample = self.samples[index]
        path = [sample, self.samples2[index]]

        image_features = [torch.from_numpy(np.load(sample)) for sample in path]

        if self.transform is not None:
            image_features = [self.transform(image_feature) for image_feature in image_features]

        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (
            image_features,
            torch.tensor(target).long(),
            self.text_targets[index],
            path
        )


class UnitedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        datasets = copy.deepcopy(datasets)
        self.name = "UnitedDataset"
        self.datasets = list(datasets)
        self.num_datasets = len(self.datasets)
        assert self.num_datasets > 0, "datasets should not be an empty iterable"
        self.classes = []
        self.classes_with_prompt = []
        self.breakpoints = [0]
        self.targets = []
        self.text_targets = []

        for d in self.datasets:
            name = d.__name__ if hasattr(d, "__name__") else d.__class__.__name__
            self.name += f"_{name}"
            assert hasattr(
                d, "class_to_idx"
            ), f"Dataset {name} should have class_to_idx attribute"
            class_to_idx = d.class_to_idx
            assert hasattr(
                d, "prompt_template"
            ), f"Dataset {name} should have prompt_template attribute"
            prompt = random.choice(d.prompt_template)
            current_classes = list(class_to_idx.keys())
            class_start_index = len(self.classes)
            self.classes += current_classes
            self.breakpoints.append(len(self.classes))
            current_classes_with_prompt = [prompt.format(c) for c in current_classes]
            self.classes_with_prompt += current_classes_with_prompt

            assert hasattr(d, "targets"), "Dataset should have targets attribute"

            text_targets = [current_classes_with_prompt[t] for t in d.targets]
            remapped_targets = [t + class_start_index for t in d.targets]

            self.targets += remapped_targets
            self.text_targets += text_targets

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = dict(enumerate(self.classes))
        self.idx_to_class_with_prompt = dict(enumerate(self.classes_with_prompt))
        self.cumulative_sizes = self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        dataset_sample = self.datasets[dataset_idx][sample_idx]
        return (
            dataset_sample[0],
            torch.tensor(self.targets[idx], dtype=torch.long),
            self.text_targets[idx],
            dataset_sample[-1],
        )

    def __len__(self):
        return self.cumulative_sizes[-1]


class ContinualConcatDataset(torch.utils.data.Dataset):
    # each dataset is a single dataset of UnitedDataset
    def __init__(self, datasets, name_list):
        assert isinstance(datasets, list), "datasets should be a list"
        self.classes_with_prompt = []
        self.classes = []
        # labels and original_classes are list of lists
        self.labels = []
        self.original_classes = []
        self.name = "_".join(name_list)
        self.datasets = datasets
        self.breakpoints = [0]

        for dataset in datasets:
            original_classes = dataset.classes
            classes_with_prompt = dataset.classes_with_prompt

            # check if there are repeated classes in the original classes
            class_idx_to_correct_idx = {}
            non_repeated_classes = []
            non_repeated_classes_with_prompt = []

            repeated_num = 0
            class_remap_index = len(self.classes)
            for idx, cls in enumerate(original_classes):
                if cls in self.classes:
                    # find the index of the repeated class in self.original_classes
                    ind = self.classes.index(cls)
                    class_idx_to_correct_idx[idx] = ind - class_remap_index
                    repeated_num += 1
                else:
                    class_idx_to_correct_idx[idx] = idx - repeated_num
                    non_repeated_classes.append(cls)
                    non_repeated_classes_with_prompt.append(classes_with_prompt[idx])

            self.original_classes.append(non_repeated_classes)
            self.classes += non_repeated_classes
            self.classes_with_prompt += non_repeated_classes_with_prompt

            dataset_labels = dataset.targets
            non_repeated_dataset_labels = np.zeros_like(dataset_labels)
            for idx, label in enumerate(dataset_labels):
                non_repeated_dataset_labels[idx] = class_idx_to_correct_idx[label]
            dataset_labels = non_repeated_dataset_labels + class_remap_index
            self.labels.append(dataset_labels)

        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx_to_class = {i: self.classes[i] for i in range(len(self.classes))}
        self.idx_to_class_with_prompt = {
            i: self.classes_with_prompt[i] for i in range(len(self.classes_with_prompt))
        }

        self.cumulative_sizes = self.cumsum(self.datasets)
        # concatenated_labels corresponds to the labels implemented in HDF5_dataset
        self.concatenated_labels = np.concatenate(self.labels, axis=0)
        self.targets = self.concatenated_labels
        self.dataset_size = self.cumulative_sizes[-1]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def init_stage(self):
        pass

    def set_stage(self, stage):
        pass

    def forward_stage(self):
        pass

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        dataset_sample = self.datasets[dataset_idx][sample_idx]
        return (
            dataset_sample[0],
            torch.tensor(self.concatenated_labels[idx], dtype=torch.long),
            self.idx_to_class_with_prompt[self.concatenated_labels[idx]],
            dataset_sample[-1],
        )


class ZeroShotDataset(ContinualConcatDataset):
    def __init__(self, datasets, name_list, num_classes=100, padding=0):
        super().__init__(datasets, name_list)

        assert num_classes <= len(
            self.classes
        ), "num_classes should be smaller than the number of classes in the dataset"
        random.seed(0)
        self.padding = padding
        self.random_classes_indices = random.sample(
            list(range(len(self.classes))), num_classes
        )
        self.classes = [self.idx_to_class[i] for i in self.random_classes_indices]
        self.zero_shot_classes = self.classes
        self.classes_with_prompt = [
            self.idx_to_class_with_prompt[i] for i in self.random_classes_indices
        ]
        self.zero_shot_classes_with_prompt = self.classes_with_prompt
        self.subset_item_indices = [
            idx
            for idx in range(self.dataset_size)
            if self.concatenated_labels[idx] in self.random_classes_indices
        ]
        self.remapping_label = {
            self.random_classes_indices[i]: i
            for i in range(len(self.random_classes_indices))
        }

    def update_padding(self, padding):
        self.padding = padding

    def __len__(self):
        return len(self.subset_item_indices)

    def __getitem__(self, index):
        subset_index = self.subset_item_indices[index]

        image, _, text_targets, path = super().__getitem__(subset_index)
        assert (
            self.concatenated_labels[subset_index] in self.random_classes_indices
        ), "label should be in random_classes"
        return (
            image,
            torch.tensor(
                self.remapping_label[self.concatenated_labels[subset_index]]
                + self.padding
            ).long(),
            text_targets,
            path,
        )


class MixDataset(ContinualConcatDataset):
    def __init__(self, datasets, name_list, split=5, num_samples=100):
        super().__init__(datasets, name_list)

        random.seed(0)
        self.repermutation = random.sample(
            list(range(len(self.classes))), len(self.classes)
        )
        self.reverse_repermutation = {
            self.repermutation[i]: i for i in range(len(self.repermutation))
        }
        self.all_classes = [self.idx_to_class[i] for i in self.repermutation]
        self.all_classes_with_prompt = [
            self.idx_to_class_with_prompt[i] for i in self.repermutation
        ]
        self.subset_classes_length = len(self.classes) // split

        self.subset_class_indices = {}
        for task in range(split):
            self.subset_class_indices[task] = self.repermutation[
                task
                * self.subset_classes_length : (task + 1)
                * self.subset_classes_length
            ]

        sample_count = {}

        self.subset_item_indices = {}
        for task in range(split):
            self.subset_item_indices[task] = []
        for idx in range(self.dataset_size):
            task = (
                self.reverse_repermutation[self.concatenated_labels[idx]]
                // self.subset_classes_length
            )
            if task < split:
                assert (
                    self.concatenated_labels[idx] in self.subset_class_indices[task]
                ), "label should be in subset_class_indices"
                if self.concatenated_labels[idx] not in sample_count:
                    sample_count[self.concatenated_labels[idx]] = 0
                if sample_count[self.concatenated_labels[idx]] < num_samples:
                    sample_count[self.concatenated_labels[idx]] += 1
                    self.subset_item_indices[task].append(idx)

        self.stage = 0
        self.num_stages = split

        self.classes = [
            self.idx_to_class[i] for i in self.subset_class_indices[self.stage]
        ]
        self.classes_with_prompt = [
            self.idx_to_class_with_prompt[i]
            for i in self.subset_class_indices[self.stage]
        ]
        self.remapping_label = {
            self.subset_class_indices[self.stage][i]: i
            for i in range(len(self.subset_class_indices[self.stage]))
        }
        
        self.additional_classes = None
        self.additional_classes_with_prompt = None
        
    def add_additional_classes(self, additional_classes, additional_classes_with_prompt):
        self.additional_classes = additional_classes
        self.additional_classes_with_prompt = additional_classes_with_prompt

        if self.stage == 0:
            self.classes += self.additional_classes
            self.classes_with_prompt += self.additional_classes_with_prompt

    def forward_stage(self):
        self.stage += 1
        self.classes = self.all_classes[
            self.stage
            * self.subset_classes_length : (self.stage + 1)
            * self.subset_classes_length
        ]
        self.classes_with_prompt = self.all_classes_with_prompt[
            self.stage
            * self.subset_classes_length : (self.stage + 1)
            * self.subset_classes_length
        ]
        self.repermutation_subset = self.repermutation[
            self.stage
            * self.subset_classes_length : (self.stage + 1)
            * self.subset_classes_length
        ]
        self.remapping_label = {
            self.repermutation_subset[i]: i
            for i in range(len(self.repermutation_subset))
        }
        
        if self.additional_classes is not None:
            self.classes += self.additional_classes
            self.classes_with_prompt += self.additional_classes_with_prompt

        if self.stage == self.num_stages:
            print("Finish all stages")
            return

    def __len__(self):
        return len(self.subset_item_indices[self.stage])

    def __getitem__(self, index):
        subset_index = self.subset_item_indices[self.stage][index]
        assert (
            self.concatenated_labels[subset_index]
            in self.subset_class_indices[self.stage]
        ), "label should be in subset_class_indices"

        image, _, text_targets, path = super().__getitem__(subset_index)
        return (
            image,
            torch.tensor(
                self.remapping_label[self.concatenated_labels[subset_index]]
            ).long(),
            text_targets,
            path,
        )


class DataIncrementalDataset(ContinualConcatDataset):
    def __init__(self, datasets, name_list, perc=0.02, incremental_type="double"):
        super().__init__(datasets, name_list)
        # build incremental list
        self.incremental_type = incremental_type
        self.perc_list = []
        if incremental_type == "double":
            incremental_perc = perc
            if perc > 1:
                perc = perc / 100
            self.perc_list.append(perc)
            while incremental_perc < 1:
                self.perc_list.append(perc)
                perc = perc * 2
                incremental_perc += perc
            self.perc_list.append(1 - sum(self.perc_list))
        elif incremental_type == "fixed":
            if perc > 1:
                perc = perc / 100
            self.perc_list = [perc] * int(1 / perc)
            if sum(self.perc_list) < 1:
                self.perc_list.append(1 - sum(self.perc_list))
        else:
            raise ValueError("incremental_type should be double or fixed")
        self.num_stages = len(self.perc_list)
        random.seed(0)
        # This is basically shuffling the indices of the dataset
        self.random_indices = random.sample(range(self.dataset_size), self.dataset_size)
        self.learned_classes_indices = torch.tensor([], dtype=torch.long)
        self.init_stage()

    def __build_subset(self):
        start_idx = int(self.dataset_size * sum(self.perc_list[: self.stage]))
        end_idx = int(self.dataset_size * sum(self.perc_list[: self.stage + 1]))
        if self.stage == self.num_stages - 1:
            end_idx = self.dataset_size
        self.subset_item_indices = self.random_indices[start_idx:end_idx]
        self.__get_current_stage_classes()
        self.__get_learned_classes()

    def __get_current_stage_classes(self):
        self.current_classes_indices = torch.unique(
            torch.from_numpy(self.concatenated_labels[self.subset_item_indices]),
            sorted=True,
        )
        self.current_classes = [
            self.idx_to_class[i.item()] for i in self.current_classes_indices
        ]
        self.current_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.current_classes_indices
        ]
        # self.current_classes_features = self.unique_class_features[
        #     self.current_classes_indices.numpy()
        # ]

    def __get_learned_classes(self):
        learned_classes_indices_current_stage = torch.unique(
            torch.from_numpy(self.concatenated_labels[self.subset_item_indices]),
            sorted=True,
        )
        self.learned_classes_indices = torch.cat(
            (self.learned_classes_indices, learned_classes_indices_current_stage)
        )
        self.learned_classes_indices = torch.unique(
            self.learned_classes_indices, sorted=True
        )
        self.learned_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.learned_labels = [
            self.class_to_idx[self.idx_to_class[i.item()]]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]
        # self.learned_classes_features = self.unique_class_features[
        #     self.learned_classes_indices.numpy()
        # ]

    def init_stage(self):
        self.stage = 0
        self.__build_subset()

    def set_stage(self, stage):
        self.stage = stage
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage == self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __len__(self):
        return len(self.subset_item_indices)

    def __getitem__(self, idx):
        actual_idx = self.subset_item_indices[idx]
        return super().__getitem__(actual_idx)


class ClassIncrementalDataset(ContinualConcatDataset):
    def __init__(self, datasets, name_list, num_classes=100, include_whole=False):
        super().__init__(datasets, name_list)
        # this means that each stage has num_classes classes
        self.num_classes = num_classes
        # self.num_stages = (
        #     len(self.classes) // num_classes
        #     if len(self.classes) % num_classes < num_classes / 2
        #     else len(self.classes) // num_classes + 1
        #     # if len(self.classes) % num_classes == 0
        #     # else len(self.classes) // num_classes + 1
        # )
       
        self.num_stages = (len(self.classes) // num_classes) + 1 if len(self.classes) % num_classes != 0 else len(self.classes) // num_classes

        # print("num_stages", self.num_stages)
        self.classes_mapping = list(range(len(self.classes)))
        self.include_whole = include_whole

        random.seed(0)
        random.shuffle(self.classes_mapping)
        self.init_stage()

    def __build_subset(self):
        start_idx = self.stage * self.num_classes
        end_idx = min((self.stage + 1) * self.num_classes, len(self.classes))
        if self.stage == self.num_stages - 1:
            end_idx = len(self.classes)
        self.subset_classes = self.classes_mapping[start_idx:end_idx]

        if self.include_whole:
            self.subset_item_indices = list(range(self.dataset_size))
        else:
            self.subset_item_indices = [
                idx
                for idx in range(self.dataset_size)
                if self.concatenated_labels[idx] in self.subset_classes
            ]
        self.__get_current_stage_classes(start_idx, end_idx)
        # if self.stage != 0:
        self.__get_past_classes()
        self.__get_learned_classes(end_idx)

    def __get_current_stage_classes(self, start_idx, end_idx):
        self.current_classes_indices = torch.sort(
            torch.tensor(self.classes_mapping[start_idx:end_idx])
        )[0]
        self.current_classes = [
            self.idx_to_class[i.item()] for i in self.current_classes_indices
        ]
        self.current_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.current_classes_indices
        ]
        # self.current_classes_features = self.unique_class_features[
        #     self.current_classes_indices.numpy()
        # ]

    def __get_past_classes(self):
        if self.stage == 0:
            self.past_classes_indices = torch.tensor([], dtype=torch.long)
            self.past_classes = []
            self.past_labels = []
            self.past_classes_with_prompt = []
            return
        # we call __get_past_classes() before __get_learned_classes()
        self.past_classes_indices = self.learned_classes_indices
        self.past_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.past_labels = [
            self.class_to_idx[self.idx_to_class[i.item()]]
            for i in self.learned_classes_indices
        ]
        self.past_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]
        # self.learned_classes_features = self.unique_class_features[
        #     self.learned_classes_indices.numpy()
        # ]

    def __get_learned_classes(self, end_idx):
        self.learned_classes_indices = torch.sort(
            torch.tensor(self.classes_mapping[:end_idx])
        )[0]
        self.learned_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.learned_labels = [
            self.class_to_idx[self.idx_to_class[i.item()]]
            for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]
        # self.learned_classes_features = self.unique_class_features[
        #     self.learned_classes_indices.numpy()
        # ]

    def init_stage(self):
        self.stage = 0
        self.__build_subset()

    def set_stage(self, stage):
        self.stage = stage
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage == self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __len__(self):
        return len(self.subset_item_indices)

    def __getitem__(self, idx):
        actual_idx = self.subset_item_indices[idx]
        return super().__getitem__(actual_idx)


class DatasetIncrementalDataset(ContinualConcatDataset):
    def __init__(self, datasets, name_list):
        super().__init__(datasets, name_list)
        self.num_stages = len(self.datasets)
        self.learned_classes_indices = torch.tensor([], dtype=torch.long)
        self.init_stage()

    def __build_subset(self):
        self.__get_current_stage_classes()
        self.__get_learned_classes()

    def init_stage(self):
        self.stage = 0
        self.__build_subset()

    def set_stage(self, stage):
        self.stage = stage
        self.__build_subset()

    def forward_stage(self):
        self.stage += 1
        if self.stage == self.num_stages:
            print("Finish all stages")
            return
        self.__build_subset()

    def __get_current_stage_classes(self):
        self.current_classes_indices = torch.unique(
            torch.from_numpy(self.labels[self.stage]), sorted=True
        )
        # print("current_classes_indices", self.current_classes_indices)
        # print("idx_to_class", self.idx_to_class)
        self.current_classes = [
            self.idx_to_class[i.item()] for i in self.current_classes_indices
        ]
        self.current_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.current_classes_indices
        ]
        # self.current_classes_features = self.unique_class_features[
        #     self.current_classes_indices.numpy()
        # ]

    def __get_learned_classes(self):
        learned_classes_indices_current_stage = torch.unique(
            torch.from_numpy(self.labels[self.stage]), sorted=True
        )
        self.learned_classes_indices = torch.cat(
            (self.learned_classes_indices, learned_classes_indices_current_stage)
        )
        self.learned_classes = [
            self.idx_to_class[i.item()] for i in self.learned_classes_indices
        ]
        self.learned_classes_with_prompt = [
            self.idx_to_class_with_prompt[i.item()]
            for i in self.learned_classes_indices
        ]

    def __len__(self):
        return len(self.labels[self.stage])

    def __getitem__(self, index):
        # TODO: notice the original implementation of this function is incorrect since the label given by self.datasets[self.stage][index] is not the correct label
        image, _, text_label, path = self.datasets[self.stage][index]
        return image, self.labels[self.stage][index], text_label, path

