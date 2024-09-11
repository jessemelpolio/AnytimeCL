import os
from data.dataset_utils import deal_with_dataset
from data.continual_learning_dataset import (
    IntermediateFeatureFolderDataset,
    TwoIntermediateFeatureFolderDataset,
    CompressedIntermediateFeatureFolderDataset,
    DataIncrementalDataset,
    ClassIncrementalDataset,
    DatasetIncrementalDataset,
    ZeroShotDataset,
    MixDataset,
    ContinualConcatDataset,
    single_dataset_build,
    ConcatWithTextDataset,
)

def create_dataset(args, dataset, path_type="single", split="test"):
    path = os.path.join(args.data_root, dataset)
    if path_type == "single":
        return IntermediateFeatureFolderDataset(path, split)
    elif path_type == "compression":
        return CompressedIntermediateFeatureFolderDataset(args, path, split)
    elif path_type == "dino":
        dino_path = os.path.join(args.dinov2_data_root, dataset)
        return TwoIntermediateFeatureFolderDataset(path, dino_path, split, load_image_feature_from_root1=False)

def get_datasets(args, dataset_list, path_type="single", split="test"):
    return [create_dataset(args, dataset, path_type, split) for dataset in dataset_list]

def get_zero_shot_task(args, path_type="single"):
    dataset_list = args.held_out_dataset.split(",")
    test_datasets = get_datasets(args, dataset_list, path_type)
    return [ContinualConcatDataset([dataset], [name]) for dataset, name in zip(test_datasets, dataset_list)]

def get_union_task(args, path_type="single"):
    dataset_list, _ = deal_with_dataset(args)
    test_datasets = get_datasets(args, dataset_list, path_type)
    return DatasetIncrementalDataset(test_datasets, dataset_list)

def get_mix_task(args, path_type="single"):
    dataset_list, _ = deal_with_dataset(args)
    test_datasets = get_datasets(args, dataset_list, path_type)
    return MixDataset(test_datasets, dataset_list)

def get_combined_zero_shot_task(args, task_type, path_type="single"):
    target_dataset_list, _ = deal_with_dataset(args)
    zero_shot_dataset_list = args.held_out_dataset.split(",")

    target_test_datasets = get_datasets(args, target_dataset_list, path_type)
    zero_shot_test_datasets = get_datasets(args, zero_shot_dataset_list, path_type)

    if task_type == "union":
        main_task = DatasetIncrementalDataset(target_test_datasets, target_dataset_list)
        num_classes = 100
        padding = len(main_task.classes)
    elif task_type == "mix":
        main_task = MixDataset(target_test_datasets, target_dataset_list)
        num_classes = len(main_task.classes)
        padding = num_classes

    zero_shot_task = ZeroShotDataset(zero_shot_test_datasets, zero_shot_dataset_list, num_classes, padding)

    all_classes = main_task.classes + (zero_shot_task.classes if task_type == "union" else zero_shot_task.zero_shot_classes)
    all_classes_with_prompt = main_task.classes_with_prompt + (zero_shot_task.classes_with_prompt if task_type == "union" else zero_shot_task.zero_shot_classes_with_prompt)

    if task_type == "union":
        for task in [main_task, zero_shot_task]:
            task.classes = all_classes
            task.classes_with_prompt = all_classes_with_prompt
        return main_task, zero_shot_task
    elif task_type == "mix":
        # The following function already adds the additional classes to the main task
        main_task.add_additional_classes(zero_shot_task.zero_shot_classes, zero_shot_task.zero_shot_classes_with_prompt)
        zero_shot_task.classes = all_classes
        zero_shot_task.classes_with_prompt = all_classes_with_prompt
        return main_task
    else:
        raise NotImplementedError("task_type not implemented")

def get_continual_learning_dataset(args, path_type="single"):
    dataset_list, _ = deal_with_dataset(args)
    train_datasets = get_datasets(args, dataset_list, path_type, "train")
    test_datasets = get_datasets(args, dataset_list, path_type, "test")

    if args.incremental == "data":
        train_dataset = DataIncrementalDataset(train_datasets, dataset_list)
        test_dataset = [ContinualConcatDataset(test_datasets, dataset_list)]
    elif args.incremental == "class":
        train_dataset = ClassIncrementalDataset(train_datasets, dataset_list, args.num_classes)
        test_dataset = ClassIncrementalDataset(test_datasets, dataset_list, args.num_classes, include_whole=True)
    elif args.incremental == "dataset":
        train_dataset = DatasetIncrementalDataset(train_datasets, dataset_list)
        test_dataset = [ContinualConcatDataset([test_datasets[i]], [dataset_list[i]]) for i in range(len(test_datasets))]
    else:
        raise NotImplementedError("incremental type not implemented")

    return train_dataset, test_dataset

def get_continual_learning_dataset_with_compression(args):
    dataset_list, _ = deal_with_dataset(args)
    train_datasets = get_datasets(args, dataset_list, "compression", "train")
    test_datasets = get_datasets(args, dataset_list, "compression", "test")

    if args.incremental == "data":
        train_dataset = DataIncrementalDataset(train_datasets, dataset_list)
        test_dataset = [ContinualConcatDataset(test_datasets, dataset_list)]
    elif args.incremental == "class":
        train_dataset = ClassIncrementalDataset(train_datasets, dataset_list, args.num_classes)
        test_dataset = ClassIncrementalDataset(test_datasets, dataset_list, args.num_classes, include_whole=True)
    elif args.incremental == "dataset":
        train_dataset = DatasetIncrementalDataset(train_datasets, dataset_list)
        test_dataset = [ContinualConcatDataset([test_datasets[i]], [dataset_list[i]]) for i in range(len(test_datasets))]
    else:
        raise NotImplementedError("incremental type not implemented")

    return train_dataset, test_dataset

def get_held_out_dataset(args, path_type="single", load_train=True):
    dataset_list = args.held_out_dataset.split(",")
    train_datasets = get_datasets(args, dataset_list, path_type, "train") if load_train else []
    test_datasets = get_datasets(args, dataset_list, path_type, "test")
    
    return ([ContinualConcatDataset([dataset], [name]) for dataset, name in zip(train_datasets, dataset_list)],
            [ContinualConcatDataset([dataset], [name]) for dataset, name in zip(test_datasets, dataset_list)])

def get_held_out_dataset_with_compression(args, load_train=True):
    dataset_list = args.held_out_dataset.split(",")
    train_datasets = get_datasets(args, dataset_list, "compression", "train") if load_train else []
    test_datasets = get_datasets(args, dataset_list, "compression", "test")
    
    return ([ContinualConcatDataset([dataset], [name]) for dataset, name in zip(train_datasets, dataset_list)],
            [ContinualConcatDataset([dataset], [name]) for dataset, name in zip(test_datasets, dataset_list)])


def get_single_image_datasets(args, transform=None):
    dataset_list, _ = deal_with_dataset(args)
    train_datasets, test_datasets = [], []
    for dataset in dataset_list:
        train_set, test_set = single_dataset_build(args, dataset, transform=transform)
        train_datasets.append(ConcatWithTextDataset([train_set]))
        test_datasets.append(ConcatWithTextDataset([test_set]))
    return train_datasets, test_datasets

# Wrapper functions to maintain backward compatibility
get_union_zero_shot_task = lambda args: get_combined_zero_shot_task(args, "union")
get_mix_zero_shot_task = lambda args: get_combined_zero_shot_task(args, "mix")
get_dino_zero_shot_task = lambda args: get_zero_shot_task(args, "dino")
get_dino_union_task = lambda args: get_union_task(args, "dino")
get_dino_mix_task = lambda args: get_mix_task(args, "dino")
get_dino_union_zero_shot_task = lambda args: get_combined_zero_shot_task(args, "union", "dino")
get_dino_mix_zero_shot_task = lambda args: get_combined_zero_shot_task(args, "mix", "dino")
get_single_npy_continual_learning_dataset = lambda args: get_continual_learning_dataset(args, "single")
get_single_npy_continual_learning_compression_dataset = lambda args: get_continual_learning_dataset_with_compression(args)
get_single_npy_held_out_dataset = lambda args, load_train=True: get_held_out_dataset(args, "single", load_train)
get_single_npy_held_out_compression_dataset = lambda args, load_train=True: get_held_out_dataset_with_compression(args, load_train)
get_dino_clip_npy_continual_learning_dataset = lambda args: get_continual_learning_dataset(args, "dino")
get_dino_clip_npy_held_out_dataset = lambda args, load_train=True: get_held_out_dataset(args, "dino", load_train)
