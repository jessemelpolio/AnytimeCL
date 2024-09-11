import torch
import sys
import json
import numpy as np
import os
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from options.base_options import BaseOptions
from models.clip_module import LearnableCLIPModule
from models.memory_module import MemoryModule
from data.tasks import (
    get_single_image_datasets,
)

sys.path.append(os.path.join(dir_path, "../models"))

from dinov2.models import build_model_from_cfg
from dinov2.configs import load_and_merge_config
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform


def get_dino_features(model, dataset, store_folder, dataset_type="train", features_of_the_n_layer_to_last=1):
    if not os.path.exists(store_folder):
        os.makedirs(store_folder, exist_ok=True)

    store_folder = os.path.join(store_folder, dataset.name.split("_")[1])
    if not os.path.exists(store_folder):
        os.makedirs(store_folder, exist_ok=True)

    if not os.path.exists(
        os.path.join(store_folder, f"{dataset_type}_dataset_info.json")
    ):
        # save dataset info as a json file
        dataset_info = {
            "classes": dataset.original_classes,
            "classes_with_prompt": dataset.classes,
            "class_to_idx": dataset.class_to_idx,
            "folder_to_class": {folder: folder for folder in dataset.original_classes},
        }
        # save dataset info as a json file
        with open(
            os.path.join(store_folder, f"{dataset_type}_dataset_info.json"), "w"
        ) as f:
            json.dump(dataset_info, f)

    store_folder = os.path.join(store_folder, dataset_type)
    if not os.path.exists(store_folder):
        os.makedirs(store_folder, exist_ok=True)

    # should be 1 to save each sample as a separate file
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    idx_to_class = {idx: cls for idx, cls in enumerate(dataset.original_classes)}

    # num_of_instances_encoded = 0
    num_instances_per_class = {cls: 0 for cls in dataset.original_classes}
    for bn, batch in enumerate(tqdm(dl)):
        image, label, label_text = batch
        image = image.to("cuda:0")
        cls = idx_to_class[label.numpy().item()]
        cls_folder = os.path.join(store_folder, cls)
        if not os.path.exists(cls_folder):
            os.makedirs(cls_folder, exist_ok=True)

        with torch.no_grad():
            intermediate_feat = model.simple_get_intermediate_features_of_the_n_layer_to_last(image, features_of_the_n_layer_to_last)

            with open(
                os.path.join(cls_folder, f"{num_instances_per_class[cls]}.npy"), "wb"
            ) as f:
                # print(
                #     f"Saving to {os.path.join(cls_folder, f'{num_instances_per_class[cls]}.npy')}"
                # )
                np.save(f, intermediate_feat.cpu().numpy())
                num_instances_per_class[cls] += 1

    print(f"Number of instances per class: {num_instances_per_class}")
    print(f"Total number of instances: {sum(num_instances_per_class.values())}")


def seed_everything(seed):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
class CustomOptions:
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--store_folder', type=str, default='/home/nick/anytime_learning/intermediate_features_npy_layer_0', help='Path to store the encoded features')
        parser.add_argument('--subsets', type=str, default='train', choices=['train', 'test', 'train_test'], help='Subset to encode')
        parser.parse_known_args()
        return parser


if __name__ == "__main__":
    seed_everything(42)

    opt = BaseOptions()
    module_list = [
        LearnableCLIPModule,
        MemoryModule,
        CustomOptions,
    ]
    args = opt.parse(module_list, is_train=True)
    print(args)

    config_file = os.path.join(dir_path, args.dinov2_config_file)

    store_path = args.store_folder
    train = 'train' in args.subsets
    val = 'test' in args.subsets
    features_of_the_n_layer_to_last = args.dinov2_train_transformer_block_to_last_index

    cfg = load_and_merge_config(config_file)

    dino_encoder, encoder_embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    dino_encoder = dino_encoder.to(args.device)

    weights = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    # load the weights
    dino_encoder.load_state_dict(weights.state_dict())

    # load transform
    transform = make_classification_eval_transform()
    cl_train_datasets, cl_test_datasets = get_single_image_datasets(args, transform=transform)

    if train:
        for dataset in cl_train_datasets:
            print(f"Processing for train dataset {dataset.name}")
            get_dino_features(
                dino_encoder,
                dataset,
                store_folder=store_path,
                dataset_type="train",
                features_of_the_n_layer_to_last=features_of_the_n_layer_to_last,
            )

    if val:
        for dataset in cl_test_datasets:
            print(f"Processing for test dataset {dataset.name}")
            get_dino_features(
                dino_encoder,
                dataset,
                store_folder=store_path,
                dataset_type="test",
                features_of_the_n_layer_to_last=features_of_the_n_layer_to_last,
            )
