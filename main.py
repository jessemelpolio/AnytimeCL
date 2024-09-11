import sys
import os
# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import csv
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from options.base_options import BaseOptions
from models.clip_module import LearnableCLIPModule
from models.memory_module import MemoryModule
from engines.engine import AnytimeCLEngine
from data.tasks import (
    get_single_npy_continual_learning_dataset,
    get_single_npy_held_out_dataset,
    get_zero_shot_task,
    get_union_task,
    get_union_zero_shot_task,
    get_mix_task,
    get_mix_zero_shot_task,
    get_dino_zero_shot_task,
    get_dino_union_task,
    get_dino_union_zero_shot_task,
    get_dino_mix_task,
    get_dino_mix_zero_shot_task,
    get_dino_clip_npy_continual_learning_dataset,
    get_single_npy_held_out_compression_dataset,
    get_single_npy_continual_learning_compression_dataset,

    get_dino_clip_npy_held_out_dataset,
)

def seed_everything(seed):
    """
    Seed all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """
    Main function to run the training and evaluation process.
    """
    opt = BaseOptions()
    module_list = [MemoryModule, AnytimeCLEngine, LearnableCLIPModule]
    args = opt.parse(module_list, is_train=True)

    seed_everything(args.seed)

    clip_branch = LearnableCLIPModule(args).to(args.device)
    model = MemoryModule(args, clip_branch).to(args.device)
    engine = AnytimeCLEngine(args, model)

    if args.resume:
        engine.resume(args.resume_ckpt)

    logger = SummaryWriter(log_dir=args.results_dir)
    engine.logger = logger


    if args.use_dino:
        incremental_train_dataset, incremental_test_dataset = get_dino_clip_npy_continual_learning_dataset(args)
        _, held_out_test_datasets = get_dino_clip_npy_held_out_dataset(args, load_train=False)
    else:
        # TODO: compression is not supported for dino for now.
        if args.need_compress:
            incremental_train_dataset, incremental_test_dataset = get_single_npy_continual_learning_compression_dataset(args)
            _, held_out_test_datasets = get_single_npy_held_out_compression_dataset(args, load_train=False)
        else:
            incremental_train_dataset, incremental_test_dataset = get_single_npy_continual_learning_dataset(args)
            _, held_out_test_datasets = get_single_npy_held_out_dataset(args, load_train=False)

    alpha_keys = ["exemplar"] if args.use_dino else ["p_ft", "p_other", "p_ft_and_p_other", "aim", "tuned", "original", "p_ft_0_1"]

    SEEN = "seen"
    UNSEEN = "unseen"

    alpha_key_acc_lists = {key: {"overall": [], SEEN: [], UNSEEN: []} for key in alpha_keys}

    for i in range(incremental_train_dataset.num_stages):
        print(f"Start wake training stage {i}")
        engine.wake_fit(
            incremental_train_dataset,
            stage=i,
            test_datasets=incremental_test_dataset,
            evaluation_tags=["target_dataset"] * len(incremental_test_dataset),
            # wake_batch_train_outside_control=False,
            wake_batch_train_outside_control=(args.learning_strategy not in ["offline", "none"] and 
                                              (not args.accumulating_data_to_the_final_stage or i == incremental_train_dataset.num_stages - 1)),
            alpha_keys=alpha_keys,
        )

        if not args.accumulating_data_to_the_final_stage or i == incremental_train_dataset.num_stages - 1:
            evaluate_and_record(engine, incremental_test_dataset, held_out_test_datasets, i, alpha_keys, alpha_key_acc_lists, SEEN, UNSEEN)

            if args.learning_strategy in ["offline", "wake_sleep"]:
                print(f"Start sleep training stage {i}")
                engine.sleep_fit(
                    incremental_train_dataset,
                    test_datasets=incremental_test_dataset,
                    evaluation_tags=["target_dataset"] * len(incremental_test_dataset),
                    stage=i,
                )
                evaluate_and_record(engine, incremental_test_dataset, held_out_test_datasets, i, alpha_keys, alpha_key_acc_lists, SEEN, UNSEEN)

        incremental_train_dataset.forward_stage()

    save_results(args, alpha_key_acc_lists, incremental_train_dataset, SEEN, UNSEEN)

    if args.incremental == "dataset":
        flexible_inference(args, engine, alpha_keys)

def evaluate_and_record(engine, incremental_test_dataset, held_out_test_datasets, stage, alpha_keys, alpha_key_acc_lists, SEEN, UNSEEN):
    """
    Evaluate and record the performance of the model.

    Args:
        engine (LearnableEngine): The engine used for evaluation.
        incremental_test_dataset (Dataset): The dataset used for incremental testing.
        held_out_test_datasets (Dataset): The dataset used for held-out testing.
        stage (int): The current stage of training.
        alpha_keys (list): List of alpha keys for evaluation.
        alpha_key_acc_lists (dict): Dictionary to store accuracy lists.
        SEEN (str): Label for seen data.
        UNSEEN (str): Label for unseen data.
    """
    evaluate_dataset(engine, incremental_test_dataset, stage, alpha_keys, alpha_key_acc_lists, "target_dataset", SEEN, UNSEEN)
    evaluate_dataset(engine, held_out_test_datasets, stage, alpha_keys, alpha_key_acc_lists, "held_out_dataset", SEEN, UNSEEN)

def evaluate_dataset(engine, datasets, stage, alpha_keys, alpha_key_acc_lists, dataset_type, SEEN, UNSEEN):
    """
    Evaluate a dataset and update accuracy lists.

    Args:
        engine (LearnableEngine): The engine used for evaluation.
        datasets (Dataset or list): The dataset(s) to evaluate.
        stage (int): The current stage of training.
        alpha_keys (list): List of alpha keys for evaluation.
        alpha_key_acc_lists (dict): Dictionary to store accuracy lists.
        dataset_type (str): Type of the dataset (e.g., "target_dataset").
        SEEN (str): Label for seen data.
        UNSEEN (str): Label for unseen data.
    """
    if isinstance(datasets, list):
        acc_dict = {key: {"overall": [], SEEN: [], UNSEEN: []} for key in alpha_keys}
        for j, dataset in enumerate(datasets):
            acc = engine.evaluate(
                [dataset],
                epoch=stage,
                evaluation_tags=[f"post_training_{dataset_type}"],
                stage=j,
                cross_validation=j == 0,
                alpha_keys=alpha_keys,
            )
            update_acc_dict(acc_dict, acc, alpha_keys, dataset_type, SEEN, UNSEEN)
        update_alpha_key_acc_lists(alpha_key_acc_lists, acc_dict, alpha_keys, SEEN, UNSEEN)
    else:
        acc = engine.evaluate(
            [datasets],
            epoch=stage,
            evaluation_tags=[f"post_training_{dataset_type}"],
            cross_validation=True,
            stage=stage,
            alpha_keys=alpha_keys,
        )
        update_alpha_key_acc_lists_single(alpha_key_acc_lists, acc, alpha_keys, dataset_type, SEEN, UNSEEN)

def update_acc_dict(acc_dict, acc, alpha_keys, dataset_type, SEEN, UNSEEN):
    """
    Update the accuracy dictionary with new evaluation results.

    Args:
        acc_dict (dict): Dictionary to store accuracy results.
        acc (dict): Evaluation results.
        alpha_keys (list): List of alpha keys for evaluation.
        dataset_type (str): Type of the dataset (e.g., "target_dataset").
        SEEN (str): Label for seen data.
        UNSEEN (str): Label for unseen data.
    """
    for alpha_key in alpha_keys:
        for metric in ["overall", SEEN, UNSEEN]:
            acc_dict[alpha_key][metric].append(acc[f"post_training_{dataset_type}"][alpha_key][metric])

def update_alpha_key_acc_lists(alpha_key_acc_lists, acc_dict, alpha_keys, SEEN, UNSEEN):
    """
    Update the alpha key accuracy lists with new evaluation results.

    Args:
        alpha_key_acc_lists (dict): Dictionary to store accuracy lists.
        acc_dict (dict): Dictionary with new accuracy results.
        alpha_keys (list): List of alpha keys for evaluation.
        SEEN (str): Label for seen data.
        UNSEEN (str): Label for unseen data.
    """
    for alpha_key in alpha_keys:
        for metric in ["overall", SEEN, UNSEEN]:
            alpha_key_acc_lists[alpha_key][metric].append(np.mean(acc_dict[alpha_key][metric]))

def update_alpha_key_acc_lists_single(alpha_key_acc_lists, acc, alpha_keys, dataset_type, SEEN, UNSEEN):
    """
    Update the alpha key accuracy lists with new evaluation results for a single dataset.

    Args:
        alpha_key_acc_lists (dict): Dictionary to store accuracy lists.
        acc (dict): Evaluation results.
        alpha_keys (list): List of alpha keys for evaluation.
        dataset_type (str): Type of the dataset (e.g., "target_dataset").
        SEEN (str): Label for seen data.
        UNSEEN (str): Label for unseen data.
    """
    for alpha_key in alpha_keys:
        for metric in ["overall", SEEN, UNSEEN]:
            alpha_key_acc_lists[alpha_key][metric].append(acc[f"post_training_{dataset_type}"][alpha_key][metric])

def save_results(args, alpha_key_acc_lists, incremental_train_dataset, SEEN, UNSEEN):
    """
    Save the evaluation results to CSV files.

    Args:
        args (Namespace): The arguments for the current run.
        alpha_key_acc_lists (dict): Dictionary to store accuracy lists.
        incremental_train_dataset (Dataset): The incremental training dataset.
        SEEN (str): Label for seen data.
        UNSEEN (str): Label for unseen data.
    """
    for alpha_key, acc_list in alpha_key_acc_lists.items():
        with open(os.path.join(args.results_dir, f"{alpha_key}.csv"), "a") as outfile:
            writer = csv.writer(outfile)
            if os.stat(os.path.join(args.results_dir, f"{alpha_key}.csv")).st_size == 0:
                writer.writerow(
                    ["dataset name", "split"]
                    + ["wake", "held_out", "sleep", "held_out"]
                    * incremental_train_dataset.num_stages
                )
            writer.writerow([incremental_train_dataset.name, "overall"] + acc_list["overall"])
            writer.writerow([incremental_train_dataset.name, SEEN] + acc_list[SEEN])
            writer.writerow([incremental_train_dataset.name, UNSEEN] + acc_list[UNSEEN])

def flexible_inference(args, engine, alpha_keys):
    """
    Perform flexible inference on various tasks and save the results.

    Args:
        args (Namespace): The arguments for the current run.
        engine (LearnableEngine): The engine used for evaluation.
        alpha_keys (list): List of alpha keys for evaluation.
    """
    if args.use_dino:
        tasks = [
            ("zero_shot", get_dino_zero_shot_task(args)),
            ("union", get_dino_union_task(args)),
            ("union_zero_shot", get_dino_union_zero_shot_task(args)),
            ("mix", get_dino_mix_task(args)),
            ("mix_zero_shot", get_dino_mix_zero_shot_task(args))
        ]
    else:
        tasks = [
            ("zero_shot", get_zero_shot_task(args)),
            ("union", get_union_task(args)),
            ("union_zero_shot", get_union_zero_shot_task(args)),
            ("mix", get_mix_task(args)),
            ("mix_zero_shot", get_mix_zero_shot_task(args))
        ]

    flexible_task_acc = {key: [] for key in alpha_keys}

    for task_name, task in tasks:
        task_acc = evaluate_flexible_task(engine, task, task_name, alpha_keys)
        for alpha_key in alpha_keys:
            flexible_task_acc[alpha_key].append(np.mean(task_acc[alpha_key]))

    save_flexible_inference_results(args, flexible_task_acc)

def evaluate_flexible_task(engine, task, task_name, alpha_keys):
    """
    Evaluate a flexible task and return the accuracy results.

    Args:
        engine (LearnableEngine): The engine used for evaluation.
        task (Dataset or list or tuple): The task(s) to evaluate.
        task_name (str): Name of the task.
        alpha_keys (list): List of alpha keys for evaluation.

    Returns:
        dict: Dictionary with accuracy results for each alpha key.
    """
    task_acc = {key: [] for key in alpha_keys}
    if isinstance(task, list):
        for i, subtask in enumerate(task):
            acc = engine.evaluate([subtask], epoch=i, evaluation_tags=[task_name], stage=i, cross_validation=i==0, alpha_keys=alpha_keys)
            for alpha_key in alpha_keys:
                task_acc[alpha_key].append(acc[task_name][alpha_key]["overall"])
    elif isinstance(task, tuple):  # Handle the case when task is a tuple (main_task, zero_shot_task)
        assert task_name == "union_zero_shot", "This only works for Union+Zero-shot evaluation"
        main_task, zero_shot_task = task
        # Evaluate the main task
        temp_acc = {key: [] for key in alpha_keys}
        for i in range(main_task.num_stages):
            acc = engine.evaluate(main_task, epoch=i, evaluation_tags=[task_name], stage=i, cross_validation=False, alpha_keys=alpha_keys)
            for alpha_key in alpha_keys:
                temp_acc[alpha_key].append(acc[task_name][alpha_key]["overall"])
            main_task.forward_stage()
        # Average the accuracy of the main task
        for alpha_key in alpha_keys:
            task_acc[alpha_key].append(np.mean(temp_acc[alpha_key]))
        # Evaluate the zero-shot task
        acc = engine.evaluate(zero_shot_task, epoch=i, evaluation_tags=[task_name], stage=i, cross_validation=False, alpha_keys=alpha_keys)
        # Append the accuracy of the zero-shot task
        for alpha_key in alpha_keys:
            task_acc[alpha_key].append(acc[task_name][alpha_key]["overall"])
    else:
        for i in range(task.num_stages):
            acc = engine.evaluate([task], epoch=i, evaluation_tags=[task_name], stage=i, cross_validation=False, alpha_keys=alpha_keys)
            for alpha_key in alpha_keys:
                task_acc[alpha_key].append(acc[task_name][alpha_key]["overall"])
            task.forward_stage()
    return task_acc

def save_flexible_inference_results(args, flexible_task_acc):
    """
    Save the flexible inference results to CSV files.

    Args:
        args (Namespace): The arguments for the current run.
        flexible_task_acc (dict): Dictionary with accuracy results for each alpha key.
    """
    flexible_inference_dir = os.path.join(args.results_dir, "flexible_inference")
    os.makedirs(flexible_inference_dir, exist_ok=True)
    for alpha_key, task_acc in flexible_task_acc.items():
        with open(os.path.join(flexible_inference_dir, f"{alpha_key}_flexible_inference.csv"), "a") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["zero_shot", "union", "union_zero_shot", "mix", "mix_zero_shot"])
            writer.writerow(task_acc)

if __name__ == "__main__":
    main()
