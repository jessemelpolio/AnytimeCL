Here are the key parameters grouped by their purpose:

### General Settings
- `--exp_name`: Experiment name for logging purposes
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: GPU device(s) to use (default: [0])
- `--num_workers`: Number of data loading workers (default: 1)

### Data Configuration
- `--dataroot`: Root directory for datasets (default: "/data/owcl_data")
- `--datasets`: Comma-separated list of datasets to use (default: "CIFAR100,SUN397,EuroSAT,OxfordIIITPet,Flowers102,FGVCAircraft,StanfordCars,Food101")
- `--held_out_dataset`: Dataset to use for held-out evaluation (default: "ImageNet,UCF101,DTD")
- `--input_size`: Input image size (default: 224)

### Model Configuration
- `--network_arc`: Network architecture to use (default: "clip")
- `--backbone`: Backbone network (e.g., 'ViT-B/32' for CLIP, 'vitb14' for DINO)

### Continual Learning Settings
- `--incremental`: Incremental learning scenario (default: "dataset", choices: ["dataset", "class", "task"]), dataset represents task incremental learning
- `--num_classes`: Number of classes per stage for class-incremental learning (default: 100)
- `--randomize`: Randomize class order (default: True)

### Training Parameters
- `--optimizer`: Optimizer to use (default: "adamw")
- `--batch_size`: Batch size for training (default: 2048)
- `--lr`: Learning rate (default: 6e-4)
- `--momentum`: Momentum for optimizer (default: 0.9)
- `--weight_decay`: Weight decay for optimizer (default: 0.05)
- `--n_epochs`: Number of training epochs (default: 20)
- `--criteria`: Loss criteria to use (default: "osce", choices: ["cs", "osce", "osce_other"])
- `--X_format`: Input data format (default: "feature", choices: ["image", "feature", "embedding", "code"])

### AnytimeCL Specific Options
- `--learning_strategy`: Learning strategy (default: "online", choices: ["online", "offline", "wake_sleep", "none"])
- `--wake_bs`: Batch size for wake training (default: 32)
- `--wake_evaluation_iter_ratio`: Ratio of iterations for wake evaluation (default: 0.25)
- `--sampler_type`: Type of sampler to use (default: "class_balanced", choices: ["weighted", "none", "fifo", "class_balanced", "uniform"])

### Compression Options
- `--need_compress`: Enable feature compression (default: False)
- `--CLS_weight`: Use CLS token weight for compression (default: False)
- `--per_instance`: Perform per-instance compression (default: True)
- `--int_quantize`: Enable integer quantization for compression (default: False)
- `--components`: Number of components for compression (default: 5)
- `--int_range`: Integer range for quantization (default: 255)

### Evaluation and Logging
- `--results_dir`: Directory to save results (default: "./results")
- `--log_interval`: Interval for logging during training (default: 10)
- `--save_interval`: Interval for saving model checkpoints (default: 5)
- `--eval_interval`: Interval for evaluation during training (default: 5)
- `--eval_scenario`: Evaluation scenario (default: "cumulative_cumulative")

### Miscellaneous
- `--include_the_other_class`: Include "other" class in classification (action: store_true)
- `--use_other_classifier`: Use a separate classifier for the "other" class (action: store_true)
- `--use_tuned_text_embedding`: Use tuned text embedding (action: store_true)
- `--accumulating_data_to_the_final_stage`: Accumulate data to the final stage (action: store_true)
- `--ema_exemplar_per_class_acc`: Use EMA for exemplar per-class accuracy (action: store_true)
- `--ema_exemplar_per_class_acc_decay`: Decay rate for EMA exemplar per-class accuracy (default: 0.9)
- `--fix_finetuned_model`: Fix the fine-tuned model (action: store_true)

For a complete list of options and their descriptions, please refer to the `options/` directory in the source code and the `modify_commandline_options` method in each module.