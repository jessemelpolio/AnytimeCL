import os
import os.path as osp
import sys
import csv
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from .base_engine import BaseEngine
from encode_features.compression_interface import decompress_from_npz

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from models.losses import (
    CosineSimilarityLoss,
    OpenSetCrossEntropyLoss,
    OpenSetCrossEntropyLossWithOtherLogit,
)
from models.clip_module import CLIPTrainablePart, CLIPTrainablePartPlusIncrementalClassifier
from models.samplers import (
    RecencyWeightedSampler,
    FIFOSampler,
    ClassBalancedSampler,
    UniformSampler,
)
from engines.engine_utils import DynamicEMARecorder
from models.dinov2_module import DINOV2Module


class SleepPhaseDataset(torch.utils.data.Dataset):
    def __init__(self, raw_images, labels=None, label_features=None, X_format="image", new=None, args=None):
        self.raw_images = raw_images
        self.label_features = label_features
        self.labels = labels
        self.X_format = X_format
        self.new = new
        self.args = args
        
        # we need to remap the labels to 0, 1, 2, 3, ...
        if self.labels is not None:
            self.label_to_index = {}
            index = 0
            for label in self.labels:
                label = label.item()
                if label not in self.label_to_index.keys():
                    self.label_to_index[label] = index
                    index += 1
            self.labels = [self.label_to_index[label.item()] for label in self.labels]

        self.index_to_label = {index: label for label, index in self.label_to_index.items()}

        self.class_to_idx = {label: index for index, label in self.index_to_label.items()}

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, index):
        label_feature = []
        if self.label_features is not None:
            label_feature = self.label_features[index]
        label = self.labels[index] if self.labels is not None else []

        if self.new is not None:
            new = self.new[index]
            self.new[index] = False
        else:
            new = False

        raw_image = self.raw_images[index]
        
        if self.args.need_compress:
            if isinstance(raw_image, str):
                if self.X_format == "feature":
                    raw_image = decompress_from_npz(self.args, raw_image)
            elif isinstance(raw_image, list):
                if self.X_format == "feature":
                    image_list = []
                    for image in raw_image:
                        image_list.append(decompress_from_npz(self.args, image))
                    raw_image = image_list
            else:
                raise NotImplementedError

        else:
            if isinstance(raw_image, str):
                if self.X_format == "feature":
                    raw_image = np.load(raw_image)
                    raw_image = torch.from_numpy(raw_image)
            elif isinstance(raw_image, list):
                if self.X_format == "feature":
                    raw_image = [torch.from_numpy(np.load(image)) for image in raw_image]
            else:
                raise NotImplementedError

        return raw_image, label, label_feature, self.index_to_label[label], new


class AnytimeCLEngine(BaseEngine):
    def __init__(self, args, model):
        """
        Initialize the AnytimeCLEngine.

        Args:
            args: Command-line arguments.
            model: The main model to be used for training and inference.
        """
        super().__init__(args)
        self.model = model
        self.args = args
        self.original_clip_branch = self.model.clip_branch

        self.original_trainable_part_model = CLIPTrainablePart(
            args,
            self.model.clip_branch.transformer_trainable_part,
            self.model.clip_branch.clip_encoder.visual.ln_post,
            self.model.clip_branch.clip_encoder.visual.proj,
        ).to(self.device)

        if not hasattr(self.args, 'use_dino'):
            self.args.use_dino = False

        if self.args.use_dino:
            mapping_dim = self.model.clip_branch.clip_encoder.text_projection.shape[-1]
            self.original_dino_encoder = DINOV2Module(self.args, mapping_dim=mapping_dim).to(self.device)

        self.criteria = args.criteria
        self.loss_func = self._get_loss_function()

    def _get_loss_function(self):
        """
        Get the appropriate loss function based on the specified criteria.

        Returns:
            torch.nn.Module: The selected loss function.

        Raises:
            NotImplementedError: If the specified criteria is not implemented.
        """
        if self.args.criteria == "cs":
            return CosineSimilarityLoss().to(self.device)
        elif self.args.criteria == "osce":
            return OpenSetCrossEntropyLoss().to(self.device)
        elif self.args.criteria == "osce_other":
            return OpenSetCrossEntropyLossWithOtherLogit(self.args).to(self.device)
        else:
            raise NotImplementedError(f"Criteria {self.args.criteria} not implemented")

    @staticmethod
    def modify_commandline_options(parser):
        """
        Modify the command line options parser by adding engine-specific arguments.

        Args:
            parser (argparse.ArgumentParser): The argument parser to modify.

        Returns:
            argparse.ArgumentParser: The modified argument parser.
        """
        # Model architecture and training strategy
        parser.add_argument("--criteria", type=str, default="osce", choices=["cs", "osce", "osce_other"])
        parser.add_argument("--X_format", type=str, default="feature", choices=["image", "feature", "embedding", "code"])
        parser.add_argument("--learning_strategy", type=str, default="online", 
                            choices=["online", "offline", "wake_sleep", "none"])
        parser.add_argument("--use_tuned_text_embedding", action="store_true")
        parser.add_argument("--fix_finetuned_model", action="store_true")

        # Other class handling
        parser.add_argument("--include_the_other_class", action="store_true")
        parser.add_argument("--other_class_calibration_loss_weight", type=float, default=0.1)
        parser.add_argument("--use_other_classifier", action="store_true")

        # Wake-sleep cycle parameters
        parser.add_argument("--wake_evaluation_iter_ratio", type=float, default=0.25)
        parser.add_argument("--wake_bs", type=int, default=32)
        parser.add_argument("--accumulating_data_to_the_final_stage", action="store_true")

        # Sampling strategy
        parser.add_argument("--sampler_type", type=str, default="class_balanced", 
                            choices=["weighted", "none", "fifo", "class_balanced", "uniform"])
        parser.add_argument("--weighted_sampler_decay_rate", type=float, default=0.99)

        # EMA-related parameters
        parser.add_argument("--ema_exemplar_per_class_acc", action="store_true")
        parser.add_argument("--ema_exemplar_per_class_acc_decay", type=float, default=0.99)

        # DINOv2-specific arguments
        parser.add_argument("--use_dino", action="store_true")
        parser.add_argument("--dinov2_train_transformer_block_to_last_index", type=int, default=1)
        parser.add_argument("--dinov2_backbone", type=str, default="dinov2_vitb14")
        parser.add_argument("--dinov2_config_file", type=str, default="eval/vitb14_pretrain")
        parser.add_argument("--dinov2_data_root", type=str, default="/data/owcl_data/dino_intermediate_features_npy")
        
        return parser

    def configure_optimizers(self, for_batch_train=False, **kwargs):
        """
        Configure the optimizer for training.

        Args:
            for_batch_train (bool): Whether the optimizer is being configured for batch training.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Adjust learning rate if we're doing batch training
        lr = self.args.lr * self.args.wake_bs / self.args.batch_size if for_batch_train else self.args.lr
        optimizer_params = {
            'params': self.learnable_parameters,
            'lr': lr,
            'weight_decay': self.args.weight_decay
        }
        
        # Create optimizer based on the specified type
        if self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(**optimizer_params)
        elif self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(**optimizer_params)
        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(**optimizer_params, momentum=self.args.momentum)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optimizer} not implemented")

    def configure_train_dataloader(self, train_dataset, sample_buffer, for_batch_train=False, **kwargs):
        """
        Configure the data loader for training.

        Args:
            train_dataset: The training dataset.
            sample_buffer: Buffer containing samples for training.
            for_batch_train (bool): Whether the data loader is being configured for batch training.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Create a dataset for the sleep phase
        self.sleep_train_dataset = SleepPhaseDataset(
            sample_buffer["raw_images"],
            sample_buffer["labels"],
            sample_buffer["label_features"],
            X_format=self.args.X_format,
            new=sample_buffer["new"],
            args=self.args
        )

        # Get the appropriate sampler
        sampler = self._get_sampler(sample_buffer, for_batch_train)

        # Create the DataLoader
        if isinstance(sampler, torch.utils.data.Sampler):
            self.sleep_train_loader = DataLoader(
                self.sleep_train_dataset,
                batch_sampler=sampler,
                num_workers=1
            )
        else:
            batch_size = self.args.wake_bs if for_batch_train else self.args.batch_size
            self.sleep_train_loader = DataLoader(
                self.sleep_train_dataset,
                batch_size=batch_size,
                shuffle=(sampler == "shuffle"),
                sampler=None if sampler == "shuffle" else sampler,
                num_workers=1
            )

        self._process_label_features(sample_buffer)

    def _get_sampler(self, sample_buffer, for_batch_train):
        """
        Get the appropriate sampler for the data loader based on the configuration.

        Args:
            sample_buffer (dict): Buffer containing samples and their metadata.
            for_batch_train (bool): Whether the sampler is for batch training.

        Returns:
            torch.utils.data.Sampler or str: The selected sampler or "shuffle" for random sampling.

        Raises:
            NotImplementedError: If the specified sampler type is not implemented.
        """
        # If not batch training and not using weighted sampler, use shuffle
        if not for_batch_train and self.args.sampler_type != "weighted":
            return "shuffle"

        sampler_params = {
            'new': sample_buffer["new"],
            'batch_size': self.args.wake_bs if for_batch_train else self.args.batch_size,
            'num_new_samples_per_batch': 1
        }

        # Create the appropriate sampler based on the specified type
        if self.args.sampler_type == "weighted":
            return RecencyWeightedSampler(
                **sampler_params,
                initial_weights=sample_buffer["sample_weights"],
                initial_counts=sample_buffer["sample_counts"],
                decay_rate=self.args.weighted_sampler_decay_rate,
                min_weight=0.01,
                use_weights_only=not for_batch_train
            )
        elif self.args.sampler_type == "fifo":
            return FIFOSampler(**sampler_params)
        elif self.args.sampler_type == "class_balanced":
            return ClassBalancedSampler(sample_buffer["labels"], **sampler_params)
        elif self.args.sampler_type == "uniform":
            return UniformSampler(**sampler_params)
        else:
            raise NotImplementedError(f"Sampler type {self.args.sampler_type} not implemented")

    def _process_label_features(self, sample_buffer):
        """
        Process label features from the sample buffer and prepare them for training.

        Args:
            sample_buffer (dict): Buffer containing samples and their metadata.

        Returns:
            None
        """
        # Create a dictionary of unique label features
        label_features = {}
        for i, label in enumerate(sample_buffer["labels"]):
            label = label.item()
            if label not in label_features:
                label_features[label] = sample_buffer["label_features"][i]

        # Stack the unique label features and move to device
        self.train_target_class_features = torch.stack(list(label_features.values()), dim=0).to(self.device)
        self.num_classes = len(label_features)

    def configure_learnable_params(self, model=None, param_keys="all", requires_grad=True, **kwargs):
        """
        Configure the learnable parameters of the model.

        Args:
            model: The model whose parameters are to be configured. If None, a new model is created.
            param_keys (str): Keys of the parameters to be configured.
            requires_grad (bool): Whether the parameters require gradients.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if model is None:
            trainable_part_model = self._create_exemplar_model()
        else:
            trainable_part_model = model.to(self.device)

        # Set all model parameters to non-trainable
        for param in self.model.parameters():
            param.requires_grad = False

        self.learnable_parameters = []

        if self.args.use_dino:
            self._configure_dinov2_params(trainable_part_model)
        else:
            self._configure_finetuned_model_params(trainable_part_model)

        if self.args.include_the_other_class:
            self._configure_other_class_params(trainable_part_model)

        if self.args.use_tuned_text_embedding:
            self._configure_linear_classifier_params(trainable_part_model)

        self.trainable_part_model = trainable_part_model

    def _configure_finetuned_model_params(self, trainable_part_model):
        """
        Configure parameters for the finetuned model part.

        Args:
            trainable_part_model (torch.nn.Module): The trainable part of the model.

        Returns:
            None
        """
        # Configure parameters for  fine-tuning
        if self.args.fix_finetuned_model:
            return
        for param in trainable_part_model.block.parameters():
            param.requires_grad = True
        self.learnable_parameters.extend(trainable_part_model.block.parameters())

    def _configure_other_class_params(self, trainable_part_model):
        """
        Configure parameters for the "other" class classifier.

        Args:
            trainable_part_model (torch.nn.Module): The trainable part of the model.

        Returns:
            None
        """
        # Configure parameters for the "other" class
        if self.args.use_other_classifier:
            for param in trainable_part_model.other_classifier.parameters():
                param.requires_grad = True
            self.learnable_parameters.extend(trainable_part_model.other_classifier.parameters())
        else:
            trainable_part_model.other_bias.requires_grad = True
            self.learnable_parameters.append(trainable_part_model.other_bias)

    def _configure_linear_classifier_params(self, trainable_part_model):
        """
        Configure the linear classifier parameters.

        Args:
            trainable_part_model (torch.nn.Module): The trainable part of the model.

        Returns:
            None
        """
        # Configure the linear classifier
        trainable_part_model.adaptation(self.train_target_class_features)
        for param in trainable_part_model.incremental_classifier.parameters():
            param.requires_grad = True
        self.learnable_parameters.extend(trainable_part_model.incremental_classifier.parameters())

    def _configure_dinov2_params(self, trainable_part_model):
        """
        Configure parameters for the DINOv2 model.

        Args:
            trainable_part_model (torch.nn.Module): The trainable part of the model.

        Returns:
            None
        """
        for param in trainable_part_model.parameters():
            param.requires_grad = False

        for block in trainable_part_model.dino_encoder.blocks[-self.args.dinov2_train_transformer_block_to_last_index:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in trainable_part_model.mapping.parameters():
            param.requires_grad = True

        self.learnable_parameters = []
        for block in trainable_part_model.dino_encoder.blocks[-self.args.dinov2_train_transformer_block_to_last_index:]:
            self.learnable_parameters += list(block.parameters())
        self.learnable_parameters += list(trainable_part_model.mapping.parameters())

    @staticmethod
    def similarity_calculation(features, target_label_features):
        """
        Calculate similarity between features and target label features.

        Args:
            features (torch.Tensor): Input features.
            target_label_features (torch.Tensor): Target label features.

        Returns:
            torch.Tensor: Calculated similarities.
        """
        # Calculate similarity in chunks to avoid memory issues
        chunk_size = 1000
        similarities = []
        for i in range(0, target_label_features.shape[0], chunk_size):
            chunk = target_label_features[i : i + chunk_size]
            sim = 100 * features @ chunk.T
            similarities.append(sim)
        return torch.cat(similarities, dim=-1).softmax(dim=-1)

    def unpack_batch(self, batch, training=False):
        """
        Unpack a batch of data and prepare it for processing.

        Args:
            batch (tuple): A batch of data from the data loader.
            training (bool): Whether the unpacking is for training or inference.

        Returns:
            None
        """
        if training:
            # Unpack the batch and move relevant parts to the device
            self.X, self.y = batch[:2]
            self.y = self.y.to(self.device)
            
            # Handle DINO and CLIP inputs separately
            if self.args.use_dino:
                self.X_clip = self.X[0].to(self.device)  # CLIP input
                self.X_dino = self.X[-1].to(self.device).squeeze(1)  # DINO input
            else:
                self.X_clip = self.X[0].to(self.device)  # Only CLIP input

            # Handle label features if present
            if len(batch) > 2 and batch[2] is not None:
                self.text_y = batch[2].to(self.device)
            else:
                self.text_y = None

            # Store original targets if present
            if len(batch) > 3:
                self.original_y = batch[3]
            else:
                self.original_y = None

            # Store information about new samples if present
            if len(batch) > 4:
                self.new = batch[4]
            else:
                self.new = None
        else:
            # image, label, text_targets, path
            # image can be a tensor or a list of tensors
            X = batch[0]
            self.y = batch[1].to(self.device)
            # here text_y is a string
            self.text_y = batch[2]
            self.path = batch[3]
            
            if isinstance(X, list):
                if self.args.use_dino:
                    assert len(X) == 2
                    self.X_clip = X[0].to(self.device)
                    self.X_dino = X[-1].to(self.device).squeeze(1)
                else:
                    assert len(X) == 1
                    self.X_clip = X[0].to(self.device)
            else:
                self.X_clip = X.to(self.device)

    def dino_forward_core(self, X_format, trainable_model, input_X):
        """
        Perform forward pass for DINO model.

        Args:
            X_format (str): Format of the input data.
            trainable_model (torch.nn.Module): The trainable model.
            input_X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the forward pass.

        Raises:
            NotImplementedError: If the X_format is not supported for DINO.
        """
        # Forward pass for DINO model
        if X_format == "feature":
            zs_out = trainable_model(input_X)
        else:
            raise NotImplementedError("Only 'feature' X_format is supported for DINO")
        return zs_out

    def clip_forward_core(self, X_format, trainable_model, input_X):
        """
        Perform forward pass for CLIP model.

        Args:
            X_format (str): Format of the input data.
            trainable_model (torch.nn.Module): The trainable model.
            input_X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the forward pass.

        Raises:
            NotImplementedError: If the X_format is not supported.
        """
        # Forward pass for CLIP model
        if X_format == "code":
            # Handle code input format
            zs_out = self.model.quantizer.get_codes_from_indices(input_X.unsqueeze(-1))
            zs_out = self.model.quantizer.project_out(zs_out)
            zs_out = zs_out.permute(1, 0, 2)  # Permute from NLD to LND
            zs_out = trainable_model(zs_out)
        elif X_format == "embedding":
            # Handle embedding input format
            zs_out = input_X
        elif X_format == "feature":
            # Handle feature input format
            input_X = input_X.permute(1, 0, 2)  # Permute from NLD to LND
            zs_out = trainable_model(input_X)
        elif X_format == "image":
            # Handle image input format
            zs_out = trainable_model(input_X)
        else:
            raise NotImplementedError(f"Unsupported X_format: {X_format}")
        return zs_out

    def model_forward(self, normalize=True, X_format=None, get_combined_output=True, alpha_keys=None, training=False):
        """
        Perform forward pass through the entire model.

        Args:
            normalize (bool): Whether to normalize the output.
            X_format (str): Format of the input data.
            get_combined_output (bool): Whether to get combined output for inference.
            alpha_keys (list): Keys for alpha values used in evaluation.
            training (bool): Whether the forward pass is for training.

        Returns:
            dict: A dictionary containing various outputs of the forward pass.
        """
        # Main forward pass function
        X_format = X_format or self.args.X_format
        trainable_model = getattr(self, 'trainable_part_model', self.original_trainable_part_model)
        
        # Choose between DINO and CLIP forward passes
        if self.args.use_dino:
            tuned_out = self.dino_forward_core(X_format, trainable_model, self.X_dino)
        else:
            tuned_out = self.clip_forward_core(X_format, trainable_model, self.X_clip)

        # Get zero-shot features from CLIP
        original_out = self.clip_forward_core(X_format, self.original_trainable_part_model, self.X_clip)
        
        # Get classifier output if needed (inference)
        if get_combined_output and not training:
            combined_output = self.model(self.X_dino if self.args.use_dino else self.X_clip, original_out.detach(), alpha_keys=alpha_keys)
        else:
            combined_output = None
        
        # Normalize outputs if required
        if normalize:
            if not self.args.use_tuned_text_embedding:
                tuned_out = tuned_out / tuned_out.norm(dim=-1, keepdim=True)
            if training:
                original_out = original_out / original_out.norm(dim=-1, keepdim=True)
                
        return {"tuned_out": tuned_out, "original_out": original_out, "combined_output": combined_output}

    def fit(self, train_dataset, test_datasets=None, evaluation_tags=None, stage=0, wake_fit=True, sleep_fit=True, **kwargs):
        """
        Fit the model to the training data.

        Args:
            train_dataset: The training dataset.
            test_datasets: Datasets used for testing/evaluation.
            evaluation_tags: Tags for evaluation datasets.
            stage (int): The current training stage.
            wake_fit (bool): Whether to perform wake cycle fitting.
            sleep_fit (bool): Whether to perform sleep cycle fitting.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Main fitting function that orchestrates wake and sleep cycles
        if wake_fit:
            print("Starting wake cycle...")
            self.wake_fit(train_dataset, stage, test_datasets, evaluation_tags, **kwargs)
        if sleep_fit:
            print("Starting sleep cycle...")
            self.sleep_fit(train_dataset, test_datasets, evaluation_tags, stage, **kwargs)

    def wake_fit(self, train_dataset, stage=0, test_datasets=None, evaluation_tags=None, wake_batch_train_outside_control=True, alpha_keys=None, **kwargs):
        """
        Perform wake cycle fitting.

        Args:
            train_dataset: The training dataset.
            stage (int): The current training stage.
            test_datasets: Datasets used for testing/evaluation.
            evaluation_tags: Tags for evaluation datasets.
            wake_batch_train_outside_control (bool): Whether batch training during wake is controlled externally.
            alpha_keys: Keys for alpha values used in evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.model.eval()
        dino_encoder = self.original_dino_encoder if self.args.use_dino else None
        self.model.extend_memory(
            train_dataset, clip_branch=self.original_clip_branch, dino_encoder=dino_encoder
        )
        
        # Get all trainable nodes from the memory module
        nodes = self.model.tree.get_trainable_nodes()
        for node in nodes:
            self._initialize_node(node)  # Initialize each node

        self.wake_stage_training_epoch(stage, test_datasets, evaluation_tags, wake_batch_train_outside_control, alpha_keys, **kwargs)

        torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

    def _initialize_node(self, node):
        """
        Initialize node with exemplar model and performance recorders.

        Args:
            node: The node to initialize.

        Returns:
            None
        """
        if node.exemplar_model is None:
            # Create a new exemplar model for the node if it doesn't exist
            node.exemplar_model = self._create_exemplar_model().to(self.device)

        # Initialize accuracy recorders if EMA exemplar per class accuracy is enabled
        if self.args.ema_exemplar_per_class_acc and not hasattr(node, "tuned_acc_recorder"):
            node.tuned_acc_recorder = DynamicEMARecorder(self.args.ema_exemplar_per_class_acc_decay)
            node.original_acc_recorder = DynamicEMARecorder(self.args.ema_exemplar_per_class_acc_decay)

    def _create_exemplar_model(self):
        """
        Create an exemplar model based on configuration.

        Returns:
            torch.nn.Module: The created exemplar model.
        """
        if self.args.use_dino:
            mapping_dim = self.model.clip_branch.clip_encoder.text_projection.shape[-1]
            model = DINOV2Module(self.args, mapping_dim=mapping_dim).to(self.device)
            return model
    
        # Create an exemplar model based on configuration
        if self.args.use_tuned_text_embedding:
            # Create a model with an additional linear classifier
            return CLIPTrainablePartPlusIncrementalClassifier(
                self.args,
                copy.deepcopy(self.model.clip_branch.transformer_trainable_part),
                copy.deepcopy(self.model.clip_branch.clip_encoder.visual.ln_post),
                copy.deepcopy(self.model.clip_branch.clip_encoder.visual.proj),
                self.model.enc_dim,
            )
        else:
            # Create a standard CLIP trainable part model
            return CLIPTrainablePart(
                self.args,
                copy.deepcopy(self.model.clip_branch.transformer_trainable_part),
                copy.deepcopy(self.model.clip_branch.clip_encoder.visual.ln_post),
                copy.deepcopy(self.model.clip_branch.clip_encoder.visual.proj),
            )

    def wake_stage_training_epoch(self, stage, test_datasets, evaluation_tags, wake_batch_train_outside_control, alpha_keys, **kwargs):
        """
        Train each node during the wake stage.

        Args:
            stage (int): The current training stage.
            test_datasets: Datasets used for testing/evaluation.
            evaluation_tags: Tags for evaluation datasets.
            wake_batch_train_outside_control (bool): Whether batch training during wake is controlled externally.
            alpha_keys: Keys for alpha values used in evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        nodes = self.model.tree.get_trainable_nodes()
        for node in nodes:
            self._wake_train_node(node, stage, test_datasets, evaluation_tags, wake_batch_train_outside_control, alpha_keys, **kwargs)

        torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

    def _wake_train_node(self, node, stage, test_datasets, evaluation_tags, wake_batch_train_outside_control, alpha_keys, **kwargs):
        """
        Configure and train a single node.

        Args:
            node: The node to train.
            stage (int): The current training stage.
            test_datasets: Datasets used for testing/evaluation.
            evaluation_tags: Tags for evaluation datasets.
            wake_batch_train_outside_control (bool): Whether batch training during wake is controlled externally.
            alpha_keys: Keys for alpha values used in evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Set up the data loader for the node's exemplar set
        self.configure_train_dataloader(None, node.exemplar_set, for_batch_train=True, **kwargs)
        # Configure learnable parameters for the node's exemplar model
        self.configure_learnable_params(model=node.exemplar_model, **kwargs)
        # Set up the optimizer for batch training
        self.configure_optimizers(for_batch_train=True, **kwargs)

        if wake_batch_train_outside_control:
            self.model.train()  # Set model to training mode
            # Perform a training epoch for the node
            self.train_epoch(stage, 0, test_datasets=test_datasets, evaluation_tags=evaluation_tags, wake_evaluation=True, alpha_keys=alpha_keys, node=node, **kwargs)
            self._after_train_epoch(**kwargs)  # Perform post-epoch operations

            # Update sample weights if using weighted sampler
            if self.args.sampler_type == "weighted":
                node.exemplar_set["sample_weights"] = self.sleep_train_loader.batch_sampler.weights.tolist()
                node.exemplar_set["sample_counts"] = self.sleep_train_loader.batch_sampler.sample_counts.tolist()

            # Mark all samples in the exemplar set as not new
            node.exemplar_set["new"] = [False] * len(node.exemplar_set["new"])

        print("Setting the model back to the tree...")
        self.trainable_part_model.eval()  # Set trainable part model to evaluation mode

        node.exemplar_model = copy.copy(self.trainable_part_model)  # Update node's exemplar model

        print("After batch training for node ", node.node_index)

    def sleep_fit(self, train_dataset, test_datasets=None, evaluation_tags=None, stage=0, **kwargs):
        """
        Perform sleep cycle fitting.

        Args:
            train_dataset: The training dataset.
            test_datasets: Datasets used for testing/evaluation.
            evaluation_tags: Tags for evaluation datasets.
            stage (int): The current training stage.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Sleep cycle: Train all nodes
        nodes = self.model.tree.get_trainable_nodes()

        for node in nodes:
            self._sleep_train_node(node, train_dataset, test_datasets, evaluation_tags, stage, **kwargs)

        print("After training all exemplar models...")

    def _sleep_train_node(self, node, train_dataset, test_datasets, evaluation_tags, stage, **kwargs):
        """
        Train a single node during sleep cycle.

        Args:
            node: The node to train.
            train_dataset: The training dataset.
            test_datasets: Datasets used for testing/evaluation.
            evaluation_tags: Tags for evaluation datasets.
            stage (int): The current training stage.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Configure data loader, learnable parameters, and optimizer for the node
        self.configure_train_dataloader(train_dataset, node.exemplar_set, **kwargs)
        self.configure_learnable_params(model=node.exemplar_model, **kwargs)
        self.configure_optimizers(**kwargs)

        # Train for the specified number of epochs
        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            # Evaluate if it's time to do so and test datasets are provided
            if epoch % self.args.eval_interval == 0 and test_datasets is not None:
                self._evaluate_sleep(stage, epoch, test_datasets, evaluation_tags, **kwargs)

            self.model.train()  # Set model to training mode
            # Perform a training epoch
            self.train_epoch(stage, epoch, node=node, **kwargs)
            self._after_train_epoch(**kwargs)  # Perform post-epoch operations

            # Save checkpoint if it's time to do so
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(stage, epoch, None, **kwargs)

        self._finalize_sleep_training(node, stage, test_datasets, evaluation_tags, **kwargs)

    def _evaluate_sleep(self, stage, epoch, test_datasets, evaluation_tags, **kwargs):
        """
        Evaluate model during sleep cycle.

        Args:
            stage (int): The current training stage.
            epoch (int): The current epoch.
            test_datasets: Datasets used for testing/evaluation.
            evaluation_tags: Tags for evaluation datasets.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The accuracy of the model on the test datasets.
        """
        # Evaluate model during sleep cycle
        print("Evaluating...")
        acc = self.evaluate(test_datasets, stage=stage, epoch=epoch, evaluation_tags=evaluation_tags, **kwargs)
        return acc

    def _finalize_sleep_training(self, node, stage, test_datasets, evaluation_tags, **kwargs):
        """
        Finalize training for a node after sleep cycle.

        Args:
            node: The node being trained.
            stage (int): The current training stage.
            test_datasets: Datasets used for testing/evaluation.
            evaluation_tags: Tags for evaluation datasets.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if self.args.sampler_type == "weighted":
            # Update sample weights and counts for weighted sampler
            node.exemplar_set["sample_weights"] = self.sleep_train_loader.batch_sampler.weights.tolist()
            node.exemplar_set["sample_counts"] = self.sleep_train_loader.batch_sampler.sample_counts.tolist()
            node.exemplar_set["new"] = self.sleep_train_loader.batch_sampler.new_or_old.tolist()
        else:
            # Mark all samples as not new if not using weighted sampler
            node.exemplar_set["new"] = [False] * len(node.exemplar_set["new"])

        # Perform final evaluation if test datasets are provided
        if test_datasets is not None:
            acc = self._evaluate_sleep(stage, self.args.n_epochs, test_datasets, evaluation_tags, **kwargs)
            self.save_checkpoint(stage, self.args.n_epochs, acc, **kwargs)

        torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        print("Setting the model back to the tree...")
        self.trainable_part_model.eval()  # Set trainable part model to evaluation mode
        node.exemplar_model = copy.deepcopy(self.trainable_part_model)  # Update node's exemplar model
        node.new_data_points = 0  # Reset the count of new data points

    def train_epoch(self, stage, epoch, set_train=True, test_datasets=None, evaluation_tags=None, wake_evaluation=False, alpha_keys=None, node=None, **kwargs):
        """
        Train for one epoch during sleep cycle.

        Args:
            stage (int): The current training stage.
            epoch (int): The current epoch.
            set_train (bool): Whether to set the model to training mode.
            test_datasets: Datasets used for testing/evaluation.
            evaluation_tags: Tags for evaluation datasets.
            wake_evaluation (bool): Whether this is a wake evaluation epoch.
            alpha_keys: Keys for alpha values used in evaluation.
            node: The node being trained (if applicable).
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Train for one epoch during sleep cycle
        if set_train:
            self.model.train()  # Set model to training mode
        else:
            self.model.eval()  # Set model to evaluation mode

        batch_num, train_loss, correct, total = 0, 0, 0, 0  # Initialize counters
        wake_evaluation, evaluation_interval = self._setup_wake_evaluation(wake_evaluation)  # Setup wake evaluation parameters

        # Iterate through the sleep training data loader
        for i, (batch) in enumerate(tqdm(self.sleep_train_loader)):
            # Check if wake evaluation should be performed
            if (i + 1) % evaluation_interval == 0 and wake_evaluation:
                acc = self.evaluate(test_datasets, stage=stage, epoch=i, evaluation_tags=None, alpha_keys=alpha_keys, **kwargs)
                self._log_wake_evaluation(stage, epoch, i, acc)  # Log the results

            # Process the current batch
            loss, batch_correct = self._process_batch_training(batch, node)

            # Update counters
            train_loss += loss.item()
            total += self.y.size(0)
            correct += batch_correct
            batch_num += 1

        # Log the results of this epoch
        self._log_train_epoch_results(stage, epoch, train_loss, batch_num, correct, total)
        
    def _common_forward_pass(self, batch, alpha_keys=None, training=False):
        """
        Perform a common forward pass for both training and inference.

        Args:
            batch: The input batch.
            alpha_keys: Keys for alpha values used in evaluation.
            training (bool): Whether this is a training forward pass.

        Returns:
            dict: A dictionary containing various outputs of the forward pass.
        """
        self.unpack_batch(batch, training=training)
        outputs = self.model_forward(normalize=True, get_combined_output=True, alpha_keys=alpha_keys, training=training)
        
        tuned_out = outputs["tuned_out"]
        original_out = outputs["original_out"]
        combined_out = outputs["combined_output"]
        
        if self.args.use_tuned_text_embedding:
            if training:
                tuned_similarity = tuned_out.softmax(dim=-1)
            else:
                tuned_out = self.trainable_part_model.class_features[tuned_out.argmax(dim=-1)]
                tuned_similarity = self.similarity_calculation(tuned_out.float(), self.test_target_class_features)
        else:
            target_features = self.train_target_class_features if training else self.test_target_class_features
            tuned_similarity = self.similarity_calculation(tuned_out.float(), target_features)

        tuned_predicted = tuned_similarity.argmax(dim=1)
        
        original_similarity = self.similarity_calculation(original_out.float(), self.test_target_class_features if not training else self.train_target_class_features)
        original_predicted = original_similarity.argmax(dim=1)
        
        return {
            "tuned_out": tuned_out,
            "original_out": original_out,
            "combined_out": combined_out,
            "tuned_similarity": tuned_similarity,
            "original_similarity": original_similarity,
            "tuned_predicted": tuned_predicted,
            "original_predicted": original_predicted,
        }

    def _process_batch_training(self, batch, node):
        """
        Process a batch of data during training.

        Args:
            batch: The input batch.
            node: The node being trained.

        Returns:
            tuple: A tuple containing the loss and the number of correct predictions.
        """
        self.optimizer.zero_grad()

        outputs = self._common_forward_pass(batch, training=True)
        tuned_out = outputs["tuned_out"]
        tuned_similarity = outputs["tuned_similarity"]
        tuned_predicted = outputs["tuned_predicted"]
        original_similarity = outputs["original_similarity"]
        
        loss = self._compute_loss(tuned_out, self.text_y, self.y)

        if self.args.ema_exemplar_per_class_acc:
            self._update_ema_records(node, tuned_similarity, original_similarity, self.y, self.original_y, self.new)

        loss.backward()
        self.optimizer.step()

        batch_correct = tuned_predicted.eq(self.y).sum().item()

        return loss, batch_correct

    def _process_batch_inference(self, batch, result_statistics, learned_classes_indices, alpha_keys, evaluate_seen_unseen):
        """
        Process a batch of data during inference.

        Args:
            batch: The input batch.
            result_statistics (dict): Dictionary to store evaluation statistics.
            learned_classes_indices: Indices of learned classes.
            alpha_keys: Keys for alpha values used in evaluation.
            evaluate_seen_unseen (bool): Whether to evaluate on both seen and unseen classes.

        Returns:
            None
        """
        with torch.no_grad():
            outputs = self._common_forward_pass(batch, alpha_keys=alpha_keys, training=False)
        
        self._update_statistics(result_statistics["tuned"], outputs["tuned_predicted"], evaluate_seen_unseen, learned_classes_indices)
        self._update_statistics(result_statistics["original"], outputs["original_predicted"], evaluate_seen_unseen, learned_classes_indices)
        
        if isinstance(outputs["combined_out"], dict):
            for key, value in outputs["combined_out"].items():
                similarity = self.similarity_calculation(value.float(), self.test_target_class_features)
                predicted = similarity.argmax(dim=1)
                self._update_statistics(result_statistics[key], predicted, evaluate_seen_unseen, learned_classes_indices)
        elif outputs["combined_out"] is not None:
            similarity = self.similarity_calculation(outputs["combined_out"].float(), self.test_target_class_features)
            predicted = similarity.argmax(dim=1)
            self._update_statistics(result_statistics.setdefault("complementary", self._create_stat_dict()), predicted, evaluate_seen_unseen, learned_classes_indices)

    def _compute_loss(self, tuned_out, label_features, targets):
        """
        Compute the loss based on the specified criteria.

        Args:
            tuned_out (torch.Tensor): Model outputs.
            label_features (torch.Tensor): Label features.
            targets (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        if self.args.use_tuned_text_embedding:
            return F.cross_entropy(tuned_out, targets)
        else:
            return self.sleep_criterion(tuned_out, label_features, targets)

    def _setup_wake_evaluation(self, wake_evaluation):
        """
        Setup wake evaluation parameters.

        Args:
            wake_evaluation (bool): Whether wake evaluation is enabled.

        Returns:
            tuple: A tuple containing whether wake evaluation should be performed and the evaluation interval.
        """
        if self.args.incremental != "dataset":
            return False, float('inf')  # Disable wake evaluation if not in dataset incremental mode
        # Calculate evaluation interval based on the ratio specified in args
        evaluation_interval = max(int(len(self.sleep_train_loader) * self.args.wake_evaluation_iter_ratio), 1)
        return wake_evaluation, evaluation_interval

    def _log_wake_evaluation(self, stage, epoch, i, acc):
        """
        Log wake evaluation results to a CSV file.

        Args:
            stage (int): The current training stage.
            epoch (int): The current epoch.
            i (int): The current iteration.
            acc (float): The accuracy of the model on the test datasets.

        Returns:
            None
        """
        csv_file = os.path.join(self.args.results_dir, "wake_sleep_evaluation.csv")
        with open(csv_file, "a") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Stage", stage, "Epoch", epoch, "iter", i, "acc", acc])

    def _update_ema_records(self, node, tuned_similarity, original_similarity, targets, original_targets, new):
        """
        Update Exponential Moving Average (EMA) records for tuned and original predictions.

        Args:
            node: The node being trained.
            tuned_similarity (torch.Tensor): Similarity scores for tuned predictions.
            original_similarity (torch.Tensor): Similarity scores for original predictions.
            targets (torch.Tensor): Target labels.
            original_targets (torch.Tensor): Original target labels.
            new (bool): Whether the data is new or not.

        Returns:
            None
        """
        tuned_pred = tuned_similarity.argmax(dim=-1)
        node.tuned_acc_recorder.update(tuned_pred, targets, original_targets, new)

        original_pred = original_similarity.argmax(dim=-1)
        node.original_acc_recorder.update(original_pred, targets, original_targets, new)

    def _log_train_epoch_results(self, stage, epoch, train_loss, batch_num, correct, total):
        """
        Log training results for each epoch.

        Args:
            stage (int): The current training stage.
            epoch (int): The current epoch.
            train_loss (float): The average training loss for the epoch.
            batch_num (int): The number of batches processed in the epoch.
            correct (int): The number of correct predictions in the epoch.
            total (int): The total number of predictions in the epoch.

        Returns:
            None
        """
        if self.logger is not None:
            # Log metrics to tensorboard if logger is available
            self.logger.add_scalar(f"train/stage_{stage}/loss", train_loss / batch_num, epoch)
            self.logger.add_scalar(f"train/stage_{stage}/acc", 100.0 * correct / total, epoch)
        # Print epoch results
        print(
            "Stage {} -- Train Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                stage, epoch, train_loss / batch_num, correct, total, 100.0 * correct / total,
            )
        )

    def sleep_criterion(self, outputs, label_features, targets):
        """
        Compute the loss based on the specified criteria for sleep phase.

        Args:
            outputs (torch.Tensor): Model outputs.
            label_features (torch.Tensor): Label features.
            targets (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: The computed loss.

        Raises:
            NotImplementedError: If the specified criteria is not implemented.
        """
        if self.criteria == "cs":
            # Cosine similarity loss
            label_features = F.normalize(label_features, dim=-1)
            return self.loss_func(outputs, label_features.to(self.device))
        elif self.criteria == "ce":
            # Cross-entropy loss
            return self.loss_func(outputs, targets)
        elif self.criteria == "osce":
            # Open-set cross-entropy loss
            return self.loss_func(outputs, targets, self.train_target_class_features)
        elif self.criteria == "osce_other":
            # Open-set cross-entropy loss with other logit
            other_logit = self.trainable_part_model.other_classifier(outputs) if self.args.use_other_classifier else self.trainable_part_model.other_bias
            return self.loss_func(outputs, targets, self.train_target_class_features, other_logit)
        else:
            raise NotImplementedError

    def _before_evaluate_single_dataset(self, test_dataset, stage, tag, epoch=0, **kwargs):
        """
        Prepare for evaluation of a single dataset.

        Args:
            test_dataset: The dataset to be evaluated.
            stage (int): The current evaluation stage.
            tag (str): The tag for the evaluation.
            epoch (int): The current epoch.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.wake_bs, shuffle=False, num_workers=self.args.num_workers)
        
        self.test_target_class_features = self.model.encode_class_features(test_dataset.classes_with_prompt)
        target_classes = test_dataset.classes

        self.model.set_target_classes_features(target_classes, self.test_target_class_features)
        self.model.find_mapping_from_exemplar_to_target()
        self.model.find_overlapping_indices()

        if self.args.ema_exemplar_per_class_acc:
            self._update_ema_accuracies()
            self.model.compute_tuned_and_original_p_ft()

        self.model.eval()

    def _update_ema_accuracies(self):
        """
        Update and save Exponential Moving Average (EMA) accuracies for all leaf nodes.

        Returns:
            None
        """
        nodes = self.model.tree.get_all_leaf_nodes()
        
        tuned_accs = [node.tuned_acc_recorder.get_accuracies() for node in nodes]
        original_accs = [node.original_acc_recorder.get_accuracies() for node in nodes]
        
        self.model.tuned_model_probs = tuned_accs
        self.model.original_model_probs = original_accs
        
        self.model.compute_tuned_and_original_p_ft()

    def evaluate(self, test_datasets, stage=0, epoch=0, evaluation_tags=None, cross_validation=True, evaluate_seen_unseen=True, alpha_keys=None, **kwargs):
        """
        Evaluate the model on multiple test datasets.

        Args:
            test_datasets: Datasets used for evaluation.
            stage (int): The current evaluation stage.
            epoch (int): The current epoch.
            evaluation_tags: Tags for evaluation datasets.
            cross_validation (bool): Whether to perform cross-validation.
            evaluate_seen_unseen (bool): Whether to evaluate on both seen and unseen classes.
            alpha_keys: Keys for alpha values used in evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing evaluation results for each dataset.
        """
        if evaluation_tags is None:
            evaluation_tags = [ds.name for ds in test_datasets]
        test_datasets = [test_datasets] if not isinstance(test_datasets, list) else test_datasets
        evaluation_tags = [evaluation_tags] if not isinstance(evaluation_tags, list) else evaluation_tags

        if cross_validation and not self.args.ema_exemplar_per_class_acc:
            self.model.nearest_neighbor_cross_validation()

        results = {}
        print("Evaluating...")
        for test_dataset, evaluation_tag in zip(test_datasets, evaluation_tags):
            results[evaluation_tag] = self.evaluate_single_dataset_seen_unseen_different_alphas(
                test_dataset, stage=stage, epoch=epoch, tag=evaluation_tag, alpha_keys=alpha_keys, evaluate_seen_unseen=evaluate_seen_unseen, **kwargs
            )
            print(f"Epoch {epoch}, Dataset: {test_dataset.name}, Performance: {results[evaluation_tag]}")

        return results

    def evaluate_single_dataset_seen_unseen_different_alphas(self, test_dataset, stage, tag, epoch=0, alpha_keys=None, evaluate_seen_unseen=True, **kwargs):
        """
        Evaluate the model on a single dataset, considering seen and unseen classes.

        Args:
            test_dataset: The dataset to be evaluated.
            stage (int): The current evaluation stage.
            tag (str): The tag for the evaluation.
            epoch (int): The current epoch.
            alpha_keys: Keys for alpha values used in evaluation.
            evaluate_seen_unseen (bool): Whether to evaluate on both seen and unseen classes.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing evaluation results for the dataset.
        """
        self._before_evaluate_single_dataset(test_dataset, stage, tag, epoch, **kwargs)

        result_statistics = self._initialize_result_statistics(alpha_keys)
        learned_classes_indices = self._get_learned_classes_indices(test_dataset)

        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                self._process_batch_inference(batch, result_statistics, learned_classes_indices, alpha_keys, evaluate_seen_unseen)

        return self._calculate_accuracy(result_statistics)

    def _initialize_result_statistics(self, alpha_keys):
        """
        Initialize result statistics dictionary for different alpha values.

        Args:
            alpha_keys: Keys for alpha values used in evaluation.

        Returns:
            dict: A dictionary containing initialized result statistics.
        """
        result_statistics = {"tuned": self._create_stat_dict(), "original": self._create_stat_dict()}
        if alpha_keys:
            result_statistics.update({key: self._create_stat_dict() for key in alpha_keys})
        return result_statistics

    def _create_stat_dict(self):
        """
        Create a dictionary to store evaluation statistics.

        Returns:
            dict: A dictionary containing initialized evaluation statistics.
        """
        return {
            "correct": 0, "total": 0,
            "learned_correct": 0, "learned_total": 0,
            "unseen_correct": 0, "unseen_total": 0
        }

    def _get_learned_classes_indices(self, test_dataset):
        """
        Get indices of learned classes from the test dataset.

        Args:
            test_dataset: The test dataset.

        Returns:
            torch.Tensor: Indices of learned classes.
        """
        if hasattr(test_dataset, "learned_classes_indices"):
            return test_dataset.learned_classes_indices
        return torch.unique(torch.from_numpy(test_dataset.concatenated_labels), sorted=True)

    def _update_statistics(self, stats, predicted, evaluate_seen_unseen, learned_classes_indices):
        """
        Update evaluation statistics for overall, seen, and unseen classes.

        Args:
            stats (dict): Dictionary to store the updated statistics.
            predicted (torch.Tensor): Tensor of predicted class labels.
            evaluate_seen_unseen (bool): Flag to determine if seen/unseen evaluation should be performed.
            learned_classes_indices (list or torch.Tensor): Indices of learned (seen) classes.

        This method updates the following statistics:
        - Total samples processed
        - Correct predictions (overall)
        - Seen (learned) class statistics (if evaluate_seen_unseen is True)
        - Unseen class statistics (if evaluate_seen_unseen is True)
        """
        stats["total"] += self.y.size(0)
        stats["correct"] += (predicted == self.y).sum().item()
        
        if evaluate_seen_unseen:
            for pred, label in zip(predicted, self.y):
                if label.item() in learned_classes_indices:
                    stats["learned_total"] += 1
                    if pred == label:
                        stats["learned_correct"] += 1
                else:
                    stats["unseen_total"] += 1
                    if pred == label:
                        stats["unseen_correct"] += 1

    def _calculate_accuracy(self, result_statistics):
        """
        Calculate overall, seen, and unseen accuracies for each key in result_statistics.

        Args:
            result_statistics (dict): Dictionary containing evaluation statistics.

        Returns:
            dict: A dictionary containing accuracy metrics for each key in result_statistics.
        """
        return {
            key: {
                "overall": 100.0 * value["correct"] / value["total"] if value["total"] > 0 else 0,
                "seen": 100.0 * value["learned_correct"] / value["learned_total"] if value["learned_total"] > 0 else 0,
                "unseen": 100.0 * value["unseen_correct"] / value["unseen_total"] if value["unseen_total"] > 0 else 0,
            }
            for key, value in result_statistics.items()
        }
