import torch
from .base_network import BaseNetwork
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .tree import SimpleTree
from typing import List
import torch.nn as nn

import sys
import os

# Add the path to the dinov2 directory
dinov2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dinov2'))
sys.path.append(dinov2_path)

from dinov2.models import build_model_from_cfg
from dinov2.configs import load_and_merge_config

class RawDataset(torch.utils.data.Dataset):
    def __init__(self, raw_images, labels=None):
        self.raw_images = raw_images
        self.labels = labels

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, index):
        raw_image = self.raw_images[index]
        if isinstance(raw_image, str):
            raw_image = np.load(raw_image)
        elif isinstance(raw_image, list):
            raw_image = np.load(raw_image[-1])
        else:
            raise NotImplementedError

        raw_image = torch.from_numpy(raw_image)

        if self.labels is None:
            return raw_image, []
        else:
            return raw_image, self.labels[index]

class MemoryModule(BaseNetwork):
    def __init__(self, args, clip_branch=None) -> None:
        super().__init__(args)
        self.args = args
        self.device = args.device
        self.clip_branch = clip_branch or self._load_clip_model(args.backbone)
        self.runtime_get_dim()

        self.tree = SimpleTree(args.node_capacity)

        # Initialize attributes to None
        self.sample_buffer = None
        self.exemplar_idx_to_class = {}
        self.exemplar_idx_to_class_with_prompt = {}
        self.exemplar_idx_to_label = {}
        self.exemplar_labels = None
        self.exemplar_classes_with_prompt = None
        self.exemplar_classes = None
        self.exemplar_classes_features = None

        self.target_labels = None
        self.target_classes = None
        self.target_classes_with_prompt = None
        self.target_classes_features = None

        self.nodes = None
        self.tuned_model_probs = None
        self.original_model_probs = None
        self.original_model_lc_probs = None
        self.p_ft_probs = None
        self.p_zs_nn_probs = None
        self.p_zs_lc_probs = None

        # DINO-related initialization
        if args.use_dino:
            self.dino_encoder = self._load_dino_model(args)
        else:
            self.dino_encoder = None
        
    def _load_clip_model(self, backbone):
        return clip.load(backbone, jit=False, device=self.device)[0].eval().requires_grad_(False)

    def _load_dino_model(self, args):
        cfg = load_and_merge_config(args.dinov2_config_file)
        dino_encoder, encoder_embed_dim = build_model_from_cfg(cfg, only_teacher=True)
        weights = torch.hub.load('facebookresearch/dinov2', args.dinov2_backbone)
        dino_encoder.load_state_dict(weights.state_dict())
        return dino_encoder

    @torch.no_grad()
    def runtime_get_dim(self):
        tensor = torch.randn(1, 3, 224, 224).to(self.device)
        out = self.clip_branch.encode_image(tensor)
        self.enc_dim = out.shape[-1]

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--k", type=int, default=9)
        parser.add_argument("--node_capacity", type=int, default=500000)
        return parser

    def extend_memory(self, dataset, clip_branch=None, dino_encoder=None):
        """
        Extends the memory with new data from the given dataset.
        
        Args:
            dataset: The dataset to extend the memory with.
            clip_branch: Optional CLIP branch to use for encoding. If None, uses self.clip_branch.
            dino_encoder: Optional DINO encoder to use for encoding. If None, uses self.dino_encoder.
        """
        clip_branch = clip_branch or self.clip_branch
        dino_encoder = dino_encoder or self.dino_encoder
        dl = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        self.extract_exemplar_info(dataset)

        buffers = {
            "raw_images": [], "raw_image_features": [], 
            "image_features": [], "label_features": [], "labels": []
        }

      
        for batch in tqdm(dl):
           
            if self.args.use_dino:
                batch_out = self.dino_extend_memory_forward_core(dino_encoder, clip_branch, batch, mode=self.args.X_format)
            else:
                batch_out = self.extend_memory_forward_core(clip_branch, batch, mode=self.args.X_format)
            for key in buffers:
                buffers[key].append(batch_out[key])

        for key in buffers:
            if isinstance(buffers[key][0], list):
                buffers[key] = [item for sublist in buffers[key] for item in sublist]
            else:
                buffers[key] = torch.cat(buffers[key], dim=0)

        self.tree.fit_data_into_tree(buffers)

        if self.sample_buffer is None:
            self.sample_buffer = buffers
        else:
            for key in buffers:
                if isinstance(self.sample_buffer[key], list):
                    self.sample_buffer[key] += buffers[key]
                else:
                    self.sample_buffer[key] = torch.cat([self.sample_buffer[key], buffers[key]], dim=0)

        self.update_label_features()

    def update_label_features(self):
        """
        Updates the label features and counts the number of samples per class in the exemplar set.
        """
        self.label_to_label_features = {}
        for label in self.exemplar_label_to_idx.keys():
            identical_label_features = self.sample_buffer["label_features"][self.sample_buffer["labels"] == label]
            if identical_label_features.shape[0] > 0:
                self.label_to_label_features[label] = identical_label_features[0]
        self.exemplar_classes_features = torch.stack(list(self.label_to_label_features.values()), dim=0).float().to(self.device)
        self.exemplar_classes = list(self.exemplar_idx_to_class.values())

    @torch.no_grad()
    def encode_class_features(self, label_texts: List[str]):
        """
        Encodes class features from given label texts using CLIP.

        Args:
            label_texts (List[str]): A list of label texts to encode.

        Returns:
            torch.Tensor: Encoded features for the given label texts, or None if the input list is empty.

        This method tokenizes the input texts, encodes them using CLIP, and normalizes the resulting features.
        If the number of label texts exceeds the batch size, it processes them in batches.
        """
        if not label_texts:
            return None

        def encode_batch(texts):
            with torch.no_grad():
                labels = clip.tokenize(texts).to(self.device)
                features = self.clip_branch.encode_text(labels)
            return features / features.norm(dim=-1, keepdim=True)

        if len(label_texts) > self.args.batch_size:
            all_features = []
            for i in range(0, len(label_texts), self.args.batch_size):
                batch = label_texts[i:i + self.args.batch_size]
                all_features.append(encode_batch(batch))
            return torch.cat(all_features, dim=0)
        else:
            return encode_batch(label_texts)
    
    def update_exemplar_indices(self, classes, classes_with_prompt, labels, start_index=0):
        """
        Updates the exemplar indices with class information.
        
        Args:
            classes: List of class names.
            classes_with_prompt: List of classes with their prompts.
            labels: List of class labels.
            start_index: The starting index for updating (default is 0).
        """
        for idx, (class_name, class_with_prompt, label) in enumerate(zip(classes, classes_with_prompt, labels), start=start_index):
            self.exemplar_idx_to_class[idx] = class_name
            self.exemplar_idx_to_class_with_prompt[idx] = class_with_prompt
            self.exemplar_idx_to_label[idx] = label if isinstance(label, int) else label.item()

    def extract_exemplar_info(self, dataset):
        """
        Extracts and updates exemplar information from the given dataset.
        
        Args:
            dataset: The dataset to extract exemplar information from.
        """
        new_classes = []
        new_classes_with_prompt = []
        new_labels = []

        if self.exemplar_classes is None:
            new_classes = dataset.current_classes
            new_classes_with_prompt = dataset.current_classes_with_prompt
            new_labels = dataset.current_classes_indices
        else:
            for c, p, l in zip(dataset.current_classes, dataset.current_classes_with_prompt, dataset.current_classes_indices):
                if c not in self.exemplar_classes:
                    new_classes.append(c)
                    new_classes_with_prompt.append(p)
                    new_labels.append(l.item())

        if new_classes:
            new_features = self.encode_class_features(new_classes_with_prompt)
            
            if self.exemplar_classes_features is None:
                self.exemplar_classes_features = new_features
            else:
                self.exemplar_classes_features = torch.cat([self.exemplar_classes_features, new_features], dim=0)
            
            start_index = len(self.exemplar_idx_to_class)
            self.update_exemplar_indices(new_classes, new_classes_with_prompt, new_labels, start_index)
            
            self.exemplar_classes = (self.exemplar_classes or []) + new_classes
            self.exemplar_classes_with_prompt = (self.exemplar_classes_with_prompt or []) + new_classes_with_prompt

        self.exemplar_label_to_idx = {v: k for k, v in self.exemplar_idx_to_label.items()}
        
    def extend_memory_forward_core(self, clip_branch, batch_data, mode="feature"):
        """
        Core function for extending memory with new data.
        
        Args:
            clip_branch: The CLIP branch to use for encoding.
            batch_data: The batch of data to process.
            mode: The mode of operation ("feature", "code", "image", or "embedding").
        
        Returns:
            Dictionary containing processed batch data.
        """
        # TODO (nick): add compression!!! Path given to raw_images
        X, y, y_text, X_path = batch_data

       
        if mode not in ["feature", "code", "image", "embedding"]:
            raise NotImplementedError

        with torch.no_grad():
            if mode in ["feature", "code"]:
                if mode == "feature":
                    X = X.to(self.device).permute(1, 0, 2)  # NLD -> LND
                else:  # mode == "code"
                    X = self.quantizer.project_out(
                        self.quantizer.get_codes_from_indices(X.to(self.device).unsqueeze(-1))
                    ).permute(1, 0, 2)  # NLD -> LND
                
                raw_encoded_images_features = clip_branch.image_encoder_trainable_part_forward(X)
            elif mode == "image":
                raw_encoded_images_features = clip_branch.encode_image(X.to(self.device))
            else:  # mode == "embedding"
                raw_encoded_images_features = X.to(self.device)

            normalized_encoded_images = raw_encoded_images_features / raw_encoded_images_features.norm(dim=-1, keepdim=True)

            tokens = clip.tokenize(y_text).to(self.device)
            encoded_labels = clip_branch.encode_text(tokens)
            normalized_encoded_labels = encoded_labels / encoded_labels.norm(dim=-1, keepdim=True)

        if mode == "feature":
            x_path_list = [[X_path[j][i] for j in range(len(X_path))] for i in range(len(X_path[0]))]
            raw_images = x_path_list
        else:
            raw_images = X.cpu()

        return {
            "raw_images": raw_images,
            "raw_image_features": raw_encoded_images_features.cpu(),
            "image_features": normalized_encoded_images.cpu(),
            "labels": y,
            "label_features": normalized_encoded_labels.cpu(),
        }

    def dino_extend_memory_forward_core(self, dino_encoder, clip_branch, batch_data, mode="feature"):
        """
        Core function for extending memory with new data using DINO encoder.
        
        Args:
            dino_encoder: The DINO encoder to use.
            clip_branch: The CLIP branch to use for encoding.
            batch_data: The batch of data to process.
            mode: The mode of operation (only "feature" is supported).
        
        Returns:
            Dictionary containing processed batch data.
        """
        if mode != "feature":
            raise NotImplementedError

        X, y, y_text, X_path = batch_data
        X_dino = X[-1].to(self.device).squeeze(1)

        with torch.no_grad():
            raw_encoded_images_features = dino_encoder(X_dino)
            normalized_encoded_images = raw_encoded_images_features / raw_encoded_images_features.norm(dim=-1, keepdim=True)

            tokens = clip.tokenize(y_text).to(self.device)
            encoded_labels = clip_branch.encode_text(tokens)
            normalized_encoded_labels = encoded_labels / encoded_labels.norm(dim=-1, keepdim=True)

        x_path_list = [[X_path[j][i] for j in range(len(X_path))] for i in range(len(X_path[0]))]

        return {
            "raw_images": x_path_list,
            "raw_image_features": raw_encoded_images_features.cpu(),
            "image_features": normalized_encoded_images.cpu(),
            "labels": y,
            "label_features": normalized_encoded_labels.cpu(),
        }

    def set_target_classes_features(self, target_classes, target_features):
        """
        Sets the target features for target classes.
        
        Args:
            target_classes: List of target class names.
            target_features: Tensor of target class features.
        """
        self.target_classes_features = target_features.float()
        self.target_classes = target_classes

    def find_mapping_from_exemplar_to_target(self):
        """
        Creates a mapping from exemplar classes to target classes.
        """
        self.exemplar_to_target = {
            i: self.target_classes.index(exemplar_class) if exemplar_class in self.target_classes else -1
            for i, exemplar_class in enumerate(self.exemplar_classes)
        }

    def find_overlapping_indices(self):
        """
        Finds the indices of overlapping classes between exemplar and target sets.
        """
        overlap_indices = [
            (i, j) for i, item in enumerate(self.exemplar_classes)
            for j, other_item in enumerate(self.target_classes)
            if item == other_item
        ]
        self.overlapping_exemplar_indices, self.overlapping_target_indices = zip(*overlap_indices) if overlap_indices else ([], [])

    def adaptive_instance_marginalization(self, query):
        """
        Performs adaptive instance marginalization for a given query.
        
        Args:
            query: The input query tensor.
        
        Returns:
            Tuple containing probabilities and masks for different cases.
        """
        query_target_label_sim = 100 * (self.target_classes_features @ query.unsqueeze(-1)).squeeze(-1)
        mask_qt = torch.zeros_like(query_target_label_sim).to(self.device)
        mask_qt[:, self.overlapping_target_indices] = 1
        p_case_1 = (mask_qt * query_target_label_sim.softmax(-1)).sum(dim=-1, keepdim=True)
        p_case_2 = 1 - p_case_1
        return p_case_1, p_case_2, query_target_label_sim, mask_qt

    def single_model_forward(self, model, X_format, input_X):
        """
        Performs a forward pass through a single model.
        
        Args:
            model: The model to use for the forward pass.
            X_format: The format of the input data.
            input_X: The input data.
        
        Returns:
            The output of the model.
        """
        if X_format == "code":
            zs_out = model.quantizer.get_codes_from_indices(input_X.unsqueeze(-1))
            zs_out = model.quantizer.project_out(zs_out).permute(1, 0, 2)
        elif X_format == "embedding":
            zs_out = input_X
        elif X_format == "feature":
            zs_out = input_X.permute(1, 0, 2)
        elif X_format == "image":
            zs_out = input_X
        else:
            raise NotImplementedError

        zs_out = model(zs_out)

        if self.args.use_tuned_text_embedding:
            zs_out = model.class_features[zs_out.argmax(dim=-1)]
        return zs_out

    def dino_single_model_forward(self, model, X_format, input_X):
        """
        Performs a forward pass through a DINO model.
        
        Args:
            model: The DINO model to use.
            X_format: The format of the input data (unused in this method).
            input_X: The input data.
        
        Returns:
            The output of the DINO model.
        """
        return model(input_X)

    def compute_tuned_and_original_p_ft(self):
        """
        Computes the tuned and original probabilities.
        """
        self.p_ft_probs = self.compute_p_ft(self.tuned_model_probs)
        self.p_zs_nn_probs = self.compute_p_ft(self.original_model_probs)

    def compute_p_ft(self, model_probs):
        """
        Computes the tuned probabilities for a given model.
        
        Args:
            model_probs: The probabilities from the model.
        
        Returns:
            List of computed probabilities.
        """
        probs = []
        for prob in model_probs:
            labels = list(prob.keys())
            overlapping_labels, overlapping_target_indices = self.compute_overlapping_labels_and_target_indices(labels)
            out_prob = torch.zeros(len(self.target_classes))
            for label, index in zip(overlapping_labels, overlapping_target_indices):
                out_prob[index] = prob[label]
            probs.append(out_prob)
        return probs

    def compute_overlapping_labels_and_target_indices(self, exemplar_model_labels):
        """
        Computes the overlapping labels and their corresponding target indices.
        
        Args:
            exemplar_model_labels: Labels from the exemplar model.
        
        Returns:
            Tuple of overlapping labels and their target indices.
        """
        target_labels_in_exemplar = [
            self.exemplar_idx_to_label[self.exemplar_classes.index(c)] if c in self.exemplar_classes else -1
            for c in self.target_classes
        ]

        overlapping_items = [
            (item, j)
            for i, item in enumerate(exemplar_model_labels)
            for j, other_item in enumerate(target_labels_in_exemplar)
            if item == other_item
        ]
        
        return zip(*overlapping_items) if overlapping_items else ([], [])

    def compute_p_ft_alpha(self, zero_shot_feature, ft_probs, zs_probs, p_other, alpha_keys=None):
        """
        Computes the alpha values for fine-tuned probabilities.
        
        Args:
            zero_shot_feature: The zero-shot feature.
            ft_probs: Fine-tuned probabilities.
            zs_probs: Zero-shot probabilities.
            p_other: Probability of other classes.
            alpha_keys: Keys for different alpha computation methods.
        
        Returns:
            Dictionary of computed alpha values or a single alpha value.
        """
        if alpha_keys is None:
            return ft_probs / (ft_probs + zs_probs + 1e-8) if p_other is None else (1 - p_other) * ft_probs / ((1 - p_other) * ft_probs + zs_probs + 1e-8)
        
        alphas = {}
        for key in alpha_keys:
            if key == "p_ft_and_p_other" and p_other is not None:
                alphas[key] = (1 - p_other) * ft_probs / ((1 - p_other) * ft_probs + zs_probs + 1e-8)
            elif key == "p_ft_0_1":
                tmp = ft_probs / (ft_probs + zs_probs + 1e-8)
                alphas[key] = torch.where(tmp > 0.5, torch.ones_like(tmp), torch.zeros_like(tmp))
            elif key == "p_ft":
                alphas[key] = ft_probs / (ft_probs + zs_probs + 1e-8)
            elif key == "p_other" and p_other is not None:
                alphas[key] = 1 - p_other
            elif key == "aim":
                alphas[key] = self.adaptive_instance_marginalization(zero_shot_feature)[0]
        return alphas

    def compute_p_z(self, zero_shot_feature):
        """
        Computes the zero-shot probabilities.
        
        Args:
            zero_shot_feature: The zero-shot feature.
        
        Returns:
            Computed zero-shot probabilities.
        """
        return (100 * zero_shot_feature @ self.target_classes_features.T).softmax(dim=-1)

    def compute_p_other(self, zs_out, zs_other_logits):
        """
        Computes the probabilities for other classes.
        
        Args:
            zs_out: Zero-shot output.
            zs_other_logits: Logits for other classes.
        
        Returns:
            Tuple of candidate probabilities and other probabilities.
        """
        candidate_logits = 100 * zs_out @ self.target_classes_features.T
        if self.args.use_other_classifier:
            candidate_logits /= candidate_logits.norm(dim=-1, keepdim=True)
        candidate_prob = candidate_logits.softmax(dim=-1)
        if zs_other_logits is None:
            return candidate_prob, None
        all_logits = torch.cat([candidate_logits, zs_other_logits], dim=-1)
        other_prob = all_logits.softmax(dim=-1)[:, -1].unsqueeze(-1)
        return candidate_prob, other_prob

    def p_ft_forward(self, max_node, zero_shot_feature, zs_out, zs_other_logits, alpha_keys=None):
        """
        Performs the forward pass using fine-tuned probabilities.
        
        Args:
            max_node: The maximum node indices.
            zero_shot_feature: The zero-shot feature.
            zs_out: Zero-shot output.
            zs_other_logits: Logits for other classes.
            alpha_keys: Keys for different alpha computation methods.
        
        Returns:
            The final output features or probabilities.
        """
        p_ft = torch.stack([self.p_ft_probs[node.item()] for node in max_node], dim=0).to(self.device)
        p_zs = torch.stack([self.p_zs_nn_probs[node.item()] for node in max_node], dim=0).to(self.device)
        p_z = self.compute_p_z(zero_shot_feature)
        p_model, p_other = self.compute_p_other(zs_out, zs_other_logits)
        p_ft_alpha = self.compute_p_ft_alpha(zero_shot_feature, p_ft, p_zs, p_other, alpha_keys)
        
        if isinstance(p_ft_alpha, dict):
            return {
                key: self.target_classes_features[p_model.argmax(dim=-1)] * alpha + zero_shot_feature * (1 - alpha) if key == "aim"
                else self.target_classes_features[(alpha * p_model + (1 - alpha) * p_z).argmax(dim=-1)]
                for key, alpha in p_ft_alpha.items()
            }
        else:
            final_prob = p_ft_alpha * p_model + (1 - p_ft_alpha) * p_z
            return self.target_classes_features[final_prob.argmax(dim=-1)]

    def forward(self, intermediate_feature, zero_shot_feature, alpha_keys=None):
        """
        Performs the forward pass of the TreeMemoryModule.
        
        Args:
            intermediate_feature: The intermediate feature input.
            zero_shot_feature: The zero-shot feature input.
            alpha_keys: Keys for different alpha computation methods.
        
        Returns:
            The final output features or probabilities.
        """
        zero_shot_feature = zero_shot_feature / zero_shot_feature.norm(dim=-1, keepdim=True)

        if self.nodes is None:
            self.nodes = self.tree.get_all_leaf_nodes()

        exemplar_mean_features = torch.cat([node.centroid for node in self.nodes], dim=0)
        exemplar_mean_features = exemplar_mean_features / exemplar_mean_features.norm(dim=-1, keepdim=True)
        exemplar_mean_features = exemplar_mean_features.to(self.device)

        similarity = zero_shot_feature @ exemplar_mean_features.T
        max_node = torch.argmax(similarity, dim=-1)

        unique_nodes = torch.unique(max_node)
        out_features = {}
        other_logits = {} if self.args.include_the_other_class else None

        for node in unique_nodes:
            node = node.item()
            with torch.no_grad():
                exemplar_model = self.nodes[node].exemplar_model
                zs_out = (self.dino_single_model_forward if self.args.use_dino else self.single_model_forward)(
                    exemplar_model, self.args.X_format, intermediate_feature
                )
                zs_out = zs_out / zs_out.norm(dim=-1, keepdim=True)
                
                if self.args.include_the_other_class:
                    other_logits[node] = exemplar_model.other_classifier(zs_out) if self.args.use_other_classifier else exemplar_model.other_bias

            out_features[node] = zs_out

        zs_out = torch.stack([out_features[max_node[b].item()][b] for b in range(zero_shot_feature.shape[0])], dim=0)
        
        if self.args.include_the_other_class:
            zs_other_logits = torch.stack([
                other_logits[max_node[b].item()][b] if self.args.use_other_classifier else other_logits[max_node[b].item()]
                for b in range(zero_shot_feature.shape[0])
            ], dim=0)
        else:
            zs_other_logits = None

        return self.p_ft_forward(max_node, zero_shot_feature, zs_out, zs_other_logits, alpha_keys)
