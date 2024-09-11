import os.path as osp
from torch.utils.data import Sampler
import numpy as np
import collections

# FIFO (First-In-First-Out) Sampler
class FIFOSampler(Sampler):
    def __init__(self, new, batch_size, num_new_samples_per_batch):
        self.new = np.array(new)
        self.batch_size = batch_size
        self.num_new_samples_per_batch = num_new_samples_per_batch
        self.queue = collections.deque(maxlen=batch_size)  # FIFO queue

    def __iter__(self):
        # Iterate while there are still new samples
        while np.any(self.new):
            new_indices = np.where(self.new)[0]
            num_new_samples = min(len(new_indices), self.num_new_samples_per_batch)
            selected_new_indices_this_batch = np.random.choice(
                new_indices, num_new_samples, replace=False
            )
            # Add new samples to the queue and yield from it
            self.queue.extend(selected_new_indices_this_batch)
            selected_new_indices = list(self.queue)
            self.new[selected_new_indices_this_batch] = False
            yield selected_new_indices

    def __len__(self):
        return len(np.where(self.new)[0]) // self.num_new_samples_per_batch

# Uniform Sampler
class UniformSampler(Sampler):
    def __init__(self, new, batch_size, num_new_samples_per_batch):
        self.new = np.array(new)
        self.batch_size = batch_size
        self.num_new_samples_per_batch = num_new_samples_per_batch

    def __iter__(self):
        while np.any(self.new):
            new_indices = np.where(self.new)[0]
            num_new_samples = min(len(new_indices), self.num_new_samples_per_batch)

            if num_new_samples > 0:
                selected_new_indices = np.random.choice(
                    new_indices, num_new_samples, replace=False
                )
            else:
                selected_new_indices = []

            num_old_samples = self.batch_size - len(selected_new_indices)
            old_indices = np.where(~self.new)[0]

            if num_old_samples > 0 and len(old_indices) > 0:
                if len(old_indices) <= num_old_samples:
                    selected_old_indices = old_indices
                else:
                    selected_old_indices = np.random.choice(
                        old_indices, num_old_samples, replace=False
                    )
            else:
                selected_old_indices = []

            # Mark new samples as old
            self.new[selected_new_indices] = False

            # Yield combined new and old samples
            yield np.concatenate((selected_new_indices, selected_old_indices)).astype(
                int
            ).tolist()

    def __len__(self):
        return len(np.where(self.new)[0]) // self.num_new_samples_per_batch

# Class-Balanced Sampler
class ClassBalancedSampler(Sampler):
    def __init__(self, labels, new, batch_size, num_new_samples_per_batch):
        self.labels = np.array(labels)
        self.new = np.array(new)
        self.batch_size = batch_size
        self.num_new_samples_per_batch = num_new_samples_per_batch
        self.classes = np.unique(labels)  # All classes

    def __iter__(self):
        while np.any(self.new):
            new_indices = np.where(self.new)[0]
            num_new_samples = min(len(new_indices), self.num_new_samples_per_batch)

            if num_new_samples > 0:
                selected_new_indices = np.random.choice(
                    new_indices, num_new_samples, replace=False
                ).astype(int)
            else:
                selected_new_indices = []

            num_old_samples = self.batch_size - len(selected_new_indices)
            old_indices = np.where(~self.new)[0]
            old_classes = np.unique(self.labels[old_indices])  # Classes in old data

            selected_old_indices = []
            if num_old_samples > 0 and len(old_indices) > 0:
                if len(old_classes) > self.batch_size:
                    classes_to_sample = np.random.choice(
                        old_classes, self.batch_size, replace=False
                    )
                    for c in classes_to_sample:
                        class_indices = np.where((self.labels[old_indices] == c))[0]
                        if class_indices.size > 0:
                            selected_indices = np.random.choice(
                                old_indices[class_indices], 1, replace=False
                            ).astype(int)
                            selected_old_indices.extend(selected_indices)
                else:
                    samples_per_class = num_old_samples // len(old_classes)
                    # Randomly select different samples for each class
                    for i, c in enumerate(old_classes):
                        class_indices = np.where((self.labels[old_indices] == c))[0]
                        if class_indices.size > 0:
                            selected_indices = np.random.choice(
                                old_indices[class_indices],
                                min(samples_per_class, len(class_indices)),
                                replace=False,
                            ).astype(int)
                            selected_old_indices.extend(selected_indices)

                    # Fill the remaining samples with random samples from old data
                    remaining_samples = num_old_samples - len(selected_old_indices)
                    remaining_samples = min(remaining_samples, len(old_indices))
                    if remaining_samples > 0:
                        selected_indices = np.random.choice(
                            old_indices, remaining_samples, replace=False
                        ).astype(int)
                        selected_old_indices.extend(selected_indices)

            # Mark new samples as old
            self.new[selected_new_indices] = False

            # Yield combined new and old samples
            yield np.concatenate((selected_new_indices, selected_old_indices)).astype(
                int
            ).tolist()

    def __len__(self):
        return len(np.where(self.new)[0]) // self.num_new_samples_per_batch

# Recency-Weighted Sampler
class RecencyWeightedSampler(Sampler):
    def __init__(
        self,
        new,
        num_new_samples_per_batch,
        batch_size,
        initial_weights,
        initial_counts,
        decay_rate=0.9,
        min_weight=0.01,
        use_weights_only=False,
    ):
        self.new = np.array(new)  # This should be a boolean array
        self.num_new_samples_per_batch = num_new_samples_per_batch
        self.batch_size = batch_size
        self.weights = np.array(initial_weights)
        self.sample_counts = np.array(initial_counts)  # T for each sample
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.use_weights_only = use_weights_only

    def __iter__(self):
        if self.use_weights_only:
            all_indices = np.arange(len(self.weights))
            np.random.shuffle(all_indices)  # Shuffle all indices
            current_index = 0  # Start from the beginning of the shuffled indices

            while current_index < len(all_indices):
                end_index = min(current_index + self.batch_size, len(all_indices))
                selected_indices = all_indices[current_index:end_index]
                current_index = end_index

                for idx in selected_indices:
                    self.new[idx] = False
                    self.sample_counts[idx] += 1
                    self.weights[idx] = max(
                        self.decay_rate ** self.sample_counts[idx],
                        self.min_weight,
                    )
                yield selected_indices.tolist()
        else:
            # Indices of new data
            new_data_indices = np.where(self.new)[0]
            np.random.shuffle(new_data_indices)
            batch = []

            for new_idx in new_data_indices:
                batch.append(new_idx)
                # Check if we need to yield the batch as is, without additional old samples
                num_old_samples = len(np.where(~self.new)[0])

                if num_old_samples > 0 and (len(batch) + num_old_samples) < self.batch_size:
                    # Add all old samples to the batch
                    old_samples_indices = np.where(~self.new)[0]
                    batch.extend(old_samples_indices.tolist())

                    # Yield the batch with the new and all old samples
                    yield batch
                    batch = []  # Reset the batch for the next iteration

                    # Mark the new example as old
                    self.new[new_idx] = False
                    self.sample_counts[new_idx] += 1
                    self.weights[new_idx] = max(
                        self.decay_rate ** self.sample_counts[new_idx], self.min_weight
                    )

                    continue

                # Mark the new example as old
                self.new[new_idx] = False
                self.sample_counts[new_idx] += 1
                self.weights[new_idx] = max(
                    self.decay_rate ** self.sample_counts[new_idx], self.min_weight
                )

                if (
                    len(batch) == self.batch_size
                    or (len(batch) + num_old_samples) < self.batch_size
                ):
                    yield batch
                    batch = []  # Reset the batch for the next iteration
                else:
                    # If the batch is not full and there are enough old samples,
                    # fill the batch with additional old samples
                    batch.extend(self._select_additional_indices(batch))
                    yield batch
                    batch = []  # Reset the batch for the next iteration

            # Handle any remaining new samples for the last batch
            if batch:
                # Even for the last batch, if there are not enough old samples, yield as is
                num_old_samples = len(np.where(~self.new)[0])
                if (len(batch) + num_old_samples) < self.batch_size:
                    yield batch
                else:
                    # Fill the batch with additional old samples
                    batch.extend(self._select_additional_indices(batch))
                    yield batch

    def _select_indices(self, indices):
        # Normalize weights for sampling
        weights_normalized = self.weights[indices] / (
            self.weights[indices].sum() + 1e-10
        )
        selected_indices = np.random.choice(
            indices, self.batch_size, p=weights_normalized, replace=False
        )

        # Update sample counts for selected samples
        self.sample_counts[selected_indices] += 1

        # Apply weight decay for selected samples and enforce minimum weight
        for idx in selected_indices:
            self.new[idx] = False
            self.weights[idx] = self.decay_rate ** (
                self.sample_counts[idx]
            )  # Apply decay based on the new count
            self.weights[idx] = max(
                self.weights[idx], self.min_weight
            )  # Enforce minimum weight

        return selected_indices.tolist()

    def _select_additional_indices(self, batch_new_indices):
        additional_indices = []
        num_additional = self.batch_size - len(batch_new_indices)

        old_data_indices = np.where(~self.new)[0]
        if num_additional > 0:
            selected_indices = self._select_indices(old_data_indices)
            additional_indices.extend(selected_indices[:num_additional])

        return additional_indices

    def __len__(self):
        if self.use_weights_only:
            return (len(self.weights) + self.batch_size - 1) // self.batch_size
        else:
            return len(np.where(self.new)[0]) // self.num_new_samples_per_batch