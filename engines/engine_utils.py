import torch


class DynamicEMARecorder:
    """
    A class for dynamically recording and updating Exponential Moving Average (EMA) of accuracies for each class.

    This recorder uses a dynamic EMA approach, initializing with a simple average for the first few samples,
    then transitioning to an EMA for subsequent updates.

    Attributes:
        ema_decay (float): The decay rate for the EMA calculation.
        accuracies (dict): A dictionary storing the EMA of accuracies for each class.
        num_samples_per_class (dict): A dictionary storing the number of samples seen for each class.
        init_num_samples (int): The number of samples to use for initial simple averaging.
    """

    def __init__(self, ema_decay=0.9):
        """
        Initialize the DynamicEMARecorder.

        Args:
            ema_decay (float, optional): The decay rate for the EMA calculation. Defaults to 0.9.
        """
        self.ema_decay = ema_decay
        self.accuracies = {}  # EMA of accuracies for each class
        self.num_samples_per_class = {}  # Number of samples seen for each class
        self.init_num_samples = int(1. / (1. - ema_decay))

    def update(self, predictions, labels, original_labels, new):
        """
        Update the EMA of accuracies for each class based on new predictions and labels.

        Args:
            predictions (torch.Tensor): The model's predictions.
            labels (torch.Tensor): The true labels.
            original_labels (torch.Tensor): The original labels before any transformations.
            new (torch.Tensor): A boolean tensor indicating which samples are new.
        """
        # Convert tensors to CPU
        predictions = predictions.cpu()
        labels = labels.cpu()
        original_labels = original_labels.cpu()
        new = new.cpu()

        # Update EMA of accuracies for each class
        total_count = 0
        for cls in torch.unique(labels):
            cls_mask = (labels == cls) & new
            total_count += cls_mask.sum()
            if cls_mask.sum() == 0:
                continue
            cls_accuracy = (predictions[cls_mask] == labels[cls_mask]).float().mean()

            original_cls = original_labels[cls_mask][0]

            current_num_samples = self.num_samples_per_class.get(original_cls.item(), 0)
            if current_num_samples <= self.init_num_samples:
                if original_cls.item() not in self.accuracies:
                    self.accuracies[original_cls.item()] = cls_accuracy
                else:
                    self.accuracies[original_cls.item()] = (cls_accuracy + self.accuracies.get(original_cls.item(), 0) * current_num_samples) / (current_num_samples + 1)
                self.num_samples_per_class[original_cls.item()] = current_num_samples + 1
            else:
                # Initialize if the class is seen for the first time
                if original_cls.item() not in self.accuracies:
                    self.accuracies[original_cls.item()] = (1 - self.ema_decay) * cls_accuracy
                else:
                    # Update EMA for the class
                    self.accuracies[original_cls.item()] = (
                            self.ema_decay * self.accuracies[original_cls.item()] +
                            (1 - self.ema_decay) * cls_accuracy
                    )
        # print(f"Total count: {total_count}")

    def get_accuracies(self):
        """
        Get the current EMA accuracies for all classes.

        Returns:
            dict: A dictionary mapping class indices to their current EMA accuracies.
        """
        # sort the accuracies by class index
        self.accuracies = {cls: acc for cls, acc in sorted(self.accuracies.items(), key=lambda item: item[0])}
        return {cls: acc.item() for cls, acc in self.accuracies.items()}
