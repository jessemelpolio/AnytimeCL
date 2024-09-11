import torch


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self, weight=1e6):
        super().__init__()
        self.weight = weight
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        return (1 - self.cosine_similarity(x, y)).mean()  # * self.weight


class OpenSetCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, x, y, label_features):
        y_pred = 100 * x @ label_features.T
        return self.ce(y_pred, y)


class OpenSetCrossEntropyLossWithOtherLogit(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ce = torch.nn.CrossEntropyLoss()
        self.other_class_calibration_loss_weight = self.args.other_class_calibration_loss_weight

    def forward(self, x, y, label_features, other_logit):
        if not self.args.use_other_classifier:
            # we don't need to normalize x here since it should be normalized in the forward function of the model
            zero_weights = torch.zeros(1, x.shape[1]).to(x.device)
            zero_weights.requires_grad = False
            other_logit = torch.nn.functional.linear(x, zero_weights, other_logit)
        y_pred = 100 * x @ label_features.T
        if self.args.use_other_classifier:
            y_pred = y_pred / y_pred.norm(dim=1, keepdim=True)
        y_pred = torch.cat([y_pred, other_logit], dim=1)
        # Step 1: Create a mask of the same shape as x where the correct label positions are False.
        B, C = y_pred.shape
        mask = torch.ones(B, C, dtype=bool)
        rows = torch.arange(B)
        mask[rows, y] = False
        # Step 2: Use this mask to index x and retrieve the incorrect logits.
        y_pred_incorrect_flat = y_pred[mask]
        # Step 3: Reshape to the desired shape.
        y_pred_incorrect = y_pred_incorrect_flat.reshape(B, C - 1)
        # concat y_pred_incorrect and other_logit
        y_pred_incorrect = torch.cat([other_logit, y_pred_incorrect], dim=1)
        y_other = torch.zeros_like(y).requires_grad_(False)
        main_loss = self.ce(y_pred, y)
        auxilary_loss = self.other_class_calibration_loss_weight * self.ce(
            y_pred_incorrect, y_other
        )
        return main_loss + auxilary_loss