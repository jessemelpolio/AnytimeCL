import os
import os.path as osp
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn


class DynamicModule(nn.Module):
    """Dynamic Modules are Avalanche modules that can be incrementally
    expanded to allow architectural modifications (multi-head
    classifiers, progressive networks, ...).
    Compared to pytoch Modules, they provide an additional method,
    `model_adaptation`, which adapts the model given the current experience.
    """

    def adaptation(self, classes_in_this_experience):
        """Adapt the module (freeze units, add units...) using the current
        data. Optimizers must be updated after the model adaptation.
        Avalanche strategies call this method to adapt the architecture
        *before* processing each experience. Strategies also update the
        optimizer automatically.
        :param classes_in_this_experience: number of classes in this experience.
        :return:
        """
        if self.training:
            self.train_adaptation(classes_in_this_experience)
        else:
            self.eval_adaptation(classes_in_this_experience)

    def train_adaptation(self, classes_in_this_experience):
        """Module's adaptation at training time."""
        pass

    def eval_adaptation(self, classes_in_this_experience):
        """Module's adaptation at evaluation time."""
        pass


class BaseEngine(object):
    def __init__(self, args, model=None) -> None:
        # Each engine should have the following attributes: model, optimizer, logger, criterion
        self.args = args
        self.device = args.device
        self.model = model
        self.optimizer = None
        self.loss_func = None
        self.logger = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.learnable_parameters = None

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def configure_learnable_params(
        self, param_keys="all", requires_grad=True, **kwargs
    ):
        if param_keys != "all":
            for param in self.model.parameters():
                param.requires_grad = not requires_grad

        params = []

        if isinstance(param_keys, list):
            for name in param_keys:
                for module_name, param in self.model.named_parameters():
                    if name in module_name:
                        param.requires_grad = requires_grad

        elif isinstance(param_keys, str):
            if param_keys == "all":
                for param in self.model.parameters():
                    param.requires_grad = requires_grad
            elif hasattr(self.model, param_keys):
                for param in getattr(self.model, param_keys).parameters():
                    param.requires_grad = requires_grad
        else:
            raise NotImplementedError

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Learnable parameter: {name}")
                params.append(param)

        self.learnable_parameters = params

    def configure_optimizers(self, **kwargs):
        if self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.learnable_parameters, lr=self.args.lr
            )
        elif self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.learnable_parameters,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.learnable_parameters,
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise NotImplementedError

    def model_adaptation(self, dataset, **kwargs):
        for module in self.model.modules():
            if isinstance(module, DynamicModule):
                module.adaptation(dataset)

    def configure_train_dataloader(self, train_dataset, **kwargs):
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        self.model.encode_class_features(train_dataset.classes)

    def configure_test_dataloader(self, test_dataset, **kwargs):
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        if isinstance(test_dataset, Subset):
            test_dataset = test_dataset.dataset

        self.model.encode_class_features(test_dataset.classes)

    def configure_val_dataloader(self, val_dataset, **kwargs):
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

    def fit(
        self, train_dataset, test_datasets=None, evaluation_tags=None, stage=0, **kwargs
    ):
        self._before_train_stage(train_dataset, **kwargs)
        acc = None
        for i in range(self.args.start_epoch, self.args.n_epochs):
            if i % self.args.eval_interval == 0 and test_datasets is not None:
                assert len(test_datasets) == len(
                    evaluation_tags
                ), "The number of test datasets should be equal to the number of evaluation tags"
                acc = self.evaluate(
                    test_datasets,
                    stage=stage,
                    epoch=i,
                    evaluation_tags=evaluation_tags,
                    **kwargs,
                )

            self._before_train_epoch(train_dataset, **kwargs)
            self.train_epoch(stage, i, **kwargs)
            self._after_train_epoch(**kwargs)
            if i % self.args.save_interval == 0:
                self.save_checkpoint(stage, i, acc, **kwargs)

        if test_datasets is not None:
            acc = self.evaluate(
                test_datasets,
                stage=stage,
                epoch=self.args.n_epochs,
                evaluation_tags=evaluation_tags,
                **kwargs,
            )
            self.save_checkpoint(stage, self.args.n_epochs, acc, **kwargs)
        self._after_train_stage(**kwargs)

    def _before_train_stage(self, train_dataset, **kwargs):
        self.model_adaptation(train_dataset, **kwargs)
        self.configure_learnable_params(**kwargs)
        self.configure_optimizers(**kwargs)
        self.configure_train_dataloader(train_dataset, **kwargs)

    def _after_train_stage(self, **kwargs):
        pass

    def _after_train_epoch(self, **kwargs):
        pass

    def _before_train_epoch(self, train_dataset, **kwargs):
        # Configure anything necessary before training an epoch, such as setting the model to train mode, configure dataloaders, etc.
        self.configure_train_dataloader(train_dataset, **kwargs)
        self.model.train()

    def resume(self, ckpt_path, **kwargs):
        if osp.isfile(ckpt_path):
            print(f"=> loading checkpoint '{ckpt_path}'")
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint["net"])
            self.args.start_epoch = checkpoint["epoch"]
            print(
                f"""=> loaded checkpoint '{ckpt_path}' (epoch {checkpoint["epoch"]})"""
            )
        else:
            print(f"=> no checkpoint found at '{ckpt_path}'")

    def save_checkpoint(self, stage, epoch, acc=None, **kwargs):
        if self.args.save:
            state = {
                "net": self.model.state_dict(),
                "acc": acc,
                "epoch": epoch,
                "stage": stage,
            }
            ckpt_dir = os.path.join(self.args.results_dir, "checkpoint")
            if not osp.isdir(ckpt_dir):
                os.mkdir(ckpt_dir)
            torch.save(
                state,
                os.path.join(ckpt_dir, "stage_%02d_ckpt_%04d.pth" % (stage, epoch)),
            )

    def criterion(self, outputs, targets, text_targets=None):
        return self.loss_func(outputs, targets)

    def train_step(self, inputs, targets, text_targets=None, **kwargs):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets, text_targets=text_targets)
        return loss, outputs

    def test_step(self, inputs, **kwargs):
        outputs = self.model(inputs)
        _, predicted = outputs.max(1)
        return predicted

    def transform_data(self, inputs, targets, text_targets=None):
        return inputs.to(self.device), targets.to(self.device), text_targets

    def train_epoch(self, stage, epoch, set_train=True, **kwargs):
        if set_train:
            self.model.train()
        else:
            self.model.eval()

        batch_num = 0
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets, text_targets in self.train_loader:
            inputs, targets, text_targets = self.transform_data(
                inputs, targets, text_targets=text_targets
            )
            with torch.autograd.set_detect_anomaly(True):
                self.optimizer.zero_grad()
                loss, similarity = self.train_step(
                    inputs, targets, text_targets, **kwargs
                )
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            _, predicted = similarity.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_num += 1

        if self.logger is not None:
            self.logger.add_scalar(
                f"train/stage_{stage}/loss", train_loss / batch_num, epoch
            )
            self.logger.add_scalar(
                f"train/stage_{stage}/acc", 100.0 * correct / total, epoch
            )
        print(
            "Stage {} -- Train Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                stage,
                epoch,
                train_loss / batch_num,
                correct,
                total,
                100.0 * correct / total,
            )
        )

    def _before_evaluate_single_dataset(self, test_dataset, **kwargs):
        self.configure_test_dataloader(test_dataset, **kwargs)
        self.model.eval()

    @torch.no_grad()
    def evaluate_single_dataset(
        self, test_dataset, stage=0, epoch=0, tag="train_test", **kwargs
    ):
        self._before_evaluate_single_dataset(test_dataset, **kwargs)
        correct = 0
        total = 0
        y_pred = []
        y_true = []


        for inputs, targets, text_targets in self.test_loader:
            inputs, targets, text_targets = self.transform_data(
                inputs, targets, text_targets=text_targets
            )
            predicted = self.test_step(inputs, **kwargs)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.logger is not None:
            self.logger.add_scalar(
                f"test/stage_{stage}/{tag}_acc", 100.0 * correct / total, epoch
            )

        print(
            "Stage {} -- Test Epoch: {} on {} \tAcc: {:.3f} ({}/{})".format(
                stage, epoch, tag, 100.0 * correct / total, correct, total
            )
        )

        return 100.0 * correct / total

    def evaluate(
        self, test_datasets, stage=0, epoch=0, evaluation_tags=["train_test"], **kwargs
    ):
        if not isinstance(test_datasets, list):
            test_datasets = [test_datasets]
        if not isinstance(evaluation_tags, list):
            evaluation_tags = [evaluation_tags]
        return {
            evaluation_tag: self.evaluate_single_dataset(
                test_dataset, stage=stage, epoch=epoch, tag=evaluation_tag, **kwargs
            )
            for test_dataset, evaluation_tag in zip(test_datasets, evaluation_tags)
        }
