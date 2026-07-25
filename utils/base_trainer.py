from pathlib import Path
import datetime
import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim

import utils.args
import utils.train
from model.mlp import MLPRegression
from data.regression_dataset import RegressionDataset
from data.transforms import normalize_targets


class BaseTrainer:
    def __init__(self):
        self._pre_setting()
        self._load_data()
        self._get_model()
        self._get_optimizer_and_scheduler()
        self._get_logger()

    def _additional_parser(self, parser):
        return parser

    def _pre_setting(self):
        self.execute_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.args = utils.args.get_args(self._additional_parser)
        print(self.args)

        self.device = 'cuda' if (torch.cuda.is_available() and not self.args.no_cuda) else 'cpu'
        utils.train.fix_seed(self.args.seed)

        self.min_test_metric = float('inf')
        self.max_epoch = 0

    def _load_data(self):
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        from data.transforms import normalize_features

        data = fetch_california_housing()
        X, y = data.data, data.target
        self.input_size = X.shape[1]
        self.feature_names = data.feature_names

        X_norm = normalize_features(X)
        if self.args.normalize_y:
            y_norm, self.y_min, self.y_max = normalize_targets(y)
        else:
            y_norm = y
            self.y_min, self.y_max = y.min(), y.max()

        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y_norm, test_size=0.2, random_state=self.args.seed)

        self.train_dataset = RegressionDataset(X_train, y_train)
        self.test_dataset = RegressionDataset(X_test, y_test)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False)

    def _get_model(self):
        if self.args.model == "mlp":
            model = MLPRegression(
                input_dim=self.input_size,
                hidden_dims=[128, 64],
                dropout=0.1
            ).to(self.device)
        else:
            raise NotImplementedError(f"Unknown model: {self.args.model}")

        self.model = model

    def _get_optimizer_and_scheduler(self):
        self.loss_fn = nn.MSELoss()

        model_params = self.model.parameters()

        if self.args.optim == "SGD":
            optimizer = optim.SGD(
                model_params,
                lr=self.args.lr,
                weight_decay=self.args.regularizer
            )
        elif self.args.optim == "Momentum":
            optimizer = optim.SGD(
                model_params,
                lr=self.args.lr,
                weight_decay=self.args.regularizer,
                momentum=0.9
            )
        elif self.args.optim == "ADAM":
            optimizer = optim.Adam(
                model_params,
                lr=self.args.lr,
                weight_decay=self.args.regularizer
            )
        else:
            raise NotImplementedError

        if self.args.scheduler == "CosineTorch":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                eta_min=0
            )
        elif self.args.scheduler == "Cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.epochs
            )
        elif self.args.scheduler == "Step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 60, gamma=0.2
            )
        elif self.args.scheduler == "MultiStep":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, [60, 120, 160], gamma=0.2
            )
        else:
            raise NotImplementedError

        self.optimizer = optimizer
        self.scheduler = scheduler

    def _get_logger(self):
        self.log_dir = os.path.join(
            "test" if self.args.test_exp else "runs",
            self.execute_time,
            self.args.dataset,
            str(self.args.epochs)
        )
        if self.args.exp_name is not None:
            self.log_dir = os.path.join(self.log_dir, self.args.exp_name)

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(self.log_dir, 'args.txt'), 'w') as f:
            json.dump(vars(self.args), f, indent=4)

        self.fig_dir = os.path.join(self.log_dir, "figures")
        Path(self.fig_dir).mkdir(parents=True, exist_ok=True)

        self.models_dir = os.path.join(self.log_dir, "models")
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

    def _loss(self, logit, label, index=None, epoch=None):
        return self.loss_fn(logit.squeeze(), label.float())

    def _train_single_epoch(self, epoch):
        self.model.train()

        train_total = 0
        train_loss_sum = 0.0

        for j, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(data)

            loss = self._loss(outputs, labels, epoch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_total += labels.size(0)
            train_loss_sum += loss.item() * labels.size(0)

        self.scheduler.step()

        return train_loss_sum / train_total

    def _evaluate_single_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for data, labels in self.test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(data)
                loss = self.loss_fn(logits.squeeze(), labels.float())
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

        return total_loss / total_samples

    def _print_and_write(self, epoch, metric_list, mode='train', t_epoch=None):
        prt_str = f'Epoch [{epoch}/{self.args.epochs}] {mode} MSE: {metric_list:.4f}'

        if t_epoch is not None:
            elapsed_time = time.time() - t_epoch
            prt_str += f', time: {elapsed_time:.2f}s'
        print(prt_str)

        if mode == 'test':
            self.curr_test_metric = metric_list
            self.curr_epoch = epoch
            if metric_list < self.min_test_metric:
                self.min_test_metric = metric_list
                self.max_epoch = epoch

    def _train_summary(self):
        print(f"min_test_MSE:{self.min_test_metric:.4f}({self.max_epoch})/curr:{self.curr_test_metric:.4f}")
        print(self.args)

    def save_model(self, epoch):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.models_dir, "model_" + str(epoch) + ".pt")
        )

    def train(self):
        t_training = time.time()

        for epoch in range(1, self.args.epochs + 1):
            t_epoch = time.time()
            train_metric = self._train_single_epoch(epoch)
            self._print_and_write(epoch, train_metric, t_epoch=t_epoch)

            test_metric = self._evaluate_single_epoch()
            self._print_and_write(epoch, test_metric, mode='test')

            if self.args.model_save:
                self.save_model(epoch)

        print("Total Learning time: {:2f}s".format(time.time() - t_training))
        self._train_summary()


def main():
    trainer = BaseTrainer()
    trainer.train()

if __name__ == '__main__':
    main()
