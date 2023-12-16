import os
from argparse import Namespace
from datetime import datetime

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.builder import BehaviorDataBuilder
from data.utils import load_json
from model.user_bert import UserBERT
from pretrain.datamodule import PretrainDataModule


class UserBERTPretrainingModule:
    CHECKPOINT_NAME = "best_user_bert"

    def __init__(self, config: Namespace):
        self.config = config

        self.datamodule: PretrainDataModule | None = None
        self.model: UserBERT | None = None

    def build(self):
        pl.seed_everything(self.config.seed)
        self.prepare()
        self.train()
        self.save()

    def prepare(self):
        self._build_data()
        self.datamodule = PretrainDataModule(self.config)
        self.datamodule.prepare_data()
        self.model = UserBERT(self.datamodule.user_bert_config)

    def train(self):
        trainer = self._build_trainer()
        trainer.fit(model=self.model, datamodule=self.datamodule)

    def save(self):
        pass

    def _build_data(self):
        log_start = datetime.strptime(self.config.log_start, "%Y-%m-%d")
        log_end = datetime.strptime(self.config.log_end, "%Y-%m-%d")
        BehaviorDataBuilder(
            self.config.raw_data_dir,
            self.config.user_bert_dir,
            log_start,
            log_end,
            num_items=self.config.num_items,
            n_jobs=self.config.num_workers,
        ).build()

    def _build_trainer(self) -> pl.Trainer:
        if not self.config.offline:
            key = os.environ.get(
                "WANDB_API_KEY",
                load_json("/Users/ilovemyminutes/Documents/workspace/credentials/credentials.json")["WANDB_API_KEY"],
            )
            wandb.login(key=key)
        exp_logger = WandbLogger(
            name=self.config.name,
            project=self.config.project,
            offline=self.config.offline,
        )
        exp_logger.log_hyperparams(vars(self.config))

        callbacks = [
            ModelCheckpoint(
                dirpath=self.config.ckpt_dir,
                filename=self.CHECKPOINT_NAME,
                auto_insert_metric_name=False,
                save_weights_only=True,
                monitor="valid_loss",
                save_last=False,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

        return pl.Trainer(
            max_epochs=self.config.epochs,
            precision=self.config.precision,
            accelerator="gpu",
            strategy="ddp" if self.config.devices > 1 else "auto",
            devices=self.config.devices,
            callbacks=callbacks,
            logger=exp_logger,
        )
