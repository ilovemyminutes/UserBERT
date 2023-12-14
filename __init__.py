from argparse import Namespace
from datetime import datetime
import os

import pytorch_lightning as pl

from data.builder import BehaviorDataBuilder
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data.datamodule import PretrainDataModule
from pytorch_lightning.loggers import WandbLogger
from model.user_bert import UserBERT
import wandb
from data.utils import load_json


class UserBERTBuilder:
    CHECKPOINT_NAME = "best_user_bert"

    def __init__(self, config: Namespace):
        self.config = config

        self.data_builder: BehaviorDataBuilder | None = None
        self.train_datamodule: PretrainDataModule | None = None
        self.model: UserBERT | None = None

        self.wandb_key: str | None = None

    def build(self):
        self.set_env()
        self.prepare()
        self.train()

    def set_env(self):
        pl.seed_everything(self.config.seed)
        self.wandb_key = os.environ.get(
            "WANDB_API_KEY",
            load_json("/Users/ilovemyminutes/Documents/workspace/credentials/credentials.json")["WANDB_API_KEY"],
        )

    def prepare(self):
        log_start = datetime.strptime(self.config.log_start, "%Y-%m-%d")
        log_end = datetime.strptime(self.config.log_end, "%Y-%m-%d")
        self.data_builder = BehaviorDataBuilder(
            self.config.raw_data_dir,
            self.config.user_bert_dir,
            log_start,
            log_end,
            num_items=self.config.num_items,
            n_jobs=self.config.num_workers,
        )
        self.data_builder.build()

    def train(self):
        self.train_datamodule = PretrainDataModule(self.config)
        self.train_datamodule.prepare_data()
        self.model = UserBERT(self.train_datamodule.model_config)

    def _build_trainer(self) -> pl.Trainer:
        if not self.config.offline:
            wandb.login(key=self.wandb_key)
        wandb_logger = WandbLogger(
            name=self.config.name,
            project=self.config.project,
            log_model=not self.config.offline,
            offline=self.config.offline,
        )
        wandb_logger.log_hyperparams(vars(self.config))

        ckpt_callback = self._build_checkpoint()
        lr_callback = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            precision=self.config.precision,
            accelerator="gpu",
            strategy="ddp" if self.config.devices > 1 else "auto",
            devices=self.config.devices,
            callbacks=[ckpt_callback, lr_callback],
            logger=wandb_logger,
        )
        return trainer

    def _build_checkpoint(self) -> ModelCheckpoint:
        self.config.ckpt_dir.mkdir(exist_ok=True, parents=True)
        ckpt_callback = ModelCheckpoint(
            dirpath=self.config.ckpt_dir,
            filename=self.CHECKPOINT_NAME,
            auto_insert_metric_name=False,
            save_weights_only=True,
            monitor="valid_loss",
            save_last=False,
            verbose=True,
        )
        return ckpt_callback
