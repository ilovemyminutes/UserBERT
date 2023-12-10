from argparse import Namespace
from datetime import datetime
import pytorch_lightning as pl

from data.builder import BehaviorDataBuilder
from data.datamodule import PretrainDataModule


class UserBERTBuilder:
    def __init__(self, config: Namespace):
        self.config = config

        self.data_builder: BehaviorDataBuilder | None = None
        self.train_datamodule: PretrainDataModule | None = None

    def build(self):
        pl.seed_everything(self.config.seed)
        self.prepare()
        self.train()

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
