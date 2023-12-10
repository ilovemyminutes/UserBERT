from argparse import Namespace
from datetime import datetime
import pytorch_lightning as pl

from data.builder import BehaviorDataBuilder


class UserBERTBuilder:
    def __init__(self, config: Namespace):
        self.config = config

    def build(self):
        pl.seed_everything(self.config.seed)
        self.prepare_data()

    def prepare_data(self):
        log_start = datetime.strptime("2019-01-01", "%Y-%m-%d")
        log_end = datetime.strptime("2019-12-31", "%Y-%m-%d")
        BehaviorDataBuilder(
            self.config.raw_data_dir,
            self.config.user_bert_dir,
            log_start,
            log_end,
            num_items=self.config.num_items,
            n_jobs=self.config.num_workers,
        ).build()

    def train(self):
        return
