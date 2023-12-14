from argparse import Namespace
from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.builder import USER_FILE
from data.dataset import PretrainDataset
from data.utils import (
    ITEM_TOKENIZER_FILE,
    VALUE_TOKENIZER_FILE,
    MODEL_DIR,
    MODEL_CONFIG_FILE,
    TOKEN_CLS,
    TOKEN_MASK,
    TOKEN_PAD,
    TRAIN_DIR,
    get_version_info,
)
from data.utils import load_json
from model.config import UserBERTConfig

logger = getLogger(__name__)


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config

        self.data_dir: Path | None = None
        self.num_users: int | None = None

        self.item_tokenizer: dict[str, int] | None = None
        self.value_tokenizer: dict[str, int] | None = None
        self.model_config: UserBERTConfig | None = None

        self.train_user_pool: set[int] | None = None
        self.valid_user_pool: set[int] | None = None

        self.train_dataset: PretrainDataset | None = None
        self.valid_dataset: PretrainDataset | None = None

    def prepare_data(self):
        _, self.data_dir = get_version_info(version="latest", user_bert_dir=self.config.user_bert_dir)
        if not (self.data_dir / MODEL_DIR / PARAM_FILE).exists():
            (self.data_dir / MODEL_DIR).mkdir(exist_ok=True)
            self._load_tokenizers()
            self.model_config = UserBERTConfig(
                embedding_dim=self.config.embedding_dim,
                item_vocab_size=len(self.item_tokenizer),
                value_vocab_size=len(self.value_tokenizer),
                num_hidden_layers=self.config.num_hidden_layers,
                num_train_negative_samples=self.config.num_train_negative_samples,
                num_valid_negative_samples=self.config.num_valid_negative_samples,
                pad_index=self.item_tokenizer[TOKEN_PAD],
                mask_index=self.item_tokenizer[TOKEN_MASK],
                dropout=self.config.dropout,
                temperature=self.config.temperature,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            self.model_config.to_json(self.data_dir / MODEL_DIR / MODEL_CONFIG_FILE)
        else:
            self._load_tokenizers()
            self.model_config = UserBERTConfig.from_json(self.data_dir / MODEL_DIR / MODEL_CONFIG_FILE)

    def setup(self, stage: Optional[str] = None):
        self._split_data()
        self.train_dataset = PretrainDataset(
            self.data_dir / TRAIN_DIR,
            user_pool=self.train_user_pool,
            bsm_seq_len=self.config.bsm_seq_len,
            mbp_seq_len=self.config.mbp_seq_len,
            pad_index=self.item_tokenizer[TOKEN_PAD],
            mask_index=self.item_tokenizer[TOKEN_MASK],
            cls_index=self.item_tokenizer[TOKEN_CLS],
            mask_prob=self.config.mask_prob,
        )
        self.valid_dataset = PretrainDataset(
            self.data_dir / TRAIN_DIR,
            user_pool=self.valid_user_pool,
            bsm_seq_len=self.config.bsm_seq_len,
            mbp_seq_len=self.config.mbp_seq_len,
            pad_index=self.item_tokenizer[TOKEN_PAD],
            mask_index=self.item_tokenizer[TOKEN_MASK],
            cls_index=self.item_tokenizer[TOKEN_CLS],
            mask_prob=self.config.mask_prob,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, drop_last=True, num_workers=self.config.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset, batch_size=self.config.batch_size, drop_last=True, num_workers=self.config.num_workers
        )

    def _load_tokenizers(self):
        self.item_tokenizer = load_json(self.data_dir / ITEM_TOKENIZER_FILE)
        self.value_tokenizer = load_json(self.data_dir / VALUE_TOKENIZER_FILE)

    def _split_data(self):
        user_ids = [int(u_id) for u_id in open(self.data_dir / TRAIN_DIR / USER_FILE, "r")]
        if self.config.num_users == -1:
            self.num_users = len(user_ids)
        elif len(user_ids) < self.config.num_users:
            self.num_users = len(user_ids)
        else:
            self.num_users = self.config.num_users
        np.random.shuffle(user_ids)
        split_point = int(self.num_users * self.config.valid_size)
        self.valid_user_pool = set(user_ids[:split_point])
        self.train_user_pool = set(user_ids[split_point:])
