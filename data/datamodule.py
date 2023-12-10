from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.builder import USER_FILE
from data.utils import (
    ITEM_TOKENIZER_FILE,
    VALUE_TOKENIZER_FILE,
    MODEL_DIR,
    PARAM_FILE,
    TOKEN_CLS,
    TOKEN_MASK,
    TOKEN_PAD,
    TRAIN_DIR,
    get_version,
)
from remy.tasks.train.user_modeling.data.dataset import PretrainDataset
from remy.utils import dump_data, load_data

logger = getLogger(__name__)


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.version: Optional[str] = None
        self.data_dir: Optional[Path] = None
        self.num_users: Optional[int] = None
        self.period: Optional[str] = None
        self.seq_len_range: Optional[str] = None

        self.action_type_indexer: Optional[dict[str, int]] = None
        self.content_indexer: Optional[dict[str, int]] = None

        self.train_user_pool: set[int] = set()
        self.valid_user_pool: set[int] = set()
        self.train_dataset: Optional[PretrainDataset] = None
        self.valid_dataset: Optional[PretrainDataset] = None

    def prepare_data(self):
        _, data_dir = get_version(version=self.config.version, user_bert_dir=self.config.user_bert_dir, verbose=False)
        if not (data_dir / MODEL_DIR / PARAM_FILE).exists():
            (data_dir / MODEL_DIR).mkdir(exist_ok=True)
            action_type_indexer = load_data(data_dir / ACTION_TYPE_INDEXER_FILE)
            action_type_vocab_size = len(action_type_indexer)
            content_vocab_size = len(load_data(data_dir / CONTENT_INDEXER_FILE))
            pad_index = action_type_indexer[TOKEN_PAD]
            mask_index = action_type_indexer[TOKEN_MASK]
            model_params = dict(
                embedding_dim=self.config.embedding_dim,
                intermediate_embedding_dim=self.config.intermediate_embedding_dim,
                action_type_vocab_size=action_type_vocab_size,
                content_vocab_size=content_vocab_size,
                num_hidden_layers=self.config.num_hidden_layers,
                num_train_negative_samples=self.config.num_train_negative_samples,
                num_valid_negative_samples=self.config.num_valid_negative_samples,
                pad_index=pad_index,
                mask_index=mask_index,
                dropout=self.config.dropout,
                temperature=self.config.temperature,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            dump_data(data_dir / MODEL_DIR / PARAM_FILE, model_params)

    def setup(self, stage: Optional[str] = None):
        self.version, self.data_dir = get_version(
            version=self.config.version, user_bert_dir=self.config.user_bert_dir, verbose=False
        )

        self.action_type_indexer = load_data(self.data_dir / ACTION_TYPE_INDEXER_FILE)
        self.content_indexer = load_data(self.data_dir / CONTENT_INDEXER_FILE)

        self._split_data()
        self.train_dataset = PretrainDataset(
            self.data_dir / TRAIN_DIR,
            user_pool=self.train_user_pool,
            bsm_seq_len=self.config.bsm_seq_len,
            mbp_seq_len=self.config.mbp_seq_len,
            lbp_seq_len=self.config.lbp_seq_len,
            pad_index=self.action_type_indexer[TOKEN_PAD],
            mask_index=self.action_type_indexer[TOKEN_MASK],
            cls_index=self.action_type_indexer[TOKEN_CLS],
            mask_prob=self.config.mask_prob,
        )
        self.valid_dataset = PretrainDataset(
            self.data_dir / TRAIN_DIR,
            user_pool=self.valid_user_pool,
            bsm_seq_len=self.config.bsm_seq_len,
            mbp_seq_len=self.config.mbp_seq_len,
            lbp_seq_len=self.config.lbp_seq_len,
            pad_index=self.action_type_indexer[TOKEN_PAD],
            mask_index=self.action_type_indexer[TOKEN_MASK],
            cls_index=self.action_type_indexer[TOKEN_CLS],
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

    def _split_data(self):
        user_ids = [int(u_id) for u_id in open(self.data_dir / TRAIN_DIR / USER_FILE, "r")]
        if self.config.num_users == -1:
            self.num_users = len(user_ids)
        elif len(user_ids) < self.config.num_users:
            logger.info(
                f"# existing users ({len(user_ids):,d}) is lower than config.num_users"
                f"({self.config.num_users:,d}). Thus, config.num_users will be set to {len(user_ids):,d}."
            )
            self.num_users = len(user_ids)
        else:
            self.num_users = self.config.num_users
        sample_indices = np.random.choice(len(user_ids), self.num_users, replace=False)
        split_point = int(self.num_users * self.config.valid_size)
        for i, idx in enumerate(sample_indices):
            if i < split_point:
                self.valid_user_pool.add(user_ids[idx])
            else:
                self.train_user_pool.add(user_ids[idx])
        logger.info(f"# train users: {len(self.train_user_pool):,d}")
        logger.info(f"# valid users: {len(self.valid_user_pool):,d}")
