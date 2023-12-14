from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data.utils import dump_json, load_json


@dataclass
class UserBERTConfig:
    embedding_dim: int = 384
    item_vocab_size: int = 20003
    value_vocab_size: int = (13,)
    num_hidden_layers: int = 12
    num_train_negative_samples: int = 4
    num_valid_negative_samples: int = 4
    pad_index: int = 0
    mask_index: int = 1
    dropout: float = 0.1
    temperature: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 1e-2

    @classmethod
    def from_dict(cls, params: dict) -> UserBERTConfig:
        return UserBERTConfig(**params)

    @classmethod
    def from_json(cls, fpath: Path | str) -> UserBERTConfig:
        return UserBERTConfig(load_json(fpath))

    def to_json(self, fpath: Path | str):
        dump_json(fpath, self.__dict__)
