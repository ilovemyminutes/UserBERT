from dataclasses import dataclass


@dataclass
class UserBERTConfig:
    embedding_dim: int = 384
    intermediate_embedding_dim: int = 384
    item_vocab_size: int = 20003
    value_vocab_size: int = 13,
    num_hidden_layers: int = 12
    num_train_negative_samples: int = 4
    num_valid_negative_samples: int = 4
    pad_index: int = 0
    mask_index: int = 1
    dropout: float = 0.1
    temperature: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 1e-2
