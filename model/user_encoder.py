from abc import ABC
from typing import Optional

import torch
from torch import nn
from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder


class UserEncoder(BertPreTrainedModel, ABC):
    def __init__(
        self,
        embedding_dim: int,
        item_vocab_size: int,
        rating_scale: int,
        num_hidden_layers: int = 8,
        pad_index: int = 0,
        dropout: float = 0.1,
    ):
        config = BertConfig(
            item_vocab_size=item_vocab_size,
            rating_scale=rating_scale,
            hidden_size=embedding_dim,
            pad_token_id=pad_index,
            hidden_dropout_prob=dropout,
            num_hidden_layers=num_hidden_layers,
        )
        super().__init__(config)
        self.config = config
        self.behavior_encoder = BehaviorEmbedding(config)
        self.behavior_context_encoder = BehaviorContextEncoder(config)
        self.post_init()

    def forward(
        self,
        item_ids: torch.LongTensor,
        rating_values: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ):
        return self.extract_user_embeddings(item_ids, rating_values, attention_mask)

    def extract_behavior_embeddings(
        self,
        item_ids: torch.LongTensor,
        rating_values: torch.LongTensor,
        position_indices: torch.LongTensor | None = None,
    ):
        return self.behavior_encoder(item_ids, rating_values, position_indices)

    def extract_behavior_context_embeddings(
        self,
        item_ids: torch.LongTensor,
        rating_values: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ):
        input_shape = item_ids.size()
        device = item_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask: torch.Tensor = self.get_head_mask(None, self.config.num_hidden_layers)

        behavior_embeddings = self.extract_behavior_embeddings(item_ids, rating_values)
        context_embeddings = self.behavior_context_encoder(
            behavior_embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        return context_embeddings

    def extract_user_embeddings(
        self,
        item_ids: torch.LongTensor,
        rating_values: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ):
        context_embeddings = self.extract_behavior_context_embeddings(item_ids, rating_values, attention_mask)
        user_embeddings = context_embeddings[:, 0, :]  # [CLS]
        return user_embeddings


class BehaviorContextEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.encoder = BertEncoder(config)

    def forward(
        self,
        behavior_embeddings: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(behavior_embeddings, attention_mask, head_mask=head_mask, return_dict=False)
        context_embeddings = encoder_outputs[0]
        return context_embeddings


class BehaviorEmbedding(nn.Module):
    """
    Reference. https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_bert.html#BertModel
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.rating_embeddings = nn.Embedding(
            config.rating_scale + 1,
            config.hidden_size // 2,
            padding_idx=config.pad_token_id,
        )
        self.item_embeddings = nn.Embedding(
            config.item_vocab_size,
            config.hidden_size // 2,
            padding_idx=config.pad_token_id,
        )

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        item_ids: torch.LongTensor,
        rating_values: torch.LongTensor,
        position_indices: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if position_indices is not None and position_indices.ndim != 2 and position_indices.ndim != 0:
            raise ValueError(f"ndim of position_indices({position_indices}) should be 2 or 0")

        seq_length = item_ids.size(1)
        position_ids = (
            self.position_ids[:, :seq_length]
            if position_indices is None
            else self.position_ids[:, position_indices].squeeze(dim=0)
        )
        item_embedding = self.item_embeddings(item_ids)
        rating_embedding = self.rating_embeddings(rating_values)
        embeddings = torch.cat([item_embedding, rating_embedding], dim=-1)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
