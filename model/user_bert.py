from typing import Optional

import pytorch_lightning as pl
from pytorch_optimizer import get_optimizer_parameters
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR

from model.user_encoder import UserEncoder


class UserBERT(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        action_type_vocab_size: int,
        content_vocab_size: int,
        num_hidden_layers: int = 12,
        num_train_negative_samples: int = 4,
        num_valid_negative_samples: int = 4,
        pad_index: int = 0,
        mask_index: int = 1,
        dropout: float = 0.1,
        temperature: float = 1.0,
        lr: float = 1e-5,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = UserEncoder(
            embedding_dim, action_type_vocab_size, content_vocab_size, num_hidden_layers, pad_index, dropout
        )
        self.classifier = nn.Linear(embedding_dim, content_vocab_size)

        self.train_k = num_train_negative_samples
        self.valid_k = num_valid_negative_samples
        self.mask_index = mask_index
        self.t = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()

        self.evaluation_step_outputs: list[
            tuple[any, any, any, any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

    def forward(
        self,
        action_types: torch.LongTensor,
        contents: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return self.encoder(action_types, contents, attention_mask)  # B x H

    def training_step(self, batch, _) -> torch.Tensor:
        (
            loss,
            loss_mbp,
            loss_bsm,
            loss_lbp,
            _,
            _,
            _,
            _,
            _,
            labels_lbp,  # only used to get batch size
        ) = self._shared_step(batch, k=self.train_k)

        batch_size = len(labels_lbp)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_loss_mbp", loss_mbp, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_loss_bsm", loss_bsm, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_loss_lbp", loss_lbp, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, _):
        """
        Notations:
            B: batch_size
            K: # negatives
            M: # masks
        """
        loss, loss_mbp, loss_bsm, loss_lbp, logits_mbp, logits_bsm, logits_lbp, _, _, labels_lbp = self._shared_step(
            batch, k=self.valid_k
        )
        result = loss, loss_mbp, loss_bsm, loss_lbp, logits_mbp, logits_bsm, logits_lbp, labels_lbp
        self.evaluation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        """
        Notations:
            B: batch size
            N: # batches
        """

        num_batches = len(self.evaluation_step_outputs)
        loss, loss_mbp, loss_bsm, loss_lbp, preds_mbp, preds_bsm, preds_lbp, labels_lbp = (
            0.0,
            0.0,
            0.0,
            0.0,
            [],
            [],
            [],
            [],
        )
        for ls, ls_mbp, ls_bsm, ls_lbp, lgt_mbp, lgt_bsm, lgt_lbp, lbl_lbp in self.evaluation_step_outputs:
            preds_mbp.append(lgt_mbp.argmax(dim=1))  # B
            preds_bsm.append(lgt_bsm.argmax(dim=1))  # B
            preds_lbp.append(lgt_lbp.argmax(dim=1))  # B
            labels_lbp.append(lbl_lbp)
            loss += ls
            loss_mbp += ls_mbp
            loss_bsm += ls_bsm
            loss_lbp += ls_lbp

        preds_mbp, preds_bsm, preds_lbp = torch.cat(preds_mbp), torch.cat(preds_bsm), torch.cat(preds_lbp)  # N*B
        labels_mbp = torch.zeros(len(preds_mbp), dtype=torch.long, device=preds_mbp.device)  # N*B
        labels_bsm = torch.zeros(len(preds_bsm), dtype=torch.long, device=preds_bsm.device)  # N*B
        labels_lbp = torch.cat(labels_lbp)

        loss /= num_batches
        loss_mbp /= num_batches
        loss_bsm /= num_batches
        loss_lbp /= num_batches
        acc_mbp = torch.eq(preds_mbp, labels_mbp).sum() / len(labels_mbp)
        acc_bsm = torch.eq(preds_bsm, labels_bsm).sum() / len(labels_bsm)
        acc_lbp = torch.eq(preds_lbp, labels_lbp).sum() / len(labels_lbp)
        self.log("valid_loss", loss, sync_dist=True)
        self.log("valid_loss_mbp", loss_mbp, sync_dist=True)
        self.log("valid_loss_bsm", loss_bsm, sync_dist=True)
        self.log("valid_loss_lbp", loss_lbp, sync_dist=True)
        self.log("valid_acc_mbp", acc_mbp, sync_dist=True)
        self.log("valid_acc_bsm", acc_bsm, sync_dist=True)
        self.log("valid_acc_lbp", acc_lbp, sync_dist=True)

        self.evaluation_step_outputs.clear()

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, any]]]:
        params_to_optimize = get_optimizer_parameters(self, weight_decay=self.weight_decay)
        optimizer = AdamW(params_to_optimize, lr=self.lr)
        scheduler = {
            "scheduler": OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.trainer.estimated_stepping_batches),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def _shared_step(self, batch, k: int):
        batch_mbp, batch_bsm, batch_lbp = batch[0], batch[1], batch[1][1]  # batch_lbp: BSM에서 생성한 시퀀스를 재사용
        loss_mbp, logits_mbp, labels_mbp = self._get_masked_behavior_prediction_result(
            *batch_mbp, num_negative_samples=k
        )  # B*M x K+1, B*M
        loss_bsm, logits_bsm, labels_bsm = self._get_behavior_sequence_matching_result(
            *batch_bsm, num_negative_samples=k
        )  # B x K+1, B
        loss_lbp, logits_lbp, labels_lbp = self._get_last_behavior_prediction_result(batch_lbp)
        loss = loss_mbp + loss_bsm + loss_lbp
        return (
            loss,
            loss_mbp,
            loss_bsm,
            loss_lbp,
            logits_mbp,
            logits_bsm,
            logits_lbp,
            labels_mbp,
            labels_bsm,
            labels_lbp,
        )

    def _get_masked_behavior_prediction_result(
        self,
        masked_seq: list[torch.Tensor],
        true_behaviors: list[torch.Tensor],
        masked_position_indices: torch.Tensor,
        num_negative_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MBP: Masked Behavior Prediction
        Notations:
            B: batch size
            S: sequence length
            M: # masks
            K: # negative samples
            H: embedding dim
        Arguments:
            masked_seq: B x S for each
            true_behaviors: B x M for each
            masked_position_indices: B x M
        """

        true_behaviors = [tb.flatten(0, 1) for tb in true_behaviors]
        masked_position_indices = masked_position_indices.flatten(0, 1).unsqueeze(-1)
        pos_b_embs = self.encoder.extract_behavior_embeddings(
            *true_behaviors, position_indices=masked_position_indices
        )  # B*M x 1 x H
        neg_b_embs = self._sample_negative_behavior_embeddings(
            true_behaviors, masked_position_indices, num_negative_samples
        )  # B*M x K x H
        b_embs = torch.cat([pos_b_embs, neg_b_embs], dim=1)  # B*M x K+1 x H

        anchor_embs = self.encoder.extract_behavior_context_embeddings(*masked_seq)  # B x S x H
        num_masks = b_embs.size(0) // anchor_embs.size(0)
        anchor_embs = anchor_embs.repeat_interleave(num_masks, dim=0)  # B*M x S x H
        anchor_embs = anchor_embs[range(anchor_embs.size(0)), masked_position_indices.squeeze(dim=1), :]  # B*M x H
        anchor_embs = anchor_embs.unsqueeze(dim=1)  # B*M x 1 x H

        logits = (anchor_embs @ b_embs.transpose(1, 2)).squeeze(dim=1)  # B*M x K+1
        labels = torch.zeros_like(logits[:, 0], dtype=torch.long)  # B*M (NOTE: all pseudo-labels are set as 0)
        loss = self.loss_fn(logits / self.t, labels)
        return loss, logits, labels

    def _get_behavior_sequence_matching_result(
        self, seq_0: list[torch.Tensor], seq_1: list[torch.Tensor], num_negative_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """BSM: Behavior Sequence Matching
        Notations:
            B: batch size
            S: sequence length
            K: # negative samples
            H: embedding dim
        Arguments:
            seq_0: B x S for each
            seq_1: B x S for each
        """
        seq_1_short = [s[:, :-1] for s in seq_1]
        anchor_u_embs = self.encoder(*seq_0)  # B x H
        pos_u_embs = self.encoder(*seq_1_short)  # B x H
        neg_u_embs = self._sample_negative_user_embeddings(pos_u_embs, num_negative_samples)  # B x K x H
        u_embs = torch.cat([pos_u_embs.unsqueeze(dim=1), neg_u_embs], dim=1)  # B x K+1 x H
        logits = torch.matmul(anchor_u_embs.unsqueeze(dim=1), u_embs.transpose(1, 2)).squeeze(dim=1)  # B x K+1
        labels = torch.zeros_like(logits[:, 0], dtype=torch.long)  # B (NOTE: all pseudo-labels are set as 0)
        loss = self.loss_fn(logits / self.t, labels)
        return loss, logits, labels

    def _get_last_behavior_prediction_result(
        self, raw_seq: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """LBP: Last Behavior Prediction
        Notations:
            B: batch size
            S: sequence length
            H: embedding dim
            C: # contents
        Arguments:
            raw_seq: B x (S + 1) for each
        """
        # raw_seq로부터 input behavior sequence와 content을 분리
        # Data leakage 방지를 위해 input behavior sequence 중 content을 포함한 behavior를 마스킹
        with torch.no_grad():
            content_seq: torch.tensor = raw_seq[1]
            labels: torch.tensor = content_seq[:, -1]
            same_as_content_indices: torch.Tensor = (
                torch.eq(content_seq[:, :-1], labels.unsqueeze(1)).nonzero().unbind(dim=1)
            )
            seq_l = []
            for s in raw_seq:
                s = s[:, :-1].clone()
                s[same_as_content_indices[0], same_as_content_indices[1]] = self.mask_index
                seq_l.append(s)
        logits = self.classifier(self.encoder(*seq_l))  # B x C
        loss = self.loss_fn(logits / self.t, labels)
        return loss, logits, labels

    def _sample_negative_behavior_embeddings(
        self, positive_behaviors: list[torch.Tensor], position_indices: torch.Tensor, num_negative_samples: int
    ) -> torch.Tensor:
        """각 Positive behaviors에 대응되는 medium hard negative sample을 추출하는 함수
        Neg. sample 후보군은 입력된 pos. behaviors를 바탕으로 구성되며,
        각 pos. behavior는 자기 자신과 다른 후보 neg. sample 중 내적 값이 가장 큰 K개 sample을 neg. sample로 취한다.
        엄밀하게는 후보로부터 neg.sample을 추출하는 과정에서 pos. behavior들 각각의 position 정보를 고려해야 하나,
        연산 효율을 위해 해당 정보를 임의 단일 position으로 가정하고 neg. sample을 추출한다.

        Notations:
            B: batch size
            M: # masks
            P: candidate pool size
            K: # negative samples
            H: embedding dim
        Arguments:
            positive_behaviors B*M x 1 for each
            position_indices: B*M x 1
        """

        # query
        with torch.no_grad():
            pos_b_mat = torch.concat(positive_behaviors, dim=1)  # B*M x 2
            # unique: 동일 behavior 중복 선정 방지
            candidate_pool, candidate_idx = pos_b_mat.unique(dim=0, return_inverse=True)  # P x 2
            # postive와 동일한 콘텐츠에 대한 behavior는 negative sample에서 제외
            ignores_mat = (pos_b_mat[:, 1].unsqueeze(1) == candidate_pool[:, 1].unsqueeze(0)) * -1e4  # B*M x P
            # 학습 효율을 위해 포지션을 랜덤하게 하나로 고정하고 sampling
            sample_position_idx = position_indices[torch.randint(position_indices.size(0), (1, 1))].squeeze()
            pool_embs = self.encoder.extract_behavior_embeddings(
                candidate_pool[:, :1], candidate_pool[:, 1:], position_indices=sample_position_idx
            ).squeeze(
                dim=1
            )  # P x H
            pos_embs = pool_embs[candidate_idx, :]  # B*M x H
            _, neg_indices = (pos_embs @ pool_embs.T + ignores_mat).topk(k=num_negative_samples)  # B*M x K
            neg_behaviors = candidate_pool[candidate_idx[neg_indices].flatten(), :]  # B*M*K x 2
            neg_position_indices = position_indices.repeat_interleave(num_negative_samples).unsqueeze(1)  # B*M*K x 1

        # extract
        neg_b_embs = self.encoder.extract_behavior_embeddings(
            neg_behaviors[:, :1], neg_behaviors[:, 1:], position_indices=neg_position_indices
        )  # B*M*K x 1 x H
        neg_b_embs = neg_b_embs.squeeze(1).reshape(
            neg_b_embs.size(0) // num_negative_samples, num_negative_samples, neg_b_embs.size(2)
        )  # B*M x K x H
        return neg_b_embs

    @staticmethod
    def _sample_negative_user_embeddings(
        positive_user_embeddings: torch.Tensor, num_negative_samples: int
    ) -> torch.Tensor:
        """
        Notations:
            B: batch size
            P: candidate pool size
            S: max sequence length
            K: # negative samples
            H: embedding dim
        Arguments:
            positive_user_embeddings: B x H
        """

        # query
        with torch.no_grad():
            # Positive 자체가 negative가 되는 경우를 방지
            ignores_mat = (torch.ones_like(positive_user_embeddings[:, 0]) * -1e4).diag()
            _, neg_indices = (positive_user_embeddings @ positive_user_embeddings.T + ignores_mat).topk(
                k=num_negative_samples
            )  # B x K

        # extract
        neg_u_embs = positive_user_embeddings[neg_indices.flatten(), :]  # B*K x H
        neg_u_embs = neg_u_embs.reshape(
            neg_u_embs.size(0) // num_negative_samples, num_negative_samples, neg_u_embs.size(1)
        )  # B x K x H
        return neg_u_embs
