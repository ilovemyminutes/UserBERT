from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.reader import BehaviorDataReader
from pretrain.preprocessing import (
    mask_behavior_sequence_by_items,
    pack_behavior_sequence,
    sample_behavior_sequence_pair,
    sample_single_behavior_sequence,
)


class PretrainDataset(Dataset, BehaviorDataReader):
    def __init__(
        self,
        data_dir: Path,
        user_pool: set[int] | None = None,
        bsm_seq_len: int = 200,
        mbp_seq_len: int = 400,
        pad_index: int = 0,
        mask_index: int = 1,
        cls_index: int = 2,
        mask_prob: float = 0.1,
    ):
        super().__init__(data_dir)
        self.data_dir = data_dir

        self.bsm_seq_len = bsm_seq_len
        self.mbp_seq_len = mbp_seq_len

        self.pad_idx = pad_index
        self.mask_idx = mask_index
        self.cls_index = cls_index

        self.user_ids: list[int] = sorted(user_pool & self.user_pool) if user_pool is not None else list(self.user_pool)
        self.num_masks = int(mbp_seq_len * mask_prob)

    def __getitem__(
        self, index: int
    ) -> tuple[
        tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor], tuple[list[torch.Tensor], list[torch.Tensor]]
    ]:
        user_id, seq = self.read(user_id=self.user_ids[index])

        # MBP: masked behavior prediction
        seq_m = sample_single_behavior_sequence(seq, max_seq_len=self.mbp_seq_len - 1)  # -1 for [CLS]
        packed_seq_m = pack_behavior_sequence(seq_m, self.mbp_seq_len, self.cls_index, self.pad_idx)
        packed_masked_seq, true_behaviors, masked_pos_indices = mask_behavior_sequence_by_items(
            packed_seq_m, self.mask_idx, self.num_masks, no_mask_at={0}
        )

        # BSM: behavior sequence matching
        seq_0, seq_1 = sample_behavior_sequence_pair(seq, self.bsm_seq_len)  # -1 for [CLS]
        packed_seq_0 = pack_behavior_sequence(seq_0, self.bsm_seq_len, self.cls_index, self.pad_idx)
        packed_seq_1 = pack_behavior_sequence(seq_1, self.bsm_seq_len, self.cls_index, self.pad_idx)

        batch_mbp = (
            [torch.from_numpy(s) for s in packed_masked_seq],
            [torch.from_numpy(b) for b in true_behaviors],
            torch.from_numpy(masked_pos_indices),
        )
        batch_bsm = ([torch.from_numpy(s) for s in packed_seq_0], [torch.from_numpy(s) for s in packed_seq_1])
        return batch_mbp, batch_bsm

    def __len__(self) -> int:
        return len(self.user_ids)
