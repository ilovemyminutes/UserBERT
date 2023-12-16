from collections import defaultdict

import numpy as np


def pack_behavior_sequence(
    behavior_sequence: list[np.ndarray],
    max_seq_len: int,
    cls_index: int = 2,
    pad_index: int = 0,
) -> list[np.ndarray]:
    seq_len: int = len(behavior_sequence[0])
    max_seq_len -= 1  # -1 for [CLS]

    num_pads = 0
    cls = np.full(1, fill_value=cls_index)
    if seq_len < max_seq_len:  # NOTE. in 1D array, np.concatenate is 10x faster than np.pad
        num_pads = max_seq_len - seq_len
        packed_seq = [np.concatenate([cls, seq, [pad_index] * num_pads]) for seq in behavior_sequence]
    elif seq_len > max_seq_len:
        start_idx = np.random.choice(seq_len - max_seq_len)
        packed_seq = [np.concatenate([cls, seq[start_idx : start_idx + max_seq_len]]) for seq in behavior_sequence]
    else:
        packed_seq = [np.concatenate([cls, seq]) for seq in behavior_sequence]
    packed_seq.append(
        np.ones(max_seq_len + 1, dtype=np.int64)
        if num_pads == 0
        else np.concatenate([np.ones(max_seq_len - num_pads + 1, dtype=np.int64), np.zeros(num_pads, dtype=np.int64)])
    )
    return packed_seq


def sample_single_behavior_sequence(behavior_sequence: list[np.ndarray], max_seq_len: int) -> list[np.ndarray]:
    seq_len = len(behavior_sequence[0])
    if seq_len > max_seq_len:
        start_idx = np.random.choice(seq_len - max_seq_len)
        return [s[start_idx : start_idx + max_seq_len] for s in behavior_sequence]
    return behavior_sequence


def sample_behavior_sequence_pair(
    behavior_sequence: list[np.ndarray], max_seq_len: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    seq_len = len(behavior_sequence[0])
    total_seq_len = 2 * max_seq_len
    seq_0, seq_1 = [], []
    if seq_len == total_seq_len:
        for seq in behavior_sequence:
            seq_0.append(seq[:seq_len])
            seq_1.append(seq[seq_len : seq_len * 2])
    elif seq_len > total_seq_len:
        # 1st sequence boundary
        left_0 = np.random.choice(seq_len - total_seq_len)
        right_0 = left_0 + max_seq_len

        candidates = []
        if seq_len - right_0 >= max_seq_len:  # 첫 시퀀스 우측에서 샘플링 가능할 경우
            candidates += list(range(right_0, seq_len - max_seq_len))
        if left_0 >= max_seq_len:  # 첫 시퀀스 좌측에서 샘플링 가능할 경우
            candidates += list(range(left_0 - max_seq_len))

        # 2nd sequence boundary
        left_1 = np.random.choice(candidates)
        right_1 = left_1 + max_seq_len

        for seq in behavior_sequence:
            seq_0.append(seq[left_0:right_0])
            seq_1.append(seq[left_1:right_1])
    else:
        split_idx = np.random.choice(seq_len)
        for seq in behavior_sequence:
            seq_0.append(seq[:split_idx])
            seq_1.append(seq[split_idx:])
    return seq_0, seq_1


def mask_behavior_sequence_by_items(
    behavior_sequence: list[np.ndarray],
    mask_index: int = 1,
    num_mask_items: int = 10,
    no_mask_at: set[int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    indices_by_item: defaultdict[int, list[int]] = defaultdict(list)
    for i, item_idx in enumerate(behavior_sequence[0]):
        if no_mask_at is not None and i in no_mask_at:
            continue
        indices_by_item[item_idx].append(i)
    if len(indices_by_item) <= num_mask_items:
        return mask_behavior_sequence(behavior_sequence, mask_index, num_mask_items, no_mask_at)

    masked_seq: list[np.ndarray] = [s.copy() for s in behavior_sequence]
    items_to_mask = np.random.choice(list(indices_by_item), size=num_mask_items, replace=False)
    masked_pos_to_pred, masked_pos_all = [], []
    for t in items_to_mask:
        indices = indices_by_item[t]
        idx = np.random.choice(indices)
        masked_pos_to_pred.append(idx)
        masked_pos_all.extend(indices)
    masked_pos_to_pred = np.sort(masked_pos_to_pred)
    true_behaviors: list[np.ndarray] = [s[masked_pos_to_pred][:, np.newaxis] for s in masked_seq]
    for s in masked_seq:
        s[masked_pos_all] = mask_index
    return masked_seq, true_behaviors, masked_pos_to_pred


def mask_behavior_sequence(
    behavior_sequence: list[np.ndarray], mask_index: int = 1, num_masks: int = 10, no_mask_at: set[int] | None = None
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    masked_seq: list[np.ndarray] = [s.copy() for s in behavior_sequence]
    masked_pos_to_pred = np.sort(
        np.random.choice([i for i in range(len(masked_seq[0])) if i not in no_mask_at], size=num_masks, replace=False)
    )
    true_behaviors: list[np.ndarray] = [s[masked_pos_to_pred][:, np.newaxis] for s in masked_seq]
    for s in masked_seq:
        s[masked_pos_to_pred] = mask_index
    return masked_seq, true_behaviors, masked_pos_to_pred
