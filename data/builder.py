import math
import shutil
from abc import abstractmethod, ABCMeta
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from data.utils import (
    RAW_FILE,
    COL_USER_ID,
    COL_TIMESTAMP,
    COL_ITEM_VALUE,
    COL_ITEM_ID,
    TEST_DIR,
    TOKEN_CLS,
    TOKEN_PAD,
    TOKEN_MASK,
    TRAIN_DIR,
    ITEM_TOKENIZER_FILE,
    VALUE_TOKENIZER_FILE,
    dump_pickle
)

USER_FILE = "users.txt"
ITEM_IDS_FILE = "item_ids.txt"
VALUES_FILE = "values.txt"
TIMESTAMPS_FILE = "timestamps.txt"
BEHAVIOR_DATA_FILES = [USER_FILE, ITEM_IDS_FILE, VALUES_FILE, TIMESTAMPS_FILE]


class DataBuilder(metaclass=ABCMeta):
    def build(self):
        self.collect()
        self.initialize_tokenizers()
        self.build_datasets()
        self.finalize()

    @abstractmethod
    def collect(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_tokenizers(self):
        raise NotImplementedError

    @abstractmethod
    def build_datasets(self):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError


class BehaviorDataBuilder(DataBuilder):
    DTYPE = {
        COL_USER_ID: np.int32,
        COL_ITEM_ID: np.int32,
        COL_ITEM_VALUE: np.float32,
        COL_TIMESTAMP: np.int32,
    }

    def __init__(
        self,
        data_dir: Path,
        save_dir: Path,
        log_start: datetime,
        log_end: datetime,
        test_log_start: datetime | None = None,
        num_items: int = 20000,
        rating_scale: int = 10,
        n_jobs: int = 4,
    ):
        self.data_dir = data_dir
        self.save_dir = save_dir

        self.train_period: tuple[datetime, datetime] = (
            log_start,
            test_log_start - timedelta(seconds=1) if test_log_start is not None else log_end,
        )
        self.test_period: tuple[datetime, datetime] | None = (
            (test_log_start, log_end) if test_log_start is not None else None
        )

        self.num_items = num_items
        self.rating_scale = rating_scale
        self.n_jobs = n_jobs

        self.raw_data: pd.DataFrame | None = None
        self.item_tokenizer: dict[int, int] | None = None
        self.value_tokenizer: dict[float, int] | None = None

    def collect(self):
        self.raw_data = pd.read_csv(self.data_dir / RAW_FILE, dtype=self.DTYPE).sort_values(
            by=[COL_USER_ID, COL_TIMESTAMP], ignore_index=True
        )

    def initialize_tokenizers(self):
        self._initialize_item_tokenizer()
        self._initialize_value_tokenizer()

    def build_datasets(self):
        self._build_dataset(self.train_period, self.save_dir / TRAIN_DIR)
        if self.test_period is not None:
            self._build_dataset(self.test_period, self.save_dir / TEST_DIR)

    def finalize(self):
        return

    def _build_dataset(self, period: tuple[datetime, datetime] | None, save_dir: Path):
        save_dir.mkdir(exist_ok=True, parents=True)

        ray.init(num_cpus=4, runtime_env={"working_dir": Path(__file__).absolute().parent.parent})
        source = self.raw_data[
            (period[0].timestamp() <= self.raw_data[COL_TIMESTAMP])
            & (self.raw_data[COL_TIMESTAMP] <= period[1].timestamp())
            & (self.raw_data[COL_ITEM_ID].isin(self.item_tokenizer))
            & (self.raw_data[COL_ITEM_VALUE].isin(self.value_tokenizer))
        ]
        source_ref: ray.ObjectRef = ray.put(source)
        item_tokenizer_ref: ray.ObjectRef = ray.put(self.item_tokenizer)
        value_tokenizer_ref: ray.ObjectRef = ray.put(self.value_tokenizer)

        futures = [
            self._build_dataset_by_one_process.remote(
                source_ref, split_user_pool, item_tokenizer_ref, value_tokenizer_ref, save_dir / f"{job_id}"
            )
            for job_id, split_user_pool in enumerate(
                self._split_user_pool(set(source[COL_USER_ID]), n_splits=self.n_jobs)
            )
        ]
        partition_dirs = ray.get(futures)
        ray.shutdown()
        self._merge_partitioned_datasets(partition_dirs, save_dir)

    def _initialize_item_tokenizer(self):
        self.item_tokenizer = {TOKEN_PAD: 0, TOKEN_MASK: 1, TOKEN_CLS: 2}
        source = self.raw_data[
            (self.train_period[0].timestamp() <= self.raw_data[COL_TIMESTAMP])
            & (self.raw_data[COL_TIMESTAMP] <= self.train_period[1].timestamp())
        ]
        self.item_tokenizer.update(
            {
                item_id: i
                for i, item_id in enumerate(
                    source[COL_ITEM_ID].value_counts().head(self.num_items).index, start=len(self.item_tokenizer)
                )
            }
        )
        dump_pickle(self.save_dir / ITEM_TOKENIZER_FILE, self.item_tokenizer)

    def _initialize_value_tokenizer(self):
        self.value_tokenizer = {TOKEN_PAD: 0, TOKEN_MASK: 1, TOKEN_CLS: 2}
        self.value_tokenizer.update(
            {
                float(value / 2) if self.rating_scale == 10 else value: i
                for i, value in enumerate(range(1, self.rating_scale + 1), start=len(self.value_tokenizer))
            }
        )
        dump_pickle(self.save_dir / VALUE_TOKENIZER_FILE, self.value_tokenizer)

    @staticmethod
    def _split_user_pool(user_pool: set[int], n_splits: int) -> list[set[int]]:
        user_list = list(user_pool)
        subset_size = math.ceil(len(user_pool) / n_splits)
        splits = [set(user_list[k * subset_size : (k + 1) * subset_size]) for k in range(n_splits)]
        return splits

    @staticmethod
    @ray.remote
    def _build_dataset_by_one_process(
        source: ray.ObjectRef,
        user_pool: set[int],
        item_tokenizer: ray.ObjectRef,
        value_tokenizer: ray.ObjectRef,
        save_dir: Path,
    ):
        save_dir.mkdir(exist_ok=True, parents=True)
        source = source[source[COL_USER_ID].isin(user_pool)].reset_index(drop=True)
        source[COL_ITEM_ID] = source[COL_ITEM_ID].map(item_tokenizer)
        source[COL_ITEM_VALUE] = source[COL_ITEM_VALUE].map(value_tokenizer)
        files = [open(save_dir / f, "w") for f in BEHAVIOR_DATA_FILES]
        for u, sequences in tqdm(
            source.groupby(COL_USER_ID)[[COL_ITEM_ID, COL_ITEM_VALUE, COL_TIMESTAMP]].agg(list).iterrows(),
            total=source[COL_USER_ID].nunique(),
            desc="build dataset",
        ):
            data = [u] + [",".join(str(x) for x in seq) for seq in sequences]
            for f, d in zip(files, data):
                f.write(f"{d}\n")
        for f in files:
            f.close()
        return save_dir

    @staticmethod
    def _merge_partitioned_datasets(partition_dirs: list[Path], save_dir: Path):
        for d in tqdm(BEHAVIOR_DATA_FILES, desc="merge partitioned datasets"):
            f = open(save_dir / d, "w")
            for p_dir in partition_dirs:
                for v in open(p_dir / d, "r"):
                    f.write(v)
            f.close()
        for p_dir in partition_dirs:
            shutil.rmtree(p_dir)
