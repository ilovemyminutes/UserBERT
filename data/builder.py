import shutil
from abc import abstractmethod, ABCMeta
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from data.utils import (
    COL_USER_ID,
    COL_TIMESTAMP,
    COL_ITEM_VALUE,
    COL_ITEM_ID,
    DATA_SPEC_FILE,
    ITEM_TOKENIZER_FILE,
    PREPARED_FLAG,
    RAW_FILE,
    SPECIAL_TOKENS,
    TEST_DIR,
    TRAIN_DIR,
    VALUE_TOKENIZER_FILE,
    dump_json,
    dump_pickle,
    load_pickle,
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
    DATA_TYPE = {
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
        min_seq_len: int = 50,
        max_seq_len: int = -1,
        pretrained_tokenizer_dir: Path | None = None,
        n_jobs: int = 4,
    ):
        self.data_dir = data_dir
        self.num_items = num_items
        self.rating_scale = rating_scale
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.pretrained_tokenizer_dir = pretrained_tokenizer_dir
        self.n_jobs = n_jobs

        self.version: str = datetime.now().strftime("%Y%m%d%H%M")
        self.save_dir = save_dir / self.version
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.train_period: tuple[datetime, datetime] = (
            log_start,
            test_log_start - timedelta(seconds=1) if test_log_start is not None else log_end,
        )
        self.test_period: tuple[datetime, datetime] | None = (
            (test_log_start, log_end) if test_log_start is not None else None
        )

        self.raw_data: pd.DataFrame | None = None
        self.item_tokenizer: dict[int, int] | None = None
        self.value_tokenizer: dict[float, int] | None = None

    def collect(self):
        self.raw_data = pd.read_csv(self.data_dir / RAW_FILE, dtype=self.DATA_TYPE).sort_values(
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
        spec = {
            "version": self.version,
            "train_period": f"{self.train_period[0].strftime('%Y-%m-%d %H:%M:%S')} ~ {self.train_period[1].strftime('%Y-%m-%d %H:%M:%S')}",
            "test_period": ""
            if self.test_period is None
            else f"{self.test_period[0].strftime('%Y-%m-%d %H:%M:%S')} ~ {self.test_period[1].strftime('%Y-%m-%d %H:%M:%S')}",
            "item_vocab_size": len(self.item_tokenizer),
            "value_vocab_size": len(self.value_tokenizer),
            "num_special_tokens": len(SPECIAL_TOKENS),
        }
        dump_json(self.save_dir / DATA_SPEC_FILE, spec)
        with open(self.save_dir / PREPARED_FLAG, "w"):
            pass

    def _build_dataset(self, period: tuple[datetime, datetime] | None, save_dir: Path):
        save_dir.mkdir(exist_ok=True, parents=True)

        ray.init(num_cpus=4, runtime_env={"working_dir": Path(__file__).absolute().parent.parent})
        source = self.raw_data[
            (period[0].timestamp() <= self.raw_data[COL_TIMESTAMP])
            & (self.raw_data[COL_TIMESTAMP] <= period[1].timestamp())
            & (self.raw_data[COL_ITEM_ID].isin(self.item_tokenizer))
            & (self.raw_data[COL_ITEM_VALUE].isin(self.value_tokenizer))
        ]
        seq_len_by_user = source[COL_USER_ID].value_counts()
        seq_len_by_user = seq_len_by_user[self.min_seq_len <= seq_len_by_user]
        if self.max_seq_len != -1:
            seq_len_by_user = seq_len_by_user[seq_len_by_user <= self.max_seq_len]
        source = source[COL_USER_ID].isin(seq_len_by_user.index)

        source_ref: ray.ObjectRef = ray.put(source)
        item_tokenizer_ref: ray.ObjectRef = ray.put(self.item_tokenizer)
        value_tokenizer_ref: ray.ObjectRef = ray.put(self.value_tokenizer)

        futures = [
            self._build_dataset_by_one_process.remote(
                source_ref, set(split_user_pool), item_tokenizer_ref, value_tokenizer_ref, save_dir / f"{job_id}"
            )
            for job_id, split_user_pool in enumerate(np.array_split(sorted(source[COL_USER_ID].unique()), self.n_jobs))
        ]
        partition_dirs = ray.get(futures)
        ray.shutdown()
        self._merge_partitioned_datasets(partition_dirs, save_dir)

    def _initialize_item_tokenizer(self):
        if self.pretrained_tokenizer_dir is not None and (self.pretrained_tokenizer_dir / ITEM_TOKENIZER_FILE).exists():
            self.item_tokenizer = load_pickle(self.pretrained_tokenizer_dir / ITEM_TOKENIZER_FILE)
        else:
            self.item_tokenizer = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
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
        if (
            self.pretrained_tokenizer_dir is not None
            and (self.pretrained_tokenizer_dir / VALUE_TOKENIZER_FILE).exists()
        ):
            self.value_tokenizer = load_pickle(self.pretrained_tokenizer_dir / VALUE_TOKENIZER_FILE)
        else:
            self.value_tokenizer = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
            self.value_tokenizer.update(
                {
                    float(value / 2) if self.rating_scale == 10 else value: i
                    for i, value in enumerate(range(1, self.rating_scale + 1), start=len(self.value_tokenizer))
                }
            )
        dump_pickle(self.save_dir / VALUE_TOKENIZER_FILE, self.value_tokenizer)

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
