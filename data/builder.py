from abc import abstractmethod, ABCMeta
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data.utils import (
    RAW_FILE,
    COL_USER_ID,
    COL_TIMESTAMP,
    COL_ITEM_ID,
    TOKEN_CLS,
    TOKEN_PAD,
    TOKEN_MASK,
)


class DataBuilder(metaclass=ABCMeta):
    def build(self):
        self.collect()
        self.initialize_tokenizer()
        self.make_dataset()
        self.finalize()

    @abstractmethod
    def collect(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def make_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError


class BehaviorDataBuilder:
    def __init__(
        self,
        data_dir: Path,
        save_dir: Path,
        log_start: datetime,
        log_end: datetime,
        test_log_start: datetime | None = None,
        num_items: int = 20000,
        rating_scale: int = 10,
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

        self.raw_data: pd.DataFrame | None = None
        self.item_tokenizer: dict[int, int] | None = None

    def collect(self):
        self.raw_data = pd.read_csv(self.data_dir / RAW_FILE).sort_values(
            by=[COL_USER_ID, COL_TIMESTAMP], ignore_index=True
        )

    def initialize_tokenizer(self):
        source = self.raw_data[
            (self.train_period[0].timestamp() <= self.raw_data[COL_TIMESTAMP])
            & (self.raw_data[COL_TIMESTAMP] <= self.train_period[1].timestamp())
        ]
        self.item_tokenizer = {TOKEN_PAD: 0, TOKEN_MASK: 1, TOKEN_CLS: 2}
        self.item_tokenizer.update(
            {
                item_id: i
                for i, item_id in enumerate(
                    source[COL_ITEM_ID].value_counts().head(self.num_items).index, start=len(self.item_tokenizer)
                )
            }
        )

    def make_dataset(self, period: tuple[datetime, datetime] | None):
        source = self.raw_data[
            (period[0].timestamp() <= self.raw_data[COL_TIMESTAMP])
            & (self.raw_data[COL_TIMESTAMP] <= period[1].timestamp())
            & (self.raw_data[COL_ITEM_ID].isin(self.item_tokenizer))
        ]
        return source
