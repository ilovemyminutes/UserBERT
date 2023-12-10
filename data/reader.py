from functools import cached_property
import linecache
from pathlib import Path
from typing import Optional

import numpy as np

from data.builder import USER_FILE, ITEM_IDS_FILE, VALUES_FILE, TIMESTAMPS_FILE


class BehaviorDataReader:
    def __init__(self, data_dir: Path, read_timestamp: bool = False):
        self.data_dir = data_dir
        self.read_timestamp = read_timestamp

        self.us_path = str(data_dir / USER_FILE)
        self.it_path = str(data_dir / ITEM_IDS_FILE)
        self.vl_path = str(data_dir / VALUES_FILE)
        self.ts_path = str(data_dir / TIMESTAMPS_FILE)

        self.lineno_by_user: Optional[dict[int, int]] = None
        self.update_cache()

    def read(self, index: int | None = None, user_id: int | None = None) -> tuple[int, list[np.ndarray]]:
        if index is not None:
            lineno = index + 1
            user_id: int = int(linecache.getline(self.us_path, lineno))
        elif user_id is not None:
            lineno = self.lineno_by_user.get(user_id)
            if lineno is None:
                raise ValueError(f"user_id {user_id} does not exist.")
        else:
            raise ValueError("input one of index or user_id.")
        seq = [
            np.fromstring(linecache.getline(self.it_path, lineno), sep=",", dtype=np.int64),
            np.fromstring(linecache.getline(self.vl_path, lineno), sep=",", dtype=np.int64),
        ]
        if self.read_timestamp:
            seq.append(np.fromstring(linecache.getline(self.ts_path, lineno), sep=",", dtype=np.int64))
        return user_id, seq

    def __len__(self):
        return self.size

    @property
    def user_pool(self) -> set[int]:
        return set(self.lineno_by_user.keys())

    @cached_property
    def size(self) -> int:
        return len(self.lineno_by_user)

    def update_cache(self) -> None:
        linecache.updatecache(self.us_path)
        linecache.updatecache(self.it_path)
        linecache.updatecache(self.vl_path)
        if self.read_timestamp:
            linecache.updatecache(self.ts_path)

        self.lineno_by_user: dict[int, int] = {}
        with open(self.us_path, "r") as f:
            for lineno, user_id in enumerate(f, start=1):
                self.lineno_by_user[int(user_id)] = lineno

    @staticmethod
    def clear() -> None:
        linecache.clearcache()
