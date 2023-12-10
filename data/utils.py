import pickle
from pathlib import Path

RAW_FILE = "ratings.csv"

COL_USER_ID = "userId"
COL_ITEM_ID = "movieId"
COL_ITEM_VALUE = "rating"
COL_TIMESTAMP = "timestamp"

TOKEN_PAD = "[PAD]"
TOKEN_MASK = "[MASK]"
TOKEN_CLS = "[CLS]"

TRAIN_DIR = "train"
VALID_DIR = "valid"
TEST_DIR = "test"


def dump_pickle(fpath: str | Path, data: object, protocol: int = pickle.DEFAULT_PROTOCOL, **kwargs):
    with open(fpath, "wb") as f:
        pickle.dump(data, f, protocol=protocol, **kwargs)


def load_pickle(fpath: str | Path, **kwargs) -> object:
    with open(fpath, "rb") as f:
        return pickle.load(f, **kwargs)
