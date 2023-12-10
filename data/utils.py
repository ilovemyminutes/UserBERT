import json
import pickle
from pathlib import Path

DATA_SPEC_FILE = "spec.json"
RAW_FILE = "ratings.csv"
ITEM_TOKENIZER_FILE = "item_tokenizer.pkl"
VALUE_TOKENIZER_FILE = "value_tokenizer.pkl"
PREPARED_FLAG = "prepared"

COL_USER_ID = "userId"
COL_ITEM_ID = "movieId"
COL_ITEM_VALUE = "rating"
COL_TIMESTAMP = "timestamp"

TOKEN_PAD = "[PAD]"
TOKEN_MASK = "[MASK]"
TOKEN_CLS = "[CLS]"
SPECIAL_TOKENS = (TOKEN_PAD, TOKEN_MASK, TOKEN_CLS)

TRAIN_DIR = "train"
VALID_DIR = "valid"
TEST_DIR = "test"

MODEL_DIR = "model"
PARAM_FILE = "params.pkl"


def get_version(
    version: str, user_bert_dir: Path, check_model_ready: bool = False, verbose: bool = True
) -> tuple[str, Path]:
    version_dir: Path | None = None
    if version == "latest":
        for path in sorted(user_bert_dir.glob("*[0-9]"), reverse=True):
            if (
                path.is_dir()
                and not path.stem.startswith(".")
                and (path / PREPARED_FLAG).exists()
                and (not check_model_ready or (path / MODEL_DIR / PREPARED_FLAG).exists())
            ):
                version, version_dir = path.stem, path
                break
    else:
        path = user_bert_dir / version
        if (path / PREPARED_FLAG).exists() and (not check_model_ready or (path / MODEL_DIR / PREPARED_FLAG).exists()):
            version_dir = path

    if version_dir is None:
        raise ValueError(f"version {version} does not exist or is not prepared completely")

    if verbose:
        logger.info(
            f"version: {version} ({version_dir})\n"
            f"  * pre-trained weights: {(version_dir / MODEL_DIR / PREPARED_FLAG).exists()}"
        )
        for k, v in load_data(version_dir / DATA_SPEC_FILE).items():
            logger.info(f"  * {k}: {v}")
    return version, version_dir


def dump_pickle(fpath: str | Path, data: object, protocol: int = pickle.DEFAULT_PROTOCOL, **kwargs):
    with open(fpath, "wb") as f:
        pickle.dump(data, f, protocol=protocol, **kwargs)


def load_pickle(fpath: str | Path, **kwargs) -> object:
    with open(fpath, "rb") as f:
        return pickle.load(f, **kwargs)


def dump_json(fpath: str | Path, data: object, **kwargs):
    with open(fpath, "w") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, fp=f, **kwargs)


def load_json(fpath: str | Path, **kwargs) -> object:
    with open(fpath) as f:
        return json.load(f, **kwargs)
