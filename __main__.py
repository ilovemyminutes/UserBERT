from pathlib import Path

from __init__ import UserBERTPretrainingModule
from config import get_config

WORKSPACE_DIR = Path("/Users/ilovemyminutes/Documents/workspace/")
RAW_DATA_DIR = WORKSPACE_DIR / "data/ml-25m"
USER_BERT_DIR = WORKSPACE_DIR / "data/user_bert"


def main():
    config = get_config()
    UserBERTPretrainingModule(config).build()
