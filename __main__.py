from argparse import ArgumentParser, Namespace
from pathlib import Path

from __init__ import UserBERTPretrainingModule

WORKSPACE_DIR = Path("/Users/ilovemyminutes/Documents/workspace/")
RAW_DATA_DIR = WORKSPACE_DIR / "data/ml-25m"
USER_BERT_DIR = WORKSPACE_DIR / "data/user_bert"


def main(config: Namespace):
    UserBERTPretrainingModule(config).build()


if __name__ == "__main__":
    parser = ArgumentParser(description="user modeling")
    model_parser = parser.add_argument_group("model arguments")
    model_parser.add_argument("--embedding_dim", type=int, default=120)
    model_parser.add_argument("--num_hidden_layers", type=int, default=8)
    model_parser.add_argument("--dropout", type=float, default=0.1)

    data_parser = parser.add_argument_group("data arguments")
    data_parser.add_argument("--log_start", type=str, default="2019-01-01")
    data_parser.add_argument("--log_end", type=str, default="2019-12-31")
    data_parser.add_argument("--raw_data_dir", type=Path, default=RAW_DATA_DIR)
    data_parser.add_argument("--user_bert_dir", type=Path, default=USER_BERT_DIR)
    data_parser.add_argument("--num_items", type=int, default=20000)
    data_parser.add_argument("--num_users", type=int, default=-1)
    data_parser.add_argument("--valid_size", type=float, default=0.05)

    train_parser = parser.add_argument_group("pretrain arguments")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--mbp_seq_len", type=int, default=400)
    train_parser.add_argument("--bsm_seq_len", type=int, default=200)
    train_parser.add_argument("--num_train_negative_samples", type=int, default=4)
    train_parser.add_argument("--num_valid_negative_samples", type=int, default=4)
    train_parser.add_argument("--mask_prob", type=float, default=0.1)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay rate")

    logging_parser = parser.add_argument_group("experiment logging arguments")
    logging_parser.add_argument("--project", type=str, default="user-modeling")
    logging_parser.add_argument("--name", type=str, default="experiment-1")
    logging_parser.add_argument("--offline", action="store_true")
    logging_parser.add_argument("--ckpt_dir", type=Path, default=USER_BERT_DIR / "ckpt")

    compute_parser = parser.add_argument_group("computation arguments")
    compute_parser.add_argument("--devices", type=int, default=1)
    compute_parser.add_argument("--num_workers", type=int, default=6, help="number of processes for data loading")
    compute_parser.add_argument("--precision", type=int, default=16)
    compute_parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
