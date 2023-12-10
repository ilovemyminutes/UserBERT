from argparse import ArgumentParser, Namespace
from pathlib import Path

from __init__ import UserBERTBuilder


def main(config: Namespace):
    UserBERTBuilder(config).build()


if __name__ == "__main__":
    parser = ArgumentParser(description="user modeling")
    model_parser = parser.add_argument_group("model arguments")
    model_parser.add_argument("--embedding_dim", type=int, default=768)
    model_parser.add_argument("--intermediate_embedding_dim", type=int, default=768)
    model_parser.add_argument("--num_hidden_layers", type=int, default=8)
    model_parser.add_argument("--dropout", type=float, default=0.1)

    data_parser = parser.add_argument_group("data arguments")
    data_parser.add_argument("--log_start", type=str, default="2019-01-01")
    data_parser.add_argument("--log_end", type=str, default="2019-12-31")
    data_parser.add_argument(
        "--raw_data_dir", type=Path, default=Path("/Users/ilovemyminutes/Documents/workspace/data/ml-25m/")
    )
    data_parser.add_argument(
        "--user_bert_dir", type=Path, default=Path("/Users/ilovemyminutes/Documents/workspace/data/user_bert")
    )
    data_parser.add_argument("--num_items", type=int, default=20000)
    data_parser.add_argument("--num_users", type=int, default=-1)
    data_parser.add_argument("--valid_size", type=float, default=0.05)

    compute_parser = parser.add_argument_group("computation arguments")
    compute_parser.add_argument("--num_workers", type=int, default=6, help="number of processes for data loading")
    compute_parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
