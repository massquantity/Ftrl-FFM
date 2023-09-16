import argparse
import time
from dataclasses import dataclass, fields
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def parse_args():
    parser = argparse.ArgumentParser(description="generate libcsv or libffm data")
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="split into train/eval data using a single dataset",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="",
        help="file path for train data",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="",
        help="file path for eval data",
    )
    parser.add_argument(
        "--train_output_path",
        default="",
        type=str,
        help="file path for saving transformed train data",
    )
    parser.add_argument(
        "--eval_output_path",
        default="",
        type=str,
        help="file path for saving transformed eval data",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="train fraction when splitting data",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="threshold for converting labels into 1 or 0",
    )
    parser.add_argument(
        "--neg_sampling",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="whether to use negative sampling",
    )
    parser.add_argument(
        "--num_neg",
        type=int,
        default=1,
        help="number of negative samples generated per sample",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=",",
        help="delimiter used in one sample",
    )
    parser.add_argument(
        "--label_col",
        type=int,
        default=0,
        help="label column index in data",
    )
    parser.add_argument(
        "--cat_cols",
        type=str,
        default="",
        help="categorical column indices in string format, e.g., 1,2,3,5,7",
    )
    parser.add_argument(
        "--num_cols",
        type=str,
        default="",
        help="numerical column indices in string format, e.g., 2,5,8,11,15",
    )
    parser.add_argument(
        "--normalize",
        type=str2bool,
        nargs="?",
        const=False,
        default=False,
        help="whether to normalize numerical features",
    )
    parser.add_argument(
        "--ffm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="whether to convert to libffm data format",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    return parser.parse_args()


@dataclass
class Arguments:
    data_path: str
    train_path: str
    eval_path: str
    train_output_path: str
    eval_output_path: str
    train_frac: float
    threshold: float
    neg_sampling: bool
    num_neg: int
    sep: str
    label_col: int
    cat_cols: Union[List[int], str]
    num_cols: Union[List[int], str]
    normalize: bool
    ffm: bool
    seed: int

    def __post_init__(self):
        if self.cat_cols:
            self.cat_cols = list(map(int, self.cat_cols.split(",")))
        else:
            self.cat_cols = list()
        if self.num_cols:
            self.num_cols = list(map(int, self.num_cols.split(",")))
        else:
            self.num_cols = list()


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ("yes", "true", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean-like value expected..., got `{v}`")


def read_data(args: Arguments):
    if args.data_path:
        data = pd.read_csv(args.data_path, sep=args.sep)
        train_data, eval_data = train_test_split(
            data, train_size=args.train_frac, random_state=args.seed
        )
        train_data = train_data.reset_index(drop=True)
        eval_data = eval_data.reset_index(drop=True)
    elif args.train_path and args.eval_path:
        train_data = pd.read_csv(args.train_path, sep=args.sep)
        eval_data = pd.read_csv(args.eval_path, sep=args.sep)
    else:
        raise ValueError("Must provide `data_path` or `train_path` && `eval_path`")

    # for implicit data, convert all labels to 1
    if args.neg_sampling:
        train_data.iloc[:, args.label_col] = 1
        eval_data.iloc[:, args.label_col] = 1
    else:
        train_data.iloc[:, args.label_col] = train_data.iloc[:, args.label_col].apply(
            lambda x: 1 if x > args.threshold else 0
        )
        eval_data.iloc[:, args.label_col] = eval_data.iloc[:, args.label_col].apply(
            lambda x: 1 if x > args.threshold else 0
        )
    return train_data, eval_data


def normalize_data(
    train_data: pd.DataFrame, eval_data: pd.DataFrame, num_cols: List[int]
):
    for col in num_cols:
        scaler = MinMaxScaler()
        train_feat = train_data.iloc[:, col].to_numpy().reshape(-1, 1)
        train_data.iloc[:, col] = scaler.fit_transform(train_feat).flatten()
        eval_feat = eval_data.iloc[:, col].to_numpy().reshape(-1, 1)
        eval_data.iloc[:, col] = scaler.transform(eval_feat).flatten()
    return train_data, eval_data


def get_unique_mapping(values: pd.Series, offset: int):
    unique_vals = np.unique(values)
    unique_idxs = np.arange(len(unique_vals)) + offset
    return dict(zip(unique_vals, unique_idxs))


def categorical_neg_sampling(
    rng: np.random.Generator,
    feat_idx: np.ndarray,
    unique_indices: List[int],
    num_neg: int,
):
    assert num_neg > 0
    num = len(feat_idx) * num_neg
    neg_feat_idx = rng.choice(unique_indices, size=num, replace=True)
    return np.append(feat_idx, neg_feat_idx)


def numerical_neg_sampling(
    rng: np.random.Generator, feat_val: np.ndarray, num_neg: int
):
    dtype = feat_val.dtype
    num = len(feat_val) * num_neg
    min_val, max_val = np.min(feat_val), np.max(feat_val)
    if np.issubdtype(dtype, np.integer):
        neg_feat_val = rng.integers(min_val, max_val, size=num, dtype=dtype, endpoint=True)
    else:
        neg_feat_val = rng.random(size=num, dtype=dtype) * (max_val - min_val) + min_val
    return np.append(feat_val, neg_feat_val)


def convert_categorical_str(feat_idx: np.ndarray, field: int, ffm_format: bool):
    """format: field:feat:1"""
    str_col = pd.Series(feat_idx, dtype=str)
    str_col = str_col.str.cat(["1"] * len(feat_idx), sep=":")
    if ffm_format:
        vals = [field] * len(feat_idx)
        str_col = pd.Series(vals, dtype=str).str.cat(str_col, sep=":")
    return str_col


def convert_numerical_str(feat_val: np.ndarray, feat_idx: int, field: int, ffm_format: bool):
    """format: field:feat:val"""
    if np.issubdtype(feat_val.dtype, np.floating):
        feat_val = np.round(feat_val, 4)
    str_col = pd.Series([feat_idx] * len(feat_val), dtype=str)
    str_col = str_col.str.cat(feat_val.astype(str), sep=":")
    if ffm_format:
        vals = [field] * len(feat_val)
        str_col = pd.Series(vals, dtype=str).str.cat(str_col, sep=":")
    return str_col


def convert_label_str(data: pd.DataFrame, args: Arguments):
    labels = data.iloc[:, args.label_col].astype(str)
    if args.neg_sampling and args.num_neg > 0:
        neg_num = len(data) * args.num_neg
        neg_labels = pd.Series([0] * neg_num, dtype=str)
        labels = pd.concat([labels, neg_labels], ignore_index=True)
    return labels


def transform_csv(args: Arguments):
    """
    libcsv format: `label feat1:val1 feat2:val2 feat3:val3 ...`
    libffm format: `label field1:feat1:val1 field2:feat2:val2 field3:feat3:val3 ...`
    """
    np_rng = np.random.default_rng(args.seed)
    train_data, eval_data = read_data(args)
    if args.normalize and args.num_cols:
        train_data, eval_data = normalize_data(train_data, eval_data, args.num_cols)

    # 0 is used as oov index
    offset = 1
    total_cols = args.cat_cols + args.num_cols
    train_transformed: pd.Series = convert_label_str(train_data, args)
    eval_transformed: pd.Series = convert_label_str(eval_data, args)
    for field, col in enumerate(total_cols):
        if col in args.cat_cols:
            train_vals, eval_vals = train_data.iloc[:, col], eval_data.iloc[:, col]
            val_idx_map = get_unique_mapping(train_vals, offset)
            train_feat_idx = train_vals.map(val_idx_map).to_numpy()
            eval_feat_idx = eval_vals.map(val_idx_map).fillna(0).to_numpy().astype(int)
            if args.neg_sampling and args.num_neg > 0:
                unique_indices = list(val_idx_map.values())
                train_feat_idx = categorical_neg_sampling(
                    np_rng, train_feat_idx, unique_indices, args.num_neg
                )
                eval_feat_idx = categorical_neg_sampling(
                    np_rng, eval_feat_idx, unique_indices, args.num_neg
                )
            train_feat_str = convert_categorical_str(train_feat_idx, field, args.ffm)
            eval_feat_str = convert_categorical_str(eval_feat_idx, field, args.ffm)
            offset += len(val_idx_map)
        else:
            train_feat_val = train_data.iloc[:, col].to_numpy()
            eval_feat_val = eval_data.iloc[:, col].to_numpy()
            if args.neg_sampling and args.num_neg > 0:
                train_feat_val = numerical_neg_sampling(
                    np_rng, train_feat_val, args.num_neg
                )
                eval_feat_val = numerical_neg_sampling(
                    np_rng, eval_feat_val, args.num_neg
                )
            train_feat_str = convert_numerical_str(train_feat_val, offset, field, args.ffm)
            eval_feat_str = convert_numerical_str(eval_feat_val, offset, field, args.ffm)
            offset += 1

        train_transformed = train_transformed.str.cat(train_feat_str, sep=" ")
        eval_transformed = eval_transformed.str.cat(eval_feat_str, sep=" ")
    return train_transformed, eval_transformed


if __name__ == "__main__":
    arguments = Arguments(**vars(parse_args()))
    print(f"\n========== Parsed Arguments: ============\n")
    for arg in fields(arguments):
        print(f"{arg.name}: {getattr(arguments, arg.name)}")
    print()
    if not arguments.neg_sampling or arguments.num_neg <= 0:
        print("Performing normal mode...")
    else:
        print("Performing negative sampling mode...")
    print()

    start_time = time.perf_counter()
    train_dataset, eval_dataset = transform_csv(arguments)
    train_dataset.sample(frac=1.0, random_state=arguments.seed).to_csv(
        arguments.train_output_path, index=False, header=False
    )
    eval_dataset.to_csv(arguments.eval_output_path, index=False, header=False)
    print("Output train size: ", len(train_dataset))
    print("Output eval size: ", len(eval_dataset))
    print(f"Total running time: {(time.perf_counter() - start_time):.2f}")
