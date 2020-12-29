import argparse
from collections import defaultdict
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ("yes", "true", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean liked value expected...")


def read_data(data_path, train_path, test_path, sep, header, label_col,
              train_frac, seed, implicit_threshold=0, neg_sample=False):
    if data_path is not None:
        loaded_data = pd.read_csv(data_path, sep=sep, header=header)
        if neg_sample:
            # for implicit data, convert all labels to 1
            loaded_data.iloc[:, label_col] = 1
        else:
            loaded_data.iloc[:, label_col] = (
                loaded_data.iloc[:, label_col].apply(
                    lambda x: 1 if x > implicit_threshold else 0
                )
            )
        train_data, test_data = train_test_split(
            loaded_data, train_size=train_frac, random_state=seed
        )
        train_data = train_data.to_numpy()
        test_data = test_data.to_numpy()
    elif train_path is not None and test_path is not None:
        train_data = pd.read_csv(train_path, sep=sep, header=header)
        test_data = pd.read_csv(test_path, sep=sep, header=header)
        if neg_sample:
            train_data.iloc[:, label_col] = 1
            test_data.iloc[:, label_col] = 1
        else:
            train_data.iloc[:, label_col] = train_data.iloc[:, label_col].apply(
                lambda x: 1 if x > implicit_threshold else 0)
            test_data.iloc[:, label_col] = test_data.iloc[:, label_col].apply(
                lambda x: 1 if x > implicit_threshold else 0)
        train_data = train_data.to_numpy()
        test_data = test_data.to_numpy()
    else:
        raise ValueError("must provide data_path or train_path && test_path")
    return train_data, test_data


def normalize_data(train_data, test_data, num_cols):
    for col in num_cols:
        scaler = StandardScaler()
        # data is numpy format
        train_data[:, col] = scaler.fit_transform(
            train_data[:, col].reshape(-1, 1)).flatten()
        test_data[:, col] = scaler.transform(
            test_data[:, col].reshape(-1, 1)).flatten()
    return train_data, test_data


def filter_data(train_data, test_data, cat_cols):
    print("test size before filtering: ", len(test_data))
    out_of_bounds_row_indices = set()
    for col in cat_cols:
        unique_values_set = set(pd.unique(train_data[:, col]))
        for i, t in enumerate(test_data[:, col]):
            if t not in unique_values_set:
                out_of_bounds_row_indices.add(i)

    # filter test values that are not in train_data
    mask = np.arange(len(test_data))
    test_data = test_data[~np.isin(mask, list(out_of_bounds_row_indices))]
    print("test size after filtering: ", len(test_data))
    return train_data, test_data


def index_data(train_data, cat_cols, num_cols):
    total_count = 0
    total_cols = cat_cols + num_cols
    cat_unique_vals = defaultdict(dict)  # format:  {col: {val: index}}
    num_unique_vals = defaultdict(list)  # format:  {col: [index, min, max]}
    for col in total_cols:
        if col in cat_cols:
            unique_vals, indices = np.unique(
                train_data[:, col], return_inverse=True
            )
            unique_indices = np.unique(indices)
            unique_indices += total_count
            cat_unique_vals[col].update(zip(unique_vals, unique_indices))
            unique_vals_length = len(unique_vals)
            cat_unique_vals[str(col)+"_len"] = unique_vals_length
            cat_unique_vals[str(col)+"_idx"] = list(unique_indices)
            total_count += unique_vals_length
        elif col in num_cols:
            # may need to convert numpy data types to python data types
            col_min, col_max = min(train_data[:, col]), max(train_data[:, col])
            num_unique_vals[col].extend([total_count, col_min, col_max])
            total_count += 1
    return cat_unique_vals, num_unique_vals


def pos_gen_data(data, label_col, cat_cols, num_cols, cat_vals, num_vals, ffm):
    total_cols = cat_cols + num_cols
    label = data[label_col]
    sample = list(str(label))
    for field, col in enumerate(total_cols):
        val = data[col]
        if col in cat_cols:
            idx_val_pair = (
                "{}:{}:{}".format(field, cat_vals[col][val], 1)
                if ffm else "{}:{}".format(cat_vals[col][val], 1)
            )
        elif col in num_cols:
            idx_val_pair = (
                "{}:{}:{}".format(field, num_vals[col][0], val)
                if ffm else "{}:{}".format(num_vals[col][0], val)
            )
        # noinspection PyUnboundLocalVariable
        sample.append(idx_val_pair)
    sample = " ".join(sample)
    return sample


def neg_gen_data(cat_cols, num_cols, cat_vals, num_vals, ffm):
    total_cols = cat_cols + num_cols
    sample = list("0")
    for field, col in enumerate(total_cols):
        if col in cat_cols:
            vals = cat_vals[str(col)+"_idx"]
            n_unique_vals = cat_vals[str(col)+"_len"]
            # i = random.randrange(n_unique_vals)
            i = int(n_unique_vals * random.random())
            idx_val_pair = (
                "{}:{}:{}".format(field, vals[i], 1)
                if ffm else "{}:{}".format(vals[i], 1)
            )
        elif col in num_cols and (
                np.issubdtype(type(num_vals[col][1]), np.integer)
        ):
            min_val = num_vals[col][1]
            max_val = num_vals[col][2]
            val = random.randrange(min_val, max_val + 1)
            idx_val_pair = (
                "{}:{}:{}".format(field, num_vals[col][0], val)
                if ffm else "{}:{}".format(num_vals[col][0], val)
            )
        elif col in num_cols and (
                np.issubdtype(type(num_vals[col][1]), np.floating)
        ):
            min_val = num_vals[col][1]
            max_val = num_vals[col][2]
            val = (max_val - min_val) * random.random() + min_val
            idx_val_pair = (
                "{}:{}:{}".format(field, num_vals[col][0], val)
                if ffm else "{}:{}".format(num_vals[col][0], val)
            )
        # noinspection PyUnboundLocalVariable
        sample.append(idx_val_pair)
    sample = " ".join(sample)
    return sample
