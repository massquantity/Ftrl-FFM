import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import time
import pandas as pd
from utils import (
    str2bool,
    read_data,
    filter_data,
    normalize_data,
    index_data,
    pos_gen_data,
    neg_gen_data
)


def parse_args():
    parser = argparse.ArgumentParser(description="transform_data")
    parser.add_argument("--data_path", type=str, default="None",
                        help="split into train/test data using a single dataset")
    parser.add_argument("--train_path", type=str, default="None",
                        help="file path for train data")
    parser.add_argument("--test_path", type=str, default="None",
                        help="file path for test data")
    parser.add_argument("--train_output_path", default="None", type=str,
                        help="file path for saving transformed train data")
    parser.add_argument("--test_output_path", default="None", type=str,
                        help="file path for saving transformed test data")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="train fraction when spliting data")
    parser.add_argument("--threshold", type=int, default=0,
                        help="threshold for converting labels into 1 and 0")
    parser.add_argument("--num_neg", type=int, default=1,
                        help="number of negative samples generated per sample")
    parser.add_argument("--sep", type=str, default=",",
                        help="delimiter in one sample")
    parser.add_argument("--label_col", type=int, default=0,
                        help="label column index")
    parser.add_argument("--cat_cols", type=str, default="None",
                        help="categorical column indices in string format, e.g., 1,2,3,5,7")
    parser.add_argument("--num_cols", type=str, default="None",
                        help="numerical column indices in string format, e.g., 2,5,8,11,15")
    parser.add_argument("--neg", type=str2bool, nargs="?", const=True, default=False,
                        help="whether to use negative sampling")
    # parser.add_argument("--implicit", type=str2bool, nargs="?", const=True, default=False,
    #                    help="whether to convert all labels to 1")
    parser.add_argument("--normalize", type=str2bool, nargs="?", const=False, default=False,
                        help="whether to normalize numerical features")
    parser.add_argument("--ffm", type=str2bool, nargs="?", const=True, default=True,
                        help="whether to convert to libffm data format")
    return parser.parse_args()


def transform_csv(
    data_path=None,
    train_path=None,
    test_path=None,
    train_output_path=None,
    test_output_path=None,
    header="infer",
    train_frac=0.8,
    implicit_threshold=0,
    sep=",",
    label_col=0,
    cat_cols=None,
    num_cols=None,
    normalize=False,
    num_neg=None,
    ffm=True,
    seed=2020
):
    neg_sample = True if num_neg is not None and num_neg > 0 else False
    cat_cols = (
        list(map(int, cat_cols.split(',')))
        if cat_cols is not None
        else list()
    )
    num_cols = (
        list(map(int, num_cols.split(',')))
        if num_cols is not None
        else list()
    )

    train_data, test_data = read_data(
        data_path, train_path, test_path, sep, header, label_col, train_frac,
        seed, implicit_threshold, neg_sample
    )

    if normalize and num_cols:
        train_data, test_data = normalize_data(train_data, test_data, num_cols)

    train_data, test_data = filter_data(train_data, test_data, cat_cols)
    cat_unique_vals, num_unique_vals = index_data(
        train_data, cat_cols, num_cols
    )

    if not neg_sample:
        transformed_train_data = convert_normal(
            train_data, label_col, cat_cols, num_cols,
            cat_unique_vals, num_unique_vals, ffm
        )
        transformed_test_data = convert_normal(
            test_data, label_col, cat_cols, num_cols,
            cat_unique_vals, num_unique_vals, ffm
        )
    else:
        transformed_train_data = convert_neg(
            train_data, label_col, cat_cols, num_cols, cat_unique_vals,
            num_unique_vals, num_neg, ffm, train=True
        )
        transformed_test_data = convert_neg(
            test_data, label_col, cat_cols, num_cols, cat_unique_vals,
            num_unique_vals, num_neg, ffm, train=False
        )

    pd.Series(transformed_train_data).to_csv(
        train_output_path, index=False, header=False
    )
    pd.Series(transformed_test_data).to_csv(
        test_output_path, index=False, header=False
    )


def convert_normal(
    data,
    label_col,
    cat_cols,
    num_cols,
    cat_vals,
    num_vals,
    ffm
):
    transformed_data = []
    for i, line in enumerate(data):
        if i > 0 and i % 100000 == 0:
            print("%d positive samples finished" % i)
        sample = pos_gen_data(line, label_col, cat_cols, num_cols, cat_vals,
                              num_vals, ffm)
        transformed_data.append(sample)
    return transformed_data


def convert_neg(
    data,
    label_col,
    cat_cols,
    num_cols,
    cat_vals,
    num_vals,
    num_neg,
    ffm,
    train=True
):
    transformed_data = []
    pos_count = len(data)
    neg_count = num_neg * len(data)
    pos_prob = 1 / (num_neg + 1)
    pos = 0
    neg = 0
    t0 = time.perf_counter()
    while True:
        dice = random.random()
        if dice <= pos_prob and pos < pos_count:
            if pos > 0 and pos % 100000 == 0:
                print("%d positive samples finished" % pos)
            line = data[pos]
            pos += 1
            sample = pos_gen_data(line, label_col, cat_cols, num_cols,
                                  cat_vals, num_vals, ffm)
            transformed_data.append(sample)
        if (dice > pos_prob or pos == pos_count) and neg < neg_count:
            neg += 1
            sample = neg_gen_data(cat_cols, num_cols, cat_vals, num_vals, ffm)
            transformed_data.append(sample)
        if pos == pos_count and neg == neg_count:
            if train:
                print("final train size: ", pos_count + neg_count)
            else:
                print("final test size: ", pos_count + neg_count)
            break
    print(f"time: {time.perf_counter() - t0}")
    return transformed_data


if __name__ == "__main__":
    args = vars(parse_args())
    for k in args:
        if args[k] == "None":
            args[k] = None

    if not args["neg"]:
        args["num_neg"] = None
        print("chose normal mode...")
    else:
        assert args["neg"] is not None and args["neg"] > 0
        print("chose negative sampling mode...")

    transform_csv(data_path=args["data_path"],
                  train_path=args["train_path"],
                  test_path=args["test_path"],
                  train_output_path=args["train_output_path"],
                  test_output_path=args["test_output_path"],
                  implicit_threshold=args["threshold"],
                  sep=args["sep"],
                  train_frac=args["train_frac"],
                  label_col=args["label_col"],
                  cat_cols=args["cat_cols"],
                  num_cols=args["num_cols"],
                  normalize=args["normalize"],
                  num_neg=args["num_neg"],
                  ffm=args["ffm"])


'''
python transform_data.py 
    --data_path ~/Workspace/Ftrl-FFM-orig/merged_data.csv \
    --train_output_path ~/Workspace/Ftrl-FFM-orig/train-ml.txt \
    --test_output_path ~/Workspace/Ftrl-FFM-orig/test-ml.txt \
    --threshold 3 \
    --sep , \
    --train_frac 0.8 \
    --label_col 2 \
    --cat_cols 0,1,3,5,6,7,8 \
    --num_cols 4 \
    --normalize false \
    --neg true \
    --num_neg 1 \
    --ffm true
'''
