import warnings
warnings.filterwarnings("ignore")
import time
import random
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from download_data import prepare_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ("yes", "true", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean liked value expected...")

def parse_args():
    parser = argparse.ArgumentParser(description="preprocess_data")
    parser.add_argument("--data_path", type=str, default="None", help="split into train/test data using a single dataset")
    parser.add_argument("--train_path", type=str, default="None", help="file path for train data")
    parser.add_argument("--test_path", type=str, default="None", help="file path for test data")
    parser.add_argument("--train_output_path", default="None", type=str, help="file path for saving transformed train data")
    parser.add_argument("--test_output_path", default="None", type=str, help="file path for saving transformed test data")
    parser.add_argument("--train_frac", type=float, default=0.8, help="train fraction when spliting data")
    parser.add_argument("--threshold", type=int, default=0, help="threshold for converting labels into 1 and 0")
    parser.add_argument("--num_neg", type=int, default=1, help="number of negative samples generated per sample")
    parser.add_argument("--sep", type=str, default=",", help="delimiter in one sample")
    parser.add_argument("--label_col", type=int, default=0, help="label column index")
    parser.add_argument("--cat_cols", type=str, default="None", 
        help="categorical column indices in string format, e.g., 1,2,3,5,7")
    parser.add_argument("--num_cols", type=str, default="None", 
        help="numerical column indices in string format, e.g., 2,5,8,11,15")
    parser.add_argument("--neg", type=str2bool, nargs="?", const=True, default=False, 
        help="whether to use negative sampling")
#    parser.add_argument("--implicit", type=str2bool, nargs="?", const=True, default=False, 
#        help="whether to convert all labels to 1")
    parser.add_argument("--normalize", type=str2bool, nargs="?", const=False, default=False, 
        help="whether to normalize numerical features")
    return parser.parse_args()

def pos_sample(data_iter, label_col, total_cols, cat_cols, num_cols, cat_unique_vals, num_unique_vals):
    line = next(data_iter)
    label = line[label_col]
    sample = list(str(label))
    for col in sorted(total_cols):
        val = line[col]
        if col in cat_cols:
            ind_val_pair = "%s:%s" % (cat_unique_vals[col][val], 1)
            sample.append(ind_val_pair)
        elif col in num_cols:
            ind_val_pair = "%s:%s" % (num_unique_vals[col][0], val)
            sample.append(ind_val_pair)
    sample = " ".join(sample)
    sample += "\n"
    return sample

def neg_sample(total_cols, cat_cols, num_cols, cat_unique_vals_neg, num_unique_vals):
    sample = list(str(0))
    for col in sorted(total_cols):
        if col in cat_cols:
    #        val = random.choice(list(cat_unique_vals_neg[col].keys()))  # random.choice too slow
            val = np.random.randint(0, len(cat_unique_vals_neg[col].keys()))  # sample from reindexed neg, see line 263
            ind_val_pair = "%s:%s" % (cat_unique_vals_neg[col][val], 1)
            sample.append(ind_val_pair)
        elif col in num_cols and np.issubdtype(type(num_unique_vals[col][1]), np.integer):
    #    elif col in num_cols and isinstance(num_unique_vals[col][1], int):
            # sample between min and max value of this column
            min_value = num_unique_vals[col][1]
            max_value = num_unique_vals[col][2]
            val = np.random.randint(min_value, max_value)
            ind_val_pair = "%s:%s" % (num_unique_vals[col][0], val)
            sample.append(ind_val_pair)
        elif col in num_cols and np.issubdtype(type(num_unique_vals[col][1]), np.floating):
    #    elif col in num_cols and isinstance(num_unique_vals[col][1], float):
            min_value = num_unique_vals[col][1]
            max_value = num_unique_vals[col][2]
            val = (max_value - min_value) * np.random.random_sample() + min_value
            ind_val_pair = "%s:%s" % (num_unique_vals[col][0], val)
            sample.append(ind_val_pair)
    sample = " ".join(sample)
    sample += "\n"
    return sample

def csv_to_libsvm(data_path=None, 
                  train_path=None, 
                  test_path=None, 
                  train_output_path=None, 
                  test_output_path=None, 
                  header="infer",
                  train_frac=0.8, 
                  implicit_threshold=None, 
                  sep=",", 
                  label_col=0, 
                  cate_cols=None, 
                  nume_cols=None,
                  normalize=False, 
                  seed=2020):
    np.random.seed(seed)
    if data_path is not None:
        loaded_data = pd.read_csv(data_path, sep=sep, header=header)
        if implicit_threshold is not None:
            loaded_data.iloc[:, label_col] = loaded_data.iloc[:, label_col].apply(
                lambda x: 1 if x > implicit_threshold else 0)
        train_data, test_data = train_test_split(loaded_data, train_size=train_frac, random_state=seed)
        train_data = train_data.to_numpy()
        test_data = test_data.to_numpy()
    elif train_path is not None and test_path is not None:
        train_data = pd.read_csv(train_path, sep=sep, header=header)
        test_data = pd.read_csv(test_path, sep=sep, header=header)
        if implicit_threshold is not None:
            train_data.iloc[:, label_col] = train_data.iloc[:, label_col].apply(
                lambda x: 1 if x > implicit_threshold else 0)
            test_data.iloc[:, label_col] = test_data.iloc[:, label_col].apply(
                lambda x: 1 if x > implicit_threshold else 0)
        train_data = train_data.to_numpy()
        test_data = test_data.to_numpy()
    
    if normalize and nume_cols is not None:
        for col in list(map(int, nume_cols.split(','))):
            scaler = StandardScaler()
            train_data[:, col] = scaler.fit_transform(train_data[:, col].reshape(-1, 1)).flatten()
            test_data[:, col] = scaler.transform(test_data[:, col].reshape(-1, 1)).flatten()

    print("test size before filtering: ", len(test_data))
    out_of_bounds_row_indices = set()
    for col in range(train_data.shape[1]):
        unique_values_set = set(pd.unique(train_data[:, col]))
        for i, t in enumerate(test_data[:, col]):
            if t not in unique_values_set:  # set is mush faster than list for search contains
                out_of_bounds_row_indices.add(i)
    
    # filter test values that are not in train_data
    mask = np.arange(len(test_data))
    test_data = test_data[~np.isin(mask, list(out_of_bounds_row_indices))]
    print("test size after filtering: ", len(test_data))

    # one-hot and extract indices for each column
    cat_cols = list(map(int, cate_cols.split(','))) if cate_cols is not None else list()
    num_cols = list(map(int, nume_cols.split(','))) if nume_cols is not None else list()

    total_count = 0
    total_cols = cat_cols + num_cols
    cat_unique_vals = defaultdict(dict)  # format:  {col: {val: index}}
    num_unique_vals = defaultdict(list)  # format:  {col: [index, min, max]}
    for col in sorted(total_cols):
        if col in cat_cols:
            unique_vals, indices = np.unique(train_data[:, col], return_inverse=True)
            unique_vals_length = len(unique_vals)
            indices += total_count
            cat_unique_vals[col].update(zip(unique_vals, np.unique(indices)))
            total_count += unique_vals_length
        elif col in num_cols:
            col_min, col_max = min(train_data[:, col]), max(train_data[:, col])
            num_unique_vals[col].extend([total_count, col_min, col_max])
            total_count += 1

    # write data to file
    with open(train_output_path, 'wb') as f:
        for i, line in enumerate(train_data):
            if i % 100000 == 0: 
                    print("%d positive samples finished" % i)
            sample = list()
            label = line[label_col]
            sample.append(str(label))
            for col in sorted(total_cols):
                val = line[col]
                if col in cat_cols:
                    ind_val_pair = "%s:%s" % (cat_unique_vals[col][val], 1)
                    sample.append(ind_val_pair)
                elif col in num_cols:
                    ind_val_pair = "%s:%s" % (num_unique_vals[col][0], val)
                    sample.append(ind_val_pair)
            sample = " ".join(sample)
            sample += "\n"
            f.write(bytes(sample, encoding="utf8"))

    with open(test_output_path, 'wb') as f:
        for line in test_data:
            sample = list()
            label = line[label_col]
            sample.append(str(label))
            for col in sorted(total_cols):
                val = line[col]
                if col in cat_cols:
                    ind_val_pair = "%s:%s" % (cat_unique_vals[col][val], 1)
                    sample.append(ind_val_pair)
                elif col in num_cols:
                    ind_val_pair = "%s:%s" % (num_unique_vals[col][0], val)
                    sample.append(ind_val_pair)
            sample = " ".join(sample)
            sample += "\n"
            f.write(str.encode(sample))
        
def csv_to_libsvm_neg(data_path=None, 
                      train_path=None, 
                      test_path=None, 
                      train_output_path=None, 
                      test_output_path=None, 
                      header="infer",
                      train_frac=0.8, 
                      num_neg=None, 
                      sep=",", 
                      label_col=0, 
                      cate_cols=None, 
                      nume_cols=None,
                      normalize=False, 
                      seed=2020):
#    np.random.seed(seed)
    if data_path is not None:
        loaded_data = pd.read_csv(data_path, sep=sep, header=header)
        loaded_data.iloc[:, label_col] = 1  # for implicit data, convert all labels to 1
        train_data, test_data = train_test_split(loaded_data, train_size=train_frac, random_state=seed)
        train_data = train_data.to_numpy()  # convert to dataframe to numpy array
        test_data = test_data.to_numpy()
    elif train_path is not None and test_path is not None:
        train_data = pd.read_csv(train_path, sep=sep, header=header)
        test_data = pd.read_csv(test_path, sep=sep, header=header)
        train_data.iloc[:, label_col] = 1
        test_data.iloc[:, label_col] = 1
        train_data = train_data.to_numpy()
        test_data = test_data.to_numpy()
    
    if normalize and nume_cols is not None:
        for col in list(map(int, nume_cols.split(','))):
            scaler = StandardScaler()
            train_data[:, col] = scaler.fit_transform(train_data[:, col].reshape(-1, 1)).flatten()
            test_data[:, col] = scaler.transform(test_data[:, col].reshape(-1, 1)).flatten()

    print("test size before filtering: ", len(test_data))
    out_of_bounds_row_indices = set()
    for col in range(train_data.shape[1]):
        unique_values_set = set(pd.unique(train_data[:, col]))
        for i, t in enumerate(test_data[:, col]):
            if t not in unique_values_set:  # set is mush faster than list for search contains
                out_of_bounds_row_indices.add(i)
    
    # filter test feature values that are not in train_data
    mask = np.arange(len(test_data))
    test_data = test_data[~np.isin(mask, list(out_of_bounds_row_indices))]
    print("test size after filtering: ", len(test_data))

    # one-hot and extract indices for each column
    cat_cols = list(map(int, cate_cols.split(','))) if cate_cols is not None else list()
    num_cols = list(map(int, nume_cols.split(','))) if nume_cols is not None else list()

    total_count = 0
    total_cols = cat_cols + num_cols
    cat_unique_vals = defaultdict(dict)  # format:  {col: {val: index}}
    cat_unique_vals_neg = defaultdict(dict)
    num_unique_vals = defaultdict(list)  # format:  {col: [index, min, max]}
    for col in sorted(total_cols):
        if col in cat_cols:
            unique_vals, indices = np.unique(train_data[:, col], return_inverse=True)
            unique_vals_length = len(unique_vals)
            indices += total_count
            cat_unique_vals[col].update(zip(unique_vals, np.unique(indices)))
            unique_vals_neg = range(len(unique_vals))
            cat_unique_vals_neg[col].update(zip(unique_vals_neg, np.unique(indices)))  # to make sampling faster, see line 69
            total_count += unique_vals_length
        elif col in num_cols:
            # may need to convert numpy data types to python data types
            col_min, col_max = min(train_data[:, col]), max(train_data[:, col])  
            num_unique_vals[col].extend([total_count, col_min, col_max])
            total_count += 1

    # random negative sampling and write data to file
    # spread pos and neg samples according to overall ratios
    pos_count = len(train_data)
    neg_count = num_neg * len(train_data)
    pos_prob = 1 / (num_neg + 1)
    train_data_iter = iter(train_data)
    pos = 0
    neg = 0
    with open(train_output_path, 'wb') as f:
        while True:
            dice = np.random.random()
            if dice <= pos_prob and pos < pos_count:
                if pos % 100000 == 0: 
                    print("%d positive samples finished" % pos)
                pos += 1
                sample = pos_sample(train_data_iter, label_col, total_cols, cat_cols, 
                                    num_cols, cat_unique_vals, num_unique_vals)
                f.write(bytes(sample, encoding="utf8"))
            if dice > pos_prob and neg < neg_count:
                neg += 1
                sample = neg_sample(total_cols, cat_cols, num_cols, cat_unique_vals_neg, num_unique_vals)
                f.write(bytes(sample, encoding="utf8"))
            if pos == pos_count and neg < neg_count:
                neg += 1
                sample = neg_sample(total_cols, cat_cols, num_cols, cat_unique_vals_neg, num_unique_vals)
                f.write(bytes(sample, encoding="utf8"))
            if pos == pos_count and neg == neg_count:
                print("final train size: ", pos_count + neg_count)
                break

    pos_count = len(test_data)
    neg_count = num_neg * len(test_data)
    pos_prob = 1 / (num_neg + 1)
    test_data_iter = iter(test_data)
    pos = 0
    neg = 0
    with open(test_output_path, 'wb') as f:
        while True:
            dice = np.random.random()
            if dice < pos_prob and pos < pos_count:
                pos += 1
                sample = pos_sample(test_data_iter, label_col, total_cols, cat_cols, 
                                    num_cols, cat_unique_vals, num_unique_vals)
                f.write(str.encode(sample))
            if dice >= pos_prob and neg < neg_count:
                neg += 1
                sample = neg_sample(total_cols, cat_cols, num_cols, cat_unique_vals_neg, num_unique_vals)
                f.write(str.encode(sample))
            if pos == pos_count and neg < neg_count:
                neg += 1
                sample = neg_sample(total_cols, cat_cols, num_cols, cat_unique_vals_neg, num_unique_vals)
                f.write(str.encode(sample))
            if pos == pos_count and neg == neg_count:
                print("final test size: ", pos_count + neg_count)
                break


if __name__ == "__main__":
    args = vars(parse_args())
    for k in args:
        if args[k] == "None":
            args[k] = None

    if not args["neg"]:
        print("choose normal mode...")
        csv_to_libsvm(data_path=args["data_path"], 
                      train_path=args["train_path"], 
                      test_path=args["test_path"], 
                      train_output_path=args["train_output_path"], 
                      test_output_path=args["test_output_path"], 
                      implicit_threshold=args["threshold"], 
                      sep=args["sep"], 
                      train_frac=args["train_frac"], 
                      label_col=args["label_col"], 
                      cate_cols=args["cat_cols"], 
                      nume_cols=args["num_cols"],
                      normalize=args["normalize"])
    else:
        print("choose negative sampling mode...")
        csv_to_libsvm_neg(data_path=args["data_path"], 
                      train_path=args["train_path"], 
                      test_path=args["test_path"], 
                      train_output_path=args["train_output_path"], 
                      test_output_path=args["test_output_path"], 
                      sep=args["sep"], 
                      train_frac=args["train_frac"], 
                      label_col=args["label_col"], 
                      cate_cols=args["cat_cols"], 
                      nume_cols=args["num_cols"],
                      normalize=args["normalize"], 
                      num_neg=args["num_neg"])


'''
csv_to_libsvm(data_path="./merged_data.csv", 
                train_output_path="./train-ml.txt", 
                test_output_path="./test-ml.txt", 
                implicit_threshold=3, 
                label_col=2, 
                cate_cols="0,1,3,5,6,7,8", 
                nume_cols="4",
                normalize=False)
python dataset.py --data_path merged_data.csv \
    --train_output_path train-ml.txt \
    --test_output_path test-ml.txt \
    --threshold 3 \
    --label_col 2 \
    --cat_cols 0,1,3,5,6,7,8 \
    --num_cols 4 \
    --normalize false \
    --neg true \
    --num_neg 1
'''