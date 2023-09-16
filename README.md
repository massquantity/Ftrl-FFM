# Ftrl-FFM

### `English`  &nbsp;  [`简体中文`](https://github.com/massquantity/Ftrl-FFM/blob/main/README_zh.md)

<br>

Using multi-threading version of FTRL to train logistic regression(LR), factorization machines(FM), and field-aware factorization machines(FFM) for binary classification problem. For full theory and implementation details of FTRL, see [blog post](https://www.cnblogs.com/massquantity/p/12693314.html)

Here is the pseudocode of FTRL: 

![](https://s1.ax1x.com/2020/05/12/YtmINn.png)



## Build

+ Cmake >= 3.20
+ g++ >= 7.0 or clang++ >= 5.0, which support C++17 standard.

```shell
$ git clone https://github.com/massquantity/Ftrl-FFM.git
$ cd Ftrl-FFM
# build zstd library first
$ cmake -S third_party/zstd/build/cmake -B third_party/zstd/build_output
$ cmake --build third_party/zstd/build_output -j 8
```

```shell
# build the project
$ mkdir build && cd build
$ cmake ..
$ make -j8
```

```shell
# testing(optional)
$ make test
```

## Usage

The built executable file is `Ftrl-FFM/build/src/main`.

```shell
$ ./src/main \
    --model_path model.pt \
    --train_data train_data.txt \
    --eval_data eval_data.txt \
    --init_mean 0.0 \
    --init_stddev 0.02 \
    --w_alpha 1e-4 \
    --w_beta 1.0 \
    --w_l1 0.1 \
    --w_l2 5.0 \
    --n_threads 2 \
    --n_fields 8 \
    --n_feats 10000 \
    --n_factors 16 \
    --online false \
    --n_epochs 5 \
    --model_type FFM
```

**Arguments :**

+ `--model_path` : the output model path.
+ `--train_data` : train data file path.
+ `--eval_data` : evaluate data file path.
+ `--init_mean` (default 0.0) : mean for parameter initialization.
+ `--init_stdev` (default 0.02) : standard deviation for parameter initialization.
+ `--w_alpha` (default 1e-4) : one of the learning rate parameters.
+ `--w_beta` (default 1.0) : one of the learning rate parameters.
+ `--w_l1` (default 0.1) : L1 regularization parameter of w.
+ `--w_l2` (default 5.0) : L2 regularization parameter of w.
+ `--n_threads` (default 1) : number of threads.
+ `--n_fields` (default 8) : number of fields in FFM.
+ `--n_feats` (default 10000) : number of features.
+ `--n_factors` (default 16) : embedding size.
+ `--n_epochs` (default 1) : number of training epochs.
+ `--model_type` (default FFM): LR, FM or FFM.


## Data Format

The model is primarily designed for high dimensional sparse data, so for saving memory purpose,  only libsvm or libffm data format is supported.  Two example datasets are provided in `/data` folder.

Due to the lack of `libsvm` and `libffm` data format, a python script (`/python/generate_data.py`) is provided to transform common data format (e.g. `csv`) to `libsvm` or `libffm`  format. Categorical features are converted into sparse reprensentation. Besides, for dataset only contains positive feedback, the script can also be used to generate random negative samples.

Main usage and arguments are as follows, [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html) are required : 

```shell
$ python generate_data.py \
    --data_path data.csv \
    --train_output_path train-ml.txt \
    --eval_output_path eval-ml.txt \
    --threshold 0 \
    --train_frac 0.8 \
    --label_col 0 \
    --cat_cols 0,1,3,5,8 \
    --num_cols 4,6,7 \
    --normalize true \
    --neg_sampling true \
    --num_neg 1 \
    --normalize true \
    --ffm true
```

+ `--data_path` :  single data file path, can be split into train/test data through the script. You must choose either `--data_path` mode (single data file) or `--train_path, --eval_path` mode (train +eval data files).
+ `--train_path` : train data file path, in this mode, both train and eval data must be provided. 
+ `--eval_path` : eval data file path, in this mode, both train and eval data must be provided. 
+ `--train_output_path` : file path for saving transformed train data.
+ `--eval_output_path` : file path for saving transformed eval data.
+ `--train_frac` (default 0.8) : train set proportion when splitting data.
+ `--threshold` (default 0) : threshold for converting labels into 1 and 0. Labels larger than threshold will be converted to 1, and the rest will be 0.
+ `--sep` (default ',') :  delimiter in one sample.
+ `--label_col` (default 0) : label column index.
+ `--cat_cols` : categorical column indices in string format, no spaces, e.g., 1,2,3,5,7
+ `--num_cols` : numerical column indices in string format, no spaces, e.g., 2,5,8,11,15
+ `--neg_sampling` (default False) : whether to use negative sampling.
+ `--num_neg` (default 1) : number of negative samples generated for each sample.
+ `--normalize` (default False) : whether to normalize numerical features.
+ `--ffm` (default True): whether to convert to `libffm` format, otherwise data will be converted to `libcsv` format.



## License

**MIT**
