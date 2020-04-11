# Ftrl-LR

### <font size=5>English</font>  &nbsp;  [<font size=5>简体中文</font>](https://github.com/massquantity/Ftrl-LR/blob/master/README_zh.md)

<br>

Using multi-threading version of Ftrl to train logistic regression model for binary classification problem. For full theory and implementation details of Ftrl, see article...


## Data Format

The model is primarily designed for high dimensional sparse data, so for saving memory purpose,  only *libsvm-like* data format is supported.  A tiny sample dataset is provided in `/data/sample_data.txt` folder.

Due to lack of this kind of data format, a python script (`/python/dataset.py`) is provided to transform normal data format to libsvm format. For dataset only contains positive feedback, the script can be used to generate random negative samples.

Main usage and arguments are as follows, [`Numpy`](https://numpy.org/) and [`Pandas`](https://pandas.pydata.org/) are required : 

```shell
# single dataset without negative sampling
python dataset.py --data_path data.csv \ 
                  --train_output_path train-ml.txt \
                  --test_output_path test-ml.txt \
                  --threshold 0 \
                  --label_col 0 \
                  --cat_cols 0,1,3,5,8 \
                  --num_cols 4,6,7 \
                  --normalize false \
                  --neg false
    
# train and test dataset with negative sampling
python dataset.py --train_path train.csv \ 
                  --test_path  test.csv \
                  --train_output_path train-ml.txt \
                  --test_output_path test-ml.txt \
                  --threshold 0 \
                  --label_col 0 \
                  --cat_cols 0,1,3,5,8 \
                  --num_cols 4,6,7 \
                  --normalize false \
                  --neg true \
                  --num_neg 2
```

+ `--data_path` :  single data file path, can be split into train/test data through the script. You must choose either `--data_path` mode (single data file) or `--train_path, --test_path` mode (train + test data files).
+ `--train_path` : train data file path, in this mode, both train and test data must be provided. 
+ `--test_path` : test data file path, in this mode, both train and test data must be provided. 
+ `--train_output_path` : file path for saving transformed train data.
+ `--test_output_path` : file path for saving transformed test data.
+ `--train_frac` (default 0.8) : train set proportion when splitting data.
+ `--threshold` (default 0) : threshold for converting labels into 1 and 0. Labels larger than threshold will be converted to 1, and the rest will be 0.
+ `--sep` (default ',') :  delimiter in one sample.
+ `--label_col` (default 0) : label column index.
+ `--cat_cols` : categorical column indices in string format, no spaces, e.g., 1,2,3,5,7
+ `--num_cols` : numerical column indices in string format, no spaces, e.g., 2,5,8,11,15
+ `--neg` (default False) : whether to use negative sampling.
+ `--num_neg` (default 1) : number of negative samples generated per sample.
+ `--normalize` (default False) : whether to normalize numerical features.



## Build

```shell
cmake .
make
```

Cmake version >= 3.5.1. Output binaries will be under the `bin/` folder.



## Usage

1.  **support standard input :**

```shell
cat train_data.csv | ./lr_train -m lr_model.txt -cmd true
```

2. **use binary file :**

```shell
./lr_train -m lr_model.txt \ 
           -train_data train_data.csv \
           -eval_data eval_data.csv \
           -init_mean 0.0 \
           -init_stddev 0.01 \
           -w_alpha 0.05 \
           -w_beta 1.0 \
           -w_l1 0.1 \
           -w_l2 5.0 \
           -nthreads 4 \
           -epoch 10 \
           -cmd false 
           
./lr_predict -m lr_model.txt \ 
             -data test_data.csv \
             -o result.txt \
             -nthreads 4 \
             -cmd false 
```

**Arguments for `lr_train` :**

+ `-m` : the output model path.
+ `-train_data` : train data file path.
+ `-eval_data` : evaluate data file path.
+ `-init_mean` (default 0.0) : mean for parameter initialization.
+ `-init_stdev` (default 0.01) : standard deviation for parameter initialization.
+ `-w_alpha` (default 0.05) : one of the learning rate parameters.
+ `-w_beta` (default 1.0) : one of the learning rate parameters.
+ `-w_l1` (default 0.1) : L1 regularization parameter of w.
+ `-w_l2` (default 5.0) : L2 regularization parameter of w.
+ `-nthreads` (default 1) : number of threads.
+ `-epoch` (default 1) : number of iterations of FTRL.
+ `-cmd` (default false) : whether to input data from standard input.



**Arguments for `lr_predict` :**

+ `-m` : the saved model path.
+ `-data` : the input data path.
+ `-o` : the output result path.
+ `-nthreads` (default 1) : number of threads.
+ `-cmd` (default false) : whether to input data from standard input.



<br>

Besides, the script `/python/metrics.py` provides other metrics to evaluate the model, such as `f1`, `ROC AUC`, `PR AUC` . [`Numpy`](https://numpy.org/),  [`Pandas`](https://pandas.pydata.org/) and [`Scikit-Learn`](<https://scikit-learn.org/>) are required.

```shell
python metrics.py result.txt
```



## License

**MIT**

<br>