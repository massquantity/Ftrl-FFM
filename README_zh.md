# Ftrl-LR

### [`English`](<https://github.com/massquantity/Ftrl-LR>)  &nbsp;  <font size=5>`简体中文`</font>

<br>

多线程 FTRL 训练 logistic regression 模型，用于解决二分类问题。关于 FTRL 的原理和实现细节见 ...

下面是 FTRL 的伪代码：

![](<https://raw.githubusercontent.com/massquantity/Ftrl-LR/master/pic/002.png>)




## 编译

```shell
git clone git@github.com:massquantity/Ftrl-LR.git
cd Ftrl-LR
cmake .
make
```

Cmake 版本 >= 3.5.1。输出的二进制文件位于 `bin/` 文件夹。



## 使用

1.  **支持标准输入：**

```shell
cat train_data.csv | ./lr_train -m lr_model.txt -cmd true
```

2. **使用二进制文件：**

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
           
./lr_pred -m lr_model.txt \
             -data test_data.csv \
             -o result.txt \
             -nthreads 4 \
             -cmd false 
```

**`lr_train` 的参数:**

+ `-m` : 输出模型路径。
+ `-train_data` : 训练集路径。
+ `-eval_data` : 评估集路径。
+ `-init_mean` (default 0.0) : 参数初始化均值。
+ `-init_stdev` (default 0.01) : 参数初始化标准差。
+ `-w_alpha` (default 0.05) : 学习率的参数之一。
+ `-w_beta` (default 1.0) : 学习率的参数之一。
+ `-w_l1` (default 0.1) : w 的 L1 正则参数。
+ `-w_l2` (default 5.0) : w 的 L2 正则参数。
+ `-nthreads` (default 1) : 线程数。
+ `-epoch` (default 1) : FTRL 训练轮数。
+ `-cmd` (default false) : 是否标准输入数据。



**`lr_pred`的参数 :**

+ `-m` : 存储的模型路径。
+ `-data` : 输入数据路径。
+ `-o` : 输出结果路径。
+ `-nthreads` (default 1) : 线程数。
+ `-cmd` (default false) : 是否标准输入数据。



<br>

另外，脚本 `/python/metrics.py` 提供了其他用于评估模型的指标，如 `F1`、`ROC AUC`、`PR AUC` 等。要使用需要先安装 [`Numpy`](https://numpy.org/),  [`Pandas`](https://pandas.pydata.org/) 和 [`Scikit-Learn`](<https://scikit-learn.org/>)  。

```shell
python metrics.py result.txt
```



## 数据格式

该模型主要用于处理高维稀疏数据，所以为了节省内存，仅支持类 `libsvm` 格式的数据。在 `/data/sample_data.txt` 中提供了一个小型数据样例。

由于现实中现成的 `libsvm` 格式数据较少，仓库中提供了一个 python 脚本 (`/python/dataset.py`) 用于将常见的 `csv` 格式转换为 `libsbvm` 格式。考虑到许多类别型特征经过 one-hot 后维数太大难以存储，我做了一些特殊处理，避免了直接对数据 one-hot 转换。另外对于只含有正反馈的数据集，也可以进行随机生成负样本。

使用方法和参数如下，需要先安装[`Numpy`](https://numpy.org/) 和 [`Pandas`](https://pandas.pydata.org/) : 

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

+ `--data_path` :  单个数据的文件路径，可通过脚本将其拆分为训练/测试集。只能选一个模式：`-data_path` 模式 (单个文件) 或 `--train_path, --test_path` 模式 (同时提供训练和测试集) 。
+ `--train_path` : 训练集文件路径，在此模式下，必须同时提供训练和测试集。
+ `--test_path` : 测试集文件路径，在此模式下，必须同时提供训练和测试集。
+ `--train_output_path` : 转换后的训练集存放路径。
+ `--test_output_path` : 转换后的测试集存放路径。
+ `--train_frac` (default 0.8) : 拆分出训练集的比例。
+ `--threshold` (default 0) : 将标签转换为 1 和 0 的阈值。大于 threshold 的变为 1，其余变为 0。
+ `--sep` (default ',') :  分隔符。
+ `--label_col` (default 0) : 标签列索引。
+ `--cat_cols` : 离散型特征列的索引，以字符串的形式，中间没有空格，如 1,2,3,5,7
+ `--num_cols` : 连续型特征列的索引，以字符串的形式，中间没有空格，如 2,5,8,11,15.
+ `--neg` (default False) : 是否进行负采样。
+ `--num_neg` (default 1) : 每个样本产生的负样本个数。
+ `--normalize` (default False) : 是否标准化处理连续性特征。




## License

**MIT**

<br>