# Ftrl-FFM

### [`English`](<https://github.com/massquantity/Ftrl-FFM>)  &nbsp;  <font size=5>`简体中文`</font>

<br>

多线程 FTRL 训练 logistic regression(LR), factorization machines(FM) 或 field-aware factorization machines(FFM) 模型，用于解决二分类问题。关于 FTRL 的原理和实现细节见 [博客文章](https://www.cnblogs.com/massquantity/p/12693314.html)

下面是 FTRL 的伪代码：

![](https://s1.ax1x.com/2020/05/12/Ytmh7j.png)



## 编译

+ Cmake >= 3.20
+ g++ >= 7.0 或 clang++ >= 5.0, 需要支持 C++17 标准。

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



## 使用

编译完成的二进制文件位于 `Ftrl-FFM/build/src/main` 。

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

**命令行参数 :**

+ `--model_path` : 模型输出路径.
+ `--train_data` : 训练集路径。
+ `--eval_data` : 评估集路径。
+ `--init_mean` (default 0.0) : 参数初始化均值。
+ `--init_stdev` (default 0.02) : 参数初始化标准差。
+ `--w_alpha` (default 1e-4) : 学习率的参数之一。
+ `--w_beta` (default 1.0) : 学习率的参数之一。
+ `--w_l1` (default 0.1) : w 的 L1 正则参数。
+ `--w_l2` (default 5.0) : w 的 L2 正则参数。
+ `--n_threads` (default 1) : 线程数。
+ `--n_fields` (default 8) : FFM 中的 field 数目。
+ `--n_feats` (default 10000) : 总特征数。
+ `--n_factors` (default 16) : embedding 大小。
+ `--n_epochs` (default 1) : FTRL 训练轮数。
+ `--model_type` (default FFM): 模型类别，LR，FM 或 FFM。 



## 数据格式

模型主要用于处理高维稀疏数据，为了节省内存仅支持 libsvm 或 libffm 格式的数据。在 `/data` 中提供了样例数据。

由于现实中现成的 `libsvm` 和 `libffm` 格式数据较少，仓库中提供了一个 python 脚本 (`/python/generate_data.py`) 用于将常见的 `csv` 格式转换为 `libsbvm` 或 `libffm` 格式。类别型特征会被转化为稀疏(sparse)表示。另外对于只含有正反馈的数据集，也可以进行随机生成负样本。

使用方法和参数如下，需要先安装 [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html) :

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
+ `--num_cols` : 连续型特征列的索引，以字符串的形式，中间没有空格，如 2,5,8,11,15
+ `--neg_sampling` (default False) : 是否进行负采样。
+ `--num_neg` (default 1) : 每个样本产生的负样本个数。
+ `--normalize` (default False) : 是否标准化处理连续性特征。
+ `--ffm` (default True): 是否转化为 `libffm` 格式, 若否转换为 `libcsv` 格式。




## License

**MIT**

