import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics


if __name__ == "__main__":
    res = pd.read_csv(sys.argv[1], header=None, names=["label", "prob"], sep=" ")
    res["pred"] = res["prob"].apply(lambda x: 1 if x >=0.5 else -1)
    print("f1 score: %.4f" % f1_score(res.label, res.pred))
    print("accuracy: %.4f" % accuracy_score(res.label, res.pred))
    print("roc auc: %.4f" % roc_auc_score(res.label, res.prob))
    precision, recall, _ = precision_recall_curve(res.label, res.prob)
    print("pr suc: %.4f" % metrics.auc(recall, precision))
