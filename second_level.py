"""
This basically deals with finding the optimal way to use
our different models to predict a score.
Here the data should not be changed anymore.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import glob
from sklearn.cross_validation import KFold
from datetime import datetime

LOCAL = True

xgboost_params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "logloss",
    "eta": 0.03,
    "base_score": 0.761,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 10,
    "min_child_weight": 0.75,
    }


def load_files(suffix="train"):
    """
    Loads all the first-level features in a dataframe.
    """
    df = pd.DataFrame()
    for csv in glob.glob('first_level/*_%s.csv' % suffix):
        pred_to_add = pd.read_csv(csv).pred
        pred_to_add.name = csv.replace("_" + suffix + ".csv", 
                                       "").replace("first_level/", "")
        df = pd.concat([df, pred_to_add], axis=1)
    return df

train = pd.read_csv('pre_train.csv')
predictors = load_files()
train = pd.concat([train, predictors], axis=1)
train = train.drop(['v56_%s' % i for i in range(130)], axis=1)
target = train.target
train.drop(['ID', 'target'], axis=1, inplace=True)

if LOCAL:
    train = xgb.DMatrix(train, target)
    xgb.cv(xgboost_params, train, num_boost_round=550, nfold=5,
           seed=0, verbose_eval=1, early_stopping_rounds=5)

# else:
