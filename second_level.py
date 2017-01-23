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
from helpers import output_csv

LOCAL = False

start = datetime.now()
print("Started at:", start)

xgboost_params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "logloss",
    "eta": 0.01,
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
    files = list(glob.glob('first_level/*_%s.csv' % suffix))
    files.sort()
    for csv in files:
        pred_to_add = pd.read_csv(csv).pred
        pred_to_add.name = csv.replace("_" + suffix + ".csv", 
                                       "").replace("first_level/", "")
        df = pd.concat([df, pred_to_add], axis=1)
    return df

def naive_bayes_categorical(train, test):
    text_col = train.select_dtypes(include=['object']).columns
    for col in text_col:
        frequencies = {}
        train[col].fillna(-1, inplace=True)
        test[col].fillna(-1, inplace=True)
        for value in train[col].unique():
            frequencies[value] = train[train[col] == value].target.mean()
        train[col] = train[col].apply(lambda v: frequencies[v] if v != -1 else v)
        test[col] = test[col].apply(lambda v: frequencies[v] if v in frequencies else -1)
    return train, test

train = pd.read_csv('pre_train.csv')
predictors = load_files()
train = pd.concat([train, predictors], axis=1)

if LOCAL:
    train = xgb.DMatrix(train, target)
    xgb.cv(xgboost_params, train, num_boost_round=550, nfold=5,
           seed=0, verbose_eval=1, early_stopping_rounds=5)

else:
    test = pd.read_csv('pre_test.csv')
    predictors = load_files("test")
    test = pd.concat([test, predictors], axis=1)
    train, test = naive_bayes_categorical(train, test)
    target = train.target
    train.drop(['target', 'ID'], axis=1, inplace=True)
    ids = test.ID
    test.drop(['ID'], axis=1, inplace=True)
    train = xgb.DMatrix(train, target)
    test = xgb.DMatrix(test)
    clf = xgb.train(xgboost_params, train, num_boost_round=350)
    pred = clf.predict(test)
    output_csv(ids, pred)

print("Total time:", datetime.now() - start)
