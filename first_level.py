"""
The goal here is to fit predictors to test and train
so that they can be used as variables by level 2 models.
To achieve this we use a 5-fold split to generate for train
then we generate for test.
Results are stored in first_level folder in the form of
model_train.csv and model_test.csv.
For more information on why this is useful, see stacking.
"""

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import xgboost as xgb
np.random.seed(1)
from datetime import datetime

NAME = "XGB_def_params"
XGB = "XGB" in NAME

start = datetime.now()
print('Starting: %s\n' % start)

train = pd.read_csv('pre_train.csv')
train = train.drop(['v56_%s' % i  for i in range(130)], axis=1)
target = train.target
ids = train.ID
train = train.drop(['target', 'ID'], axis=1)

if XGB:
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
else:
    rfc = ExtraTreesClassifier(n_estimators=1000, criterion='entropy',
                               n_jobs=4, random_state=1, verbose=1)
    # rfc = KNeighborsClassifier(n_neighbors=400, weights='distance', p=1,
                               # n_jobs=4)

# preds = []
# errors = np.zeros(5)

# print('Generating local scores.')
# for i, (train_ix, test_ix) in enumerate(KFold(len(target), 5)):
    # print('Iter', i + 1)
    # print('Proportion of 1', target.loc[test_ix].mean())
    # start_fitting = datetime.now()
    # if XGB:
        # xgb_train = xgb.DMatrix(train.loc[train_ix], target.loc[train_ix])
        # xgb_test = xgb.DMatrix(train.loc[test_ix], target.loc[test_ix])
        # eval = [(xgb_train, 'Train'), (xgb_test, 'Test')]
        # clf = xgb.train(xgboost_params, xgb_train, num_boost_round=200,
                        # evals=eval, verbose_eval=5)
        # pred = clf.predict(xgb_test)
    # else:
        # rfc.fit(train.loc[train_ix], target.loc[train_ix])
        # pred = rfc.predict_proba(train.loc[test_ix])[:, 1]
    # preds += list(pred)
    # error = log_loss(target.loc[test_ix], pred)
    # errors[i] = error
    # print('Error on fold:', error)
    # print('Time expended:', datetime.now() - start_fitting)

# print('Error mean: %s, stdev: %s, min: %s, max: %s' % (errors.mean(),
    # errors.std(), errors.min(), errors.max()))

# feature = pd.DataFrame({'ID': ids, 'pred': preds})
# feature.sort(['ID']).to_csv('first_level/%s_train.csv' % NAME, index=False)

# Build for test

print('Fitting and predicting whole dataset.')
start_fitting = datetime.now()

if XGB:
    xgb_train = xgb.DMatrix(train, target)
    clf = xgb.train(xgboost_params, xgb_train, num_boost_round=250,
                    verbose_eval=25)
    del train
    del xgb_train
else:
    rfc.fit(train, target)
    del train

# This is done here to avoid using too much memory.
print('Generating test scores')
test = pd.read_csv('pre_test.csv')
test = test.drop(['v56_%s' % i  for i in range(130)], axis=1)
ids = test.ID
test = test.drop(['ID'], axis=1)

pred = clf.predict(xgb.DMatrix(test)) if XGB else rfc.predict_proba(test)[:, 1]

print('Time expended:', datetime.now() - start_fitting)
feature = pd.DataFrame({'ID': ids, 'pred': pred})
feature.sort(['ID']).to_csv('first_level/%s_test.csv' % NAME, index=False)

print('Total elapsed time:', datetime.now() - start)
