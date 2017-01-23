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

def naive_bayes_categorical(train, test):
    """
    Returns a dict containing naive bayes prob for each value of categorical
    variables.
    """
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

to_drop = ['v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']

NAME = "ETC_750est_entropy"
XGB = "XGB" in NAME

start = datetime.now()
print('Starting: %s\n' % start)

train = pd.read_csv('pre_train.csv')
train.drop(to_drop, axis=1, inplace=True)
target = train.target
ids = train.ID
train = train.drop(['ID'], axis=1)

if XGB:
    xgboost_params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "logloss",
        "eta": 0.1,
        "base_score": 0.761,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_depth": 10,
        "min_child_weight": 0.75,
        }
else:
    # rfc = RandomForestClassifier(n_estimators=1000, criterion='entropy',
                               # n_jobs=4, random_state=1)
    rfc = LogisticRegression()
    rfc = ExtraTreesClassifier(n_estimators=750, criterion='entropy',
                               n_jobs=4, max_features=70, min_samples_split=4,
                               min_samples_leaf=1, max_depth=45)


preds = []
errors = np.zeros(5)

print('Generating local scores.')
for i, (train_ix, test_ix) in enumerate(KFold(len(target), 5)):
    print('Iter', i + 1)
    print('Proportion of 1', target.loc[test_ix].mean())
    start_fitting = datetime.now()
    if XGB:
        #FIXME: This does not work since NB change. 
        xgb_train = xgb.DMatrix(train.loc[train_ix], target.loc[train_ix])
        xgb_test = xgb.DMatrix(train.loc[test_ix], target.loc[test_ix])
        eval = [(xgb_train, 'Train'), (xgb_test, 'Test')]
        clf = xgb.train(xgboost_params, xgb_train, num_boost_round=200,
                        evals=eval, verbose_eval=5)
        pred = clf.predict(xgb_test)
    else:
        subtrain, subtest = naive_bayes_categorical(train.loc[train_ix],
                                                    train.loc[test_ix])
        train_target, test_target = subtrain.target, subtest.target
        subtrain.drop(['target'], axis=1, inplace=True)
        subtest.drop(['target'], axis=1, inplace=True)
        rfc.fit(subtrain, train_target)
        pred = rfc.predict_proba(subtest)[:, 1]
    preds += list(pred)
    error = log_loss(test_target, pred)
    errors[i] = error
    print('Error on fold:', error)
    print('Time expended:', datetime.now() - start_fitting)

print('Error mean: %s, stdev: %s, min: %s, max: %s' % (errors.mean(),
    errors.std(), errors.min(), errors.max()))

feature = pd.DataFrame({'ID': ids, 'pred': preds})
feature.sort(['ID']).to_csv('first_level/%s_train.csv' % NAME, index=False)

# Build for test
print('Fitting and predicting whole dataset.')
start_fitting = datetime.now()

if XGB:
    #FIXME: This doesn't work since NB changes.
    xgb_train = xgb.DMatrix(train, target)
    clf = xgb.train(xgboost_params, xgb_train, num_boost_round=250,
                    verbose_eval=25)
else:
    test = pd.read_csv('pre_test.csv')
    test.drop(to_drop, axis=1, inplace=True)
    ids = test.ID
    test.drop(['ID'], axis=1, inplace=True)
    train, test = naive_bayes_categorical(train, test)
    target = train.target
    train.drop(['target'], axis=1, inplace=True)
    rfc.fit(train, target)

print('Generating test scores')

pred = clf.predict(xgb.DMatrix(test)) if XGB else rfc.predict_proba(test)[:, 1]

print('Time expended:', datetime.now() - start_fitting)
feature = pd.DataFrame({'ID': ids, 'pred': pred})
feature.sort(['ID']).to_csv('first_level/%s_test.csv' % NAME, index=False)

print('Total elapsed time:', datetime.now() - start)
