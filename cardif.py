import pandas as pd
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from datetime import datetime
import xgboost as xgb
np.random.seed(1)

leaderboard = False
use_xgb = False


def handle_categorical(df):
    text_col = df.select_dtypes(['object']).columns
    for col in text_col:
        col_to_add = pd.get_dummies(df[col])
        df = df.drop([col], axis=1)
        for i, col2 in enumerate(col_to_add.columns):
            df['%s_%s' % (col, i)] = col_to_add[col2]
    return df

def handle_nas(df):
    for col in df.columns:
        #df[col].fillna(df[col].value_counts().index[0], inplace=True)
        df[col].fillna(0, inplace=True)
    return df

def run_xgboost(train, target, test, test_target, leaderboard=False):
    xgboost_params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "logloss",
        "eta": 0.01,
        "base_score": 0.76,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_depth": 7,
        "min_child_weight": 1,
        "seed": 1,
        #"lambda": 1.5
        }
    train = xgb.DMatrix(train, target)
    if leaderboard:
        test = xgb.DMatrix(Xtest, ytest)
        eval = [(train, 'Train'), (test, 'Test')]
    else:
        eval = [(train, 'Train')]
    print('Fitting the model')
    start = datetime.now()
    clf = xgb.train(xgboost_params, train, num_boost_round=1500, evals=eval)
    print('Predicting')
    pred = clf.predict(test)
    print('Fitting + Predicting time : %s' % datetime.now() - start)
    return pred

def output_csv(ids, pred):
    with open('submissions/submission_%s' % datetime.now(), 'w') as sub:
        writer = csv.writer(sub, separator=',')
        writer.writerow(['Id', 'PredictedProb'])
        writer.writerows(zip(ids, pred))


# Would be nice to be able to use this.
#train["64489"] = np.all([pd.isnull(train[col]) for col in train.columns if train[col].count() == 64489], axis=0).astype(int)

train = pd.read_csv('train.csv')
labels = train.target
train = train.drop(['target', 'ID', 'v22'], axis=1)
train = handle_categorical(train)
train = handle_nas(train)


# V12 custom
#train.v12 = np.log1p(train.v12)

# Fill na with most frequent value outside of NaNs


Xtrain, Xtest, ytrain, ytest = train_test_split(train, labels, test_size=0.2)



if use_xgb:
    pred = run_xgboost(Xtrain, ytrain, leaderboard)
    if leaderboard:
        output_csv(pred)
    else:
        print('Score :', log_loss(ytest, pred))
else:
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    #rfc = LogisticRegression(penalty='l2')
    rfc.fit(Xtrain, ytrain)
    pred = rfc.predict_proba(Xtest)
    pred = [i[1] for i in pred]
    #for i, j in zip(train.columns, rfc.feature_importances_):
    #    if j > 0.005:
    #        print(i, j)
    if leaderboard:
        output_csv(pred)
    else:
        print('Score :', log_loss(ytest, pred))
