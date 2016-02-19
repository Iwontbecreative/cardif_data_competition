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

leaderboard = True
use_xgb = True


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
        df[col].fillna(-1, inplace=True)
    return df

def change_vars(df):
    """
    Add variable transformations here.
    """
    #df.v12 = np.log1p(df.v12 + 5)
    #df.v21 = np.log1p(df.v21 + 5)
    #df.v34 = np.log1p(df.v34 + 5)
    return df

def run_xgboost(train, target, test, test_target=None,
    leaderboard=False):
    
    xgboost_params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "logloss",
        "eta": 0.01,
        "base_score": 0.76,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_depth": 10,
        "min_child_weight": 1,
        "seed": 1,
        #"lambda": 1.5
        }
    train = xgb.DMatrix(train, target)
    if not leaderboard:
        test = xgb.DMatrix(test, test_target)
        eval = [(train, 'Train'), (test, 'Test')]
    else:
        eval = [(train, 'Train')]
        test = xgb.DMatrix(test)
    print('Fitting the model')
    clf = xgb.train(xgboost_params, train, num_boost_round=2200, evals=eval)
    print('Predicting')
    pred = clf.predict(test)
    return pred

def output_csv(ids, pred):
    with open('submissions/submission_%s' % datetime.now(), 'w') as sub:
        writer = csv.writer(sub)
        writer.writerow(['Id', 'PredictedProb'])
        writer.writerows(zip(ids, pred))


# Would be nice to be able to use this.
#train["64489"] = np.all([pd.isnull(train[col]) for col in train.columns if train[col].count() == 64489], axis=0).astype(int)

Xtrain = pd.read_csv('train.csv')
# Remove weird items.
#Xtrain.drop(60422, axis=0, inplace=True)
ytrain = Xtrain.target
# Remove v58 because high correlation (-0.997) with v100
# Remove v22 because too many categorical
Xtrain = Xtrain.drop(['target', 'ID', 'v22', 'v58'], axis=1)



if leaderboard:
    Xtest = pd.read_csv('test.csv')
    ids = Xtest.ID
    Xtest = Xtest.drop(['ID', 'v22', 'v58'], axis=1)
    both = handle_nas(handle_categorical(pd.concat([Xtrain, Xtest])))
    both = change_vars(both)
    both.index = list(range(len(both)))
    print(len(Xtrain))
    Xtrain = both.ix[:len(Xtrain)-1]
    Xtest = both.ix[len(Xtrain):]
else:
    Xtrain = handle_categorical(Xtrain)
    Xtrain = handle_nas(Xtrain)
    Xtrain = change_vars(Xtrain)
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xtrain, ytrain, test_size=0.2)



if use_xgb:
    if leaderboard:
        pred = run_xgboost(Xtrain, ytrain, Xtest, None, leaderboard)
        output_csv(ids, pred)
    else:
        pred = run_xgboost(Xtrain, ytrain, Xtest, ytest, leaderboard)
        print('Score :', log_loss(ytest, pred))
else:
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    #rfc = LogisticRegression(penalty='l2')
    rfc.fit(Xtrain, ytrain)
    pred = rfc.predict_proba(Xtest)
    pred = [i[1] for i in pred]
    for i, j in zip(Xtrain.columns, rfc.feature_importances_):
        if j < 0.001:
            print(i, j*100)
    if leaderboard:
        output_csv(ids, pred)
    else:
        print('Score :', log_loss(ytest, pred))
