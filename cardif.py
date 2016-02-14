import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from datetime import datetime
np.random.seed(1)
import xgboost as xgb

# XGBoost here
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

train = pd.read_csv('train.csv')
labels = train.target

# Handle v22 and its thousands of labels.
# This seems to worsen the scores for now.
#v22_frequent_labels = train.v22.value_counts().index[:25]
#new_v22 = []
#for lbl in train.v22:
    #if lbl in v22_frequent_labels:
        #new_v22.append(lbl)
    #else:
        #new_v22.append('Rare Value')
#train.v22 = new_v22

# Weird feature about NA distribution
#weird = []
#for i, j, k, l in zip(train.v1, train.v6, train.v7, train.v111):
    #if pd.isnull(i) and pd.isnull(j) and pd.isnull(k) and pd.isnull(l):
        #weird.append(1)
    #else:
        #weird.append(0)
#train["weird"] = weird

train = train.drop(['target', 'ID', 'v22'], axis=1)

# V12 custom
train.v12 = np.log1p(train.v12)
# Feature importance > 0.01 with RF 100 est
#train = train[['v10', 'v12', 'v21', 'v34', 'v40', 'v50',
    #'v114', 'v129']]

# Transform text to categorical
text_col = train.select_dtypes(['object']).columns
for col in text_col:
    col_to_add = pd.get_dummies(train[col])
    train = train.drop([col], axis=1)
    for i, col2 in enumerate(col_to_add.columns):
        train['%s_%s' % (col, i)] = col_to_add[col2]

print(len(train.columns))

# Fill na with most frequent value outside of NaNs
for col in train.columns:
    #train[col].fillna(train[col].value_counts().index[0], inplace=True)
    train[col].fillna(0, inplace=True)

Xtrain, Xtest, ytrain, ytest = train_test_split(train, labels, test_size=0.2)

start = datetime.now()

#rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rfc = LogisticRegression(penalty='l2')
#rfc = KNeighborsClassifier(100)
rfc.fit(Xtrain, ytrain)

# XGBoost code
#xgtrain = xgb.DMatrix(Xtrain, ytrain)
#xgtest2 = xgb.DMatrix(Xtest, ytest)
#xgtest = xgb.DMatrix(Xtest)
#print('Fitting the model')
#clf = xgb.train(xgboost_params, xgtrain, num_boost_round=1500,
        #evals=[(xgtrain, 'Train'), (xgtest2, 'Test')])

#print('Predicting')
#ypred = clf.predict(xgtest)


# Sklearn stuff
ypred = rfc.predict_proba(Xtest)
ypred = [i[1] for i in ypred]
#for i, j in zip(train.columns, rfc.feature_importances_):
#    if j > 0.005:
#        print(i, j)

print(ypred[:50])
print(log_loss(ytest, ypred))
print(datetime.now() - start)
