import pandas as pd
from sklearn.cross_validation import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from datetime import datetime
import random
random.seed(1)
import xgboost as xgb

# XGBoost here
xgboost_params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "logloss",
        "eta": 0.01,
        "subsample": 0.75,
        "colsample_bytree": 0.68,
        "max_depth": 7
        }



train = pd.read_csv('train.csv')
labels = train.target
train = train.drop(['target', 'ID', 'v22'], axis=1)
# Feature importance > 0.01 with RF 100 est
#train = train[['v10', 'v12', 'v21', 'v34', 'v40', 'v50', 
    #'v114', 'v129']]

# Transform text to categorical
text_col = train.select_dtypes(['object']).columns
for col in text_col:
    col_to_add = pd.get_dummies(train[col])
    train = train.drop([col], axis=1)
    for i, col2 in enumerate(col_to_add.columns):
        train['col_%s' % i] = col_to_add[col2]

print(len(train.columns))

# Fill na with most frequent value outside of NaNs
for col in train.columns:
    #train[col].fillna(train[col].value_counts().index[0], inplace=True)
    train[col].fillna(0, inplace=True)

Xtrain, Xtest, ytrain, ytest = train_test_split(train, labels, test_size=0.2)

start = datetime.now()

#rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
#rfc = LogisticRegression(penalty='l2', C=2)
#rfc = KNeighborsClassifier(100)
#rfc.fit(Xtrain, ytrain)

# XGBoost code
xgtrain = xgb.DMatrix(Xtrain, ytrain)
xgtest = xgb.DMatrix(Xtest)
print('Fitting the model')
boost_round = 1800
clf = xgb.train(xgboost_params, xgtrain, num_boost_round=boost_round, evals=[(xgtrain, 'logloss')])

print('Predicting')
ypred = clf.predict(xgtest)



#ypred = rfc.predict_proba(Xtest)
#ypred = [i[1] for i in ypred]
#for i, j in zip(train.columns, rfc.feature_importances_):
#    if j > 0.005:
#        print(i, j)

print(ypred[:50])
print(log_loss(ytest, ypred))
print(datetime.now() - start)
