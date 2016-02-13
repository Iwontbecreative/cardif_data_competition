import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from datetime import datetime


train = pd.read_csv('train.csv')
labels = train.target
train = train.drop(['target', 'ID', 'v22'], axis=1)

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
    train[col].fillna(train[col].value_counts().index[0], inplace=True)

Xtrain, Xtest, ytrain, ytest = train_test_split(train, labels, test_size=0.2)

start = datetime.now()

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(Xtrain, ytrain)

ypred = rfc.predict_proba(Xtest)
ypred = [i[1] for i in ypred]
print(ypred[:50])
print(log_loss(ytest, ypred))
print(datetime.now() - start)
