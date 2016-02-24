"""
The goal here is to fit predictors to test and train
so that they can be used as variables by level 2 models.
To achieve this we use a 5-fold split to generate for train
then we generate for test.
Results are stored in first_level folder in the form of
model_train.csv and model_test.csv.
For more information on why this is useful, see stacking.
"""

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
np.random.seed(1)
from datetime import datetime

NAME = "RFC_100est"

start = datetime.now()
print('Starting :', start)

train = pd.read_csv('pre_train.csv')
train = train.reindex(np.random.permutation(train.index))
target = train.target
ids = train.ID
train = train.drop(['target', 'ID'], axis=1)

# Train KFold
rfc = RandomForestClassifier()
preds = []

for i, (train_ix, test_ix) in enumerate(StratifiedKFold(target, 2, shuffle=False)):
    print('Iter:', i)
    start_fitting = datetime.now()
    rfc.fit(train.loc[train_ix], target.loc[train_ix])
    preds += list(rfc.predict_proba(train.loc[test_ix])[:, 1])
    print('Time expended:', datetime.now() - start_fitting)

feature = pd.DataFrame({'ID': ids, 'pred': preds})
feature.sort(['ID']).to_csv('first_level/%s_train.csv' % NAME, index=False)

# Build for test
test = pd.read_csv('pre_test.csv')
ids = test.ID
test = test.drop(['ID'], axis=1)

rfc.fit(train, target)
pred = rfc.predict_proba(test)[:, 1]
feature = pd.DataFrame({'ID': ids, 'pred': pred})
feature.sort(['ID']).to_csv('first_level/%s_test.csv' % NAME, index=False)
