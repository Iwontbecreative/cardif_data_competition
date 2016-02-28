"""
KNN need to be treated differently when it comes to NA
and different kind of values.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

NAME = "KNN_400_p1_wdistance"

train = pd.read_csv('train.csv')
# Picking only the columns with counts > 100k.
train = train[['target', 'v10', 'v12', 'v14', 'v21', 'v34', 'v38',
               'v40', 'v50', 'v62', 'v72', 'v114', 'v128']]
train.fillna(-1, inplace=True) # Is there a better way ?
target = train.target
train.drop(['target'], axis=1, inplace=True)

knc = KNeighborsClassifier(n_neighbors=400, p=1, weights='distance',
                           n_jobs=4)

preds = []
errors = np.zeros(5)

start = datetime.now()
print("Started fitting and predicting at:", start)

for i, (train_ix, test_ix) in enumerate(KFold(len(target), 5)):
    print('Iter', i+1)
    print('Proportion of 1', target.loc[test_ix].mean())
    start_fitting = datetime.now()
    knc.fit(train.loc[train_ix], target.loc[train_ix])
    pred = knc.predict_proba(train.loc[test_ix])[:, 1]
    preds += list(pred)
    error = log_loss(target.loc[test_ix], pred)
    errors[i] = error
    print('Error on fold:', error)
    print('Time expended:', datetime.now() - start_fitting)

print('Error mean: %s, stdev: %s, min: %s, max: %s' % (errors.mean(),
        errors.std(), errors.min(), errors.max()))

pd.DataFrame(preds).to_csv('first_level/%s_train.csv' % NAME, index=False)

print("Total time spent:", datetime.now() - start)
