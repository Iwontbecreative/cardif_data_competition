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

def handle_categorical(df):
    text_col = df.select_dtypes(['object']).columns
    for col in text_col:
        col_to_add = pd.get_dummies(df[col])
        df = df.drop([col], axis=1)
        for i, col2 in enumerate(col_to_add.columns):
            df['%s_%s' % (col, i)] = col_to_add[col2]
    return df
 
train = pd.read_csv('train.csv')
# Picking only the columns with counts > 100k.
train = train[['target', 'v3', 'v10', 'v12', 'v14', 'v21',
               'v31', 'v34', 'v38', 'v40', 'v47', 'v50', 'v52', 'v62',
               'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v107',
               'v112', 'v114', 'v125', 'v128']]
train = train[['v3', 'v31', 'v47', 'v52', 'v71', 'v74', 'v75', 
    'v79', 'v107', 'v112', 'v125']]
train = handle_categorical(train)
print(len(train.columns))
# Overrepresenting v50 to reflect its prediction accuracy.
train['v50bis'] = train.v50
train['v50tris'] = train.v50
train['v50tetris'] = train.v50
train.fillna(-1, inplace=True) # Is there a better way ?
target = train.target
train.drop(['target'], axis=1, inplace=True)

# Scaling variables enable distances to work well.
train = pd.DataFrame(StandardScaler().fit_transform(train))

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
