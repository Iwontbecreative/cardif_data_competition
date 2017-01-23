import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cross_validation import train_test_split, KFold

def handle_categorical(df):
    # text_col = df.select_dtypes(['object']).columns
    # for col in text_col:
        # col_to_add = pd.get_dummies(df[col])
        # df = df.drop([col], axis=1)
        # for i, col2 in enumerate(col_to_add.columns):
            # df['%s_%s' % (col, i)] = col_to_add[col2]
    return df

def handle_nas(df):
    """
    Several ways can be used to replace NAs.
    Currently it looks like the best option is to use -1.
    """
    for col in df.columns:
        df[col].fillna(-1, inplace=True)
    return df

def change_vars(df):
    """
    Add variable transformations here, those will be applied
    before handle_nas and handle_categorical. The changes in confirmed
    were tested against 3-CV XGB.
    """
    # Confirmed:
    df.v50 = np.log1p(df.v50 + 0.01)
    return df

train = pd.read_csv('train.csv')
target = train.target
train = train.drop(['target'], axis=1)
test = pd.read_csv('test.csv')

# Preprocessed train/test
both = change_vars(pd.concat([train, test]))
# We drop v107 because it is a duplicate of v91 
both = both.drop(['v22', 'v107'], axis=1)
both = handle_nas(handle_categorical(both))
both.index = list(range(len(both)))
train = both.ix[:len(train)-1]
train["target"] = target
test = both.ix[len(train):]
train.to_csv('pre_train.csv', index=False)
test.to_csv('pre_test.csv', index=False)
