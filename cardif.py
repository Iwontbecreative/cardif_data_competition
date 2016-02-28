from helpers import output_csv
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
np.random.seed(4)
from sklearn.cross_validation import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

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
    """
    Several ways can be used to replace NAs.
    Currently it looks like the best option is to use -1.
    Creating variables with info about NAs seems to worsen score.
    """
    for col in df.columns:
        #df[col].fillna(df[col].value_counts().index[0], inplace=True)
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

def run_xgboost(train, target, test=None, test_target=None,
    leaderboard=False):
    """
    Run XGBoost in both local and leaderboard mode.
    """
    xgboost_params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "logloss",
        "eta": 0.03,
        "base_score": 0.761,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_depth": 10,
        "min_child_weight": 0.75,
        }
    train = xgb.DMatrix(train, target)

    if not leaderboard:
        xgb.cv(xgboost_params, train, num_boost_round=150, nfold=5,
               seed=0, verbose_eval=1)
    else:
        eval = [(train, 'Train')]
        test = xgb.DMatrix(test)
        clf = xgb.train(xgboost_params, train, num_boost_round=180, evals=eval)
        return clf.predict(test)

def run_sklearn(train, target, test):
    """
    Run a RFC.
    """
    rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    rfc.fit(train, target)
    pred = rfc.predict_proba(test)
    pred = [i[1] for i in pred]
    # for i, j in zip(Xtrain.columns, rfc.feature_importances_):
        # if j < 0.001:
            # print(i, j*100)
    return pred



# Would be nice to be able to use this.
#train["64489"] = np.all([pd.isnull(train[col]) for col in train.columns if train[col].count() == 64489], axis=0).astype(int)

print("Start processing.")
start = datetime.now()
Xtrain = pd.read_csv('train.csv')
bonus = pd.read_csv('first_level/RFC_1000est_entropy_train.csv').pred
Xtrain = pd.concat([Xtrain, bonus], axis=1)
bonus = pd.read_csv('first_level/LR_l1_train.csv').pred
bonus.name = "pred2"
Xtrain = pd.concat([Xtrain, bonus], axis=1)
bonus = pd.read_csv('first_level/LR_train.csv').pred
bonus.name = "pred3"
Xtrain = pd.concat([Xtrain, bonus], axis=1)
bonus = pd.read_csv('first_level/XGB_def_params_train.csv').pred
bonus.name = "pred4"
Xtrain = pd.concat([Xtrain, bonus], axis=1)
# Remove weird items.
#Xtrain.drop(60422, axis=0, inplace=True)
ytrain = Xtrain.target
# Remove v22 because too many categorical
Xtrain = Xtrain.drop(['target', 'ID', 'v22'], axis=1)



if leaderboard:
    Xtest = pd.read_csv('test.csv')
    ids = Xtest.ID
    Xtest = Xtest.drop(['ID', 'v22', 'v56'], axis=1)
    bonus = pd.read_csv('first_level/RFC_1000est_entropy_test.csv').pred
    Xtest = pd.concat([Xtest, bonus], axis=1)
    bonus = pd.read_csv('first_level/LR_l1_test.csv').pred
    bonus.name = "pred2"
    Xtest = pd.concat([Xtest, bonus], axis=1)
    bonus = pd.read_csv('first_level/LR_test.csv').pred
    bonus.name = "pred3"
    Xtest = pd.concat([Xtest, bonus], axis=1)
    bonus = pd.read_csv('first_level/XGB_def_params_test.csv').pred
    bonus.name = "pred4"
    Xtest = pd.concat([Xtest, bonus], axis=1)
    both = change_vars(pd.concat([Xtrain, Xtest]))
    both.v56.fillna('AAAAAAAA', inplace=True)
    both.v56 = LabelEncoder().fit_transform(both.v56)
    both = handle_nas(handle_categorical(both))
    both.index = list(range(len(both)))
    Xtrain = both.ix[:len(Xtrain)-1]
    Xtest = both.ix[len(Xtrain):]
else:
    Xtrain = change_vars(Xtrain)
    Xtrain = handle_categorical(Xtrain)
    Xtrain = handle_nas(Xtrain)
    print("Processing time : ", datetime.now() - start)


print('Fitting and predicting')
start = datetime.now()
if use_xgb:
    if leaderboard:
        pred = run_xgboost(Xtrain, ytrain, Xtest, None, leaderboard)
        output_csv(ids, pred)
    else:
        run_xgboost(Xtrain, ytrain, leaderboard=leaderboard)
# Split if we use RFC or sklearn. XGBoost handles it.
else:
    if not leaderboard:
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xtrain, ytrain, test_size=0.2)
    pred = run_sklearn(Xtrain, ytrain, Xtest)
    if leaderboard:
        output_csv(ids, pred)
    else:
        print('Score :', log_loss(ytest, pred))
print('Fitting and predicting time :', datetime.now() - start)
