"""
Functions that don't change the data.
"""

import csv
from datetime import datetime

def output_csv(ids, pred, name='submissions/submission_%s'):
    with open(name % datetime.now(), 'w') as sub:
        writer = csv.writer(sub)
        writer.writerow(['Id', 'PredictedProb'])
        writer.writerows(zip(ids, pred))

def combine_csv(files, weights='mean'):
    """
    This is a method used to ensemble several csv as produced by
    output_csv and output a new csv.
    weights possible values:
        - a len(files) list with weights for each file
        - 'mean' where for each obs we take the mean prediction
        - 'max' where for each obs we take the max prediction
        - 'min' where for each obs we take the min prediction
    """
    #FIXME: ugly
    ids = []
    preds = []
    for submission in files:
        with open(submission, 'r') as sub:
            reader = csv.reader(sub)
            if not ids:
                for id, pred in reader:
                    try:
                        ids.append(int(id))
                        preds.append([float(pred)])
                    except:
                        pass
            else:
                for i, pred in enumerate(reader):
                    try:
                        preds[i-1].append(float(pred[1]))
                    except:
                        pass

    if weights == 'mean':
        preds = [sum(i)/len(i) for i in preds]
    elif weights == 'max':
        preds = [max(i) for i in preds]
    elif weights == 'min':
        preds = [min(i) for i in preds]
    elif isinstance(weights, list) and len(weights) == len(files):
        preds = [i * w for i, w in zip(preds, weights)]
    else:
        print('Incorrect weights given.')
        return(-1)

    output_csv(ids, preds, 'submissions/combined_%s')
