"""
Functions that don't change the data.
"""

import csv
from datetime import datetime

def output_csv(ids, pred, name='submissions/submission_%s.csv'):
    with open(name % datetime.now(), 'w') as sub:
        writer = csv.writer(sub)
        writer.writerow(['Id', 'PredictedProb'])
        writer.writerows(zip(ids, pred))

def combine_csv(files, weights='mean', center=0.7611):
    """
    This is a method used to ensemble several csv as produced by
    output_csv and output a new csv.
    weights possible values:
        - a len(files) list with weights for each file
        - 'mean' where for each obs we take the mean prediction
        - 'max' where for each obs we take the max prediction
        - 'min' where for each obs we take the min prediction
        - 'min_dist' where for each obs we take the closest to a number
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
    elif weights == 'min_dist':
        #TODO: Implement this elegantly.
        pass
    else:
        print('Incorrect weights given.')
        return(-1)

    output_csv(ids, preds, 'submissions/combined_%s.csv')
