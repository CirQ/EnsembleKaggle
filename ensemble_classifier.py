import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from data_preprocessor import fetch_all_data
from util import annotated_timer


def evaluate_accuracy(model, X, y, testX, testy):
    model.fit(X, y)
    predy = model.predict(testX)
    return pd.Series(predy), accuracy_score(testy, predy), model

def write_data(yhat, filename):
    with open('result/{}.csv'.format(filename), 'w') as w:
        w.write('Id,Category')
        for i, y in enumerate(yhat, start=1):
            w.write('\n{},{}'.format(i, int(y)))
    print 'write predicted file'



# @annotated_timer('adaboost dt single round')
def AdaDTEnsembler(X_train, y_train, X_test, y_real, nest=500):
    dt = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=16,
        max_features='log2',
    )
    abc = AdaBoostClassifier(
        base_estimator=dt,
        n_estimators=nest,
        algorithm='SAMME',
    )
    return evaluate_accuracy(abc, X_train, y_train, X_test, y_real)


@annotated_timer('random forest single round')
def RFEnsembler(X_train, y_train, X_test, y_real, nest=500):
    rf = RandomForestClassifier(
        n_estimators=nest,
        criterion='entropy',
        max_depth=16,
        max_features='log2',
        n_jobs=-1,
    )
    return evaluate_accuracy(rf, X_train, y_train, X_test, y_real)


# @annotated_timer('bagging single round')
def BgEnsembler(X_train, y_train, X_test, y_real, nest=500):
    dt = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=16,
        max_features='log2',
    )
    bg = BaggingClassifier(
        base_estimator=dt,
        n_estimators=nest,
        n_jobs=-1,
    )
    return evaluate_accuracy(bg, X_train, y_train, X_test, y_real)


if __name__ == '__main__':
    X_train, y_train, X_test, y_real = fetch_all_data()
    # y_pred, acc, _ = AdaDTEnsembler(X_train, y_train, X_test, y_real)
    y_pred, acc, _ = RFEnsembler(X_train, y_train, X_test, y_real)
    # y_pred, acc, _ = BgEnsembler(X_train, y_train, X_test, y_real)
    print 'accuracy:', acc
    write_data(y_pred, '11849180-submission')
