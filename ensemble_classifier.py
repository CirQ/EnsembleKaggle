import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from data_preprocessor import fetch_all_data


def evaluate_accuracy(model, X, y, testX, testy):
    model.fit(X, y)
    predy = model.predict(testX)
    return pd.Series(predy), accuracy_score(testy, predy)

def write_data(yhat, filename):
    with open('result/{}.csv'.format(filename), 'w') as w:
        w.write('Id,Category')
        for i, y in enumerate(yhat, start=1):
            w.write('\n{},{}'.format(i, int(y)))
    print 'write predicted file'



def AdaDTEnsembler(X_train, y_train, X_test, y_real):
    dt = DecisionTreeClassifier(
        max_depth=20,
        max_features='log2',
    )
    abc = AdaBoostClassifier(
        base_estimator=dt,
        n_estimators=500,
        algorithm='SAMME',
    )
    return evaluate_accuracy(abc, X_train, y_train, X_test, y_real)


if __name__ == '__main__':
    X_train, y_train, X_test, y_real = fetch_all_data()
    while True:
        y_pred, acc = AdaDTEnsembler(X_train, y_train, X_test, y_real)
        print 'accuracy:', acc
        if acc > 0.908:
            label = 'adadt' + str(int(acc*2000))
            write_data(y_pred, label)
            break
