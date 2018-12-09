import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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



X_train, y_train, X_test, y_real = fetch_all_data()

accs = []
for trial in range(100):
    rf = RandomForestClassifier(
        n_estimators=100,
        bootstrap=False,
        n_jobs=-1,
    )
    y_pred, acc = evaluate_accuracy(rf, X_train, y_train, X_test, y_real)
    accs.append(acc)
    print '{} trial: {}'.format(trial+1, acc)

print 'accuracy', np.mean(accs)

# label = 'rforest' + str(int(acc*2000))

# write_data(y_pred, label)