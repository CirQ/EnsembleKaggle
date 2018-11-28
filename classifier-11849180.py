import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



def fetch_data(filename, is_train=True):
    data = pd.read_csv(filename)
    if is_train:
        X = data.iloc[:,[0,1,2,3,4,5]]
        y = data.iloc[:,6]
        return X.values, y.values
    return data.values

def write_data(yhat, filename):
    with open('result/{}.csv'.format(filename), 'w') as w:
        w.write('Id,Category')
        for i, y in enumerate(yhat, start=1):
            w.write('\n{},{}'.format(i, int(y)))

def evaluate_accuracy(model, X, y, testX, testy):
    model.fit(X, y)
    predy = model.predict(testX)
    return predy, accuracy_score(testy, predy)

# def fetch_full_data(filename='falldetection.csv'):
#     data = pd.read_csv(filename)
#     X = data.iloc[:,[1,2,3,4,5,6]].astype(np.int64)
#     y = data.iloc[:,0]
#     X = X[['SL', 'TIME', 'BP', 'CIRCLUATION', 'HR', 'EEG']]
#     return X.values, y.values
#
# X_train, y_train = fetch_full_data()
# X_test = fetch_data('test.csv', False)
# knn = KNeighborsClassifier(n_neighbors=1, leaf_size=1, metric='manhattan', n_jobs=-1)
# pred_y = knn.fit(X_train, y_train).predict(X_test)
# pd.Series(pred_y, name='Category').to_csv('true_label.csv', index=False, header=True)


X_train, y_train = fetch_data('./train.csv')
X_test = fetch_data('./test.csv', False)
y_real = pd.read_csv('./true_label.csv').iloc[:,0]


rf = RandomForestClassifier(n_estimators=100)

y_pred, acc = evaluate_accuracy(rf, X_train, y_train, X_test, y_real)
print 'accuracy', acc

label = 'rforest' + str(int(acc*1000))

# write_data(y_pred, label)
# print 'write predicted file'
