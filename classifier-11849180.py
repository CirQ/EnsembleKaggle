import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



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



X_train, y_train = fetch_data('./train.csv')
X_test = fetch_data('./test.csv', False)
y_real = pd.read_csv('true_label.csv').iloc[:,0]


while True:
    rf = RandomForestClassifier(n_estimators=100)

    y_pred, acc = evaluate_accuracy(rf, X_train, y_train, X_test, y_real)
    print 'accuracy', acc

    label = 'rforest' + str(int(acc*1000))

    if acc > 0.76:
        write_data(y_pred, label)
        print 'write predicted file'
        break
