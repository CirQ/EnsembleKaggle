import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as tic
import sklearn
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

def display_features(X):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
    names = ['Attribute1', 'Attribute2', 'Attribute3',
             'Attribute4', 'Attribute5', 'Attribute6']
    for i, column in enumerate(X.T):
        x, y = i / 3, i % 3
        axes[x,y].hist(column)
        axes[x,y].xaxis.set_major_locator(tic.MaxNLocator(4))
        axes[x,y].set_title('#'+names[i])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.show()

def evaluate_accuracy(model, X, y, testX):
    realy = pd.read_csv('true_label.csv').iloc[:,0]
    model.fit(X, y)
    predy = model.predict(testX)
    return predy, accuracy_score(realy, predy)

def display_clean_diff():
    before_X, _ = fetch_data('train.csv')
    after_X, _ = fetch_data('train_clean.csv')
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(20,5))
    names = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6']
    for i, column in enumerate(before_X.T):
        axes[0,i].hist(column)
        axes[0,i].xaxis.set_major_locator(tic.MaxNLocator(3))
        axes[0,i].yaxis.set_major_locator(tic.MaxNLocator(4))
        axes[0,i].set_title('before #'+names[i])
    for i, column in enumerate(after_X.T):
        axes[1,i].hist(column)
        axes[1,i].xaxis.set_major_locator(tic.MaxNLocator(3))
        axes[1,i].yaxis.set_major_locator(tic.MaxNLocator(4))
        axes[1,i].set_title('after #'+names[i])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()



X_train, y_train = fetch_data('./train.csv')
X_test = fetch_data('./test.csv', False)

rf = RandomForestClassifier(n_estimators=100)

y_pred, acc = evaluate_accuracy(rf, X_train, y_train, X_test)
print 'accuracy', acc
write_data(y_pred, 'naive')
