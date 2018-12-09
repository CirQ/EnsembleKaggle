#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: dell
# Created Time: 2018-12-10 00:31:53
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def outlier_detection(data):
    q1 = data['SL'].quantile(0.995)
    q6 = data['EEG'].quantile(0.005)
    return (data['SL'] > q1) | (data['EEG'] < q6)

def fetch_all_data():
    traindata = pd.read_csv('train.csv')
    outlier_train = outlier_detection(traindata)
    traindata = traindata[~outlier_train]
    X_train = traindata.iloc[:,:6]
    y_train = traindata.iloc[:,-1]

    X_test = pd.read_csv('test.csv')
    y_real = pd.read_csv('true_label.csv')
    # outlier_test = outlier_detection(X_test)

    X_train = X_train.drop(['HR'], axis=1)
    X_test = X_test.drop(['HR'], axis=1)
    return X_train, y_train, X_test, y_real


train_X, train_y, test_X, real_y = fetch_all_data()


def trial(**kwargs):
    dtc = DecisionTreeClassifier(**kwargs)
    dtc.fit(train_X, train_y)
    pred_y = dtc.predict(test_X)
    return accuracy_score(real_y, pred_y)


def draw_depth(accs):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(accs)+1), accs, 'r-')
    ax.set_xlabel('Tree depth')
    ax.set_xlim(0, len(accs)+1)
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    plt.savefig('dt_depth_plot.eps', idp=1000)


def draw_criterion(ent, gini):
    assert len(ent) == len(gini)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(ent)+1), ent, 'b-', label='entropy')
    ax.plot(range(1, len(ent)+1), gini, 'r-', label='gini')
    ax.set_xlabel('Tree depth')
    ax.set_xlim(0, len(gini)+1)
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    plt.savefig('dt_criterion_plot.eps', idp=1000)


def main_depth():
    accs = []
    for d in range(1, 32):
        acc = trial(max_depth=d)
        accs.append(acc)
        print d, 'acc:', acc
    draw_depth(accs)

def main_criterion():
    ents, ginis = [], []
    for d in range(1, 32):
        ent = trial(criterion='entropy', max_depth=d)
        ents.append(ent)
        gini = trial(criterion='gini', max_depth=d)
        ginis.append(gini)
        print d, 'ent:', ent, 'gini:', gini
    draw_criterion(ents, ginis)



if __name__ == '__main__':
    # main_depth()
    main_criterion()
