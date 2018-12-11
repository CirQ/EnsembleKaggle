#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: dell
# Created Time: 2018-12-12 00:08:05

import pickle
import time

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from data_preprocessor import drop_coulmn



def runner(ensembler, nest, *datas):
    start = time.time()
    _, acc, _ = ensembler(*datas, nest=nest)
    elapsed = time.time() - start
    return acc, elapsed


def stats(ensembler, name, *datas):
    statistic = []
    for d in range(5, 200, 5):
        ass, tss = [], []
        for _ in range(3):
            a, t = runner(ensembler, d, *datas)
            ass.append(a)
            tss.append(t)
        a = np.mean(ass)
        t = np.mean(tss)
        print d, a, t
        statistic.append((d,a,t))
    with open('statistics_'+name+'.pickle', 'wb') as w:
        pickle.dump(statistic, w)
    return statistic


def draw(name):
    with open('statistics_'+name+'.pickle', 'rb') as r:
        stat = pickle.load(r)
    print stat
    n, acc, tim = zip(*stat)

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('number of classifiers')
    ax1.set_ylabel('accuracy', color=color)
    ax1.plot(n, acc, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_xlim(0, 200)
    ax2.set_ylabel('time (s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(n, tim, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(name+'_plot.eps', dpi=1000)


def analyze(skipdump=False):
    if not skipdump:
        clf = joblib.load('best_model.pickle')
        X_test = pd.read_csv('test.csv')
        X_test = drop_coulmn(X_test)
        y_real = pd.read_csv('true_label.csv')
        y_pred = clf.predict(X_test)
        c = confusion_matrix(y_real, y_pred)
    else:
        c = np.array([[261,   0,   0,  10,   8,   3],
                      [  0,  29,   2,   0,   0,   0],
                      [  3,   7, 111,  25,   4,   1],
                      [  1,   1,  27, 161,  30,   2],
                      [  6,   0,   6,  38, 146,  19],
                      [  2,   0,   4,  11,  22,  60]], dtype=np.int64)
    labels = ['Standing', 'Walking', 'Sitting', 'Falling', 'Cramps', 'Running']
    c = pd.DataFrame(c, columns=labels, index=labels)
    fig, ax = plt.subplots()
    colormap = sns.cubehelix_palette(256, start=.5, rot=-.75)
    sns.heatmap(c, cmap=colormap, annot=True, fmt='d')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, '', ha='center', va='center', color='w')
    fig.tight_layout()
    plt.savefig('disc_conf.eps', dpi=1000)



if __name__ == '__main__':
    # datasets = fetch_all_data()
    # sta = stats(AdaDTEnsembler, 'ada', *datasets)
    # draw('ada')
    analyze(True)
