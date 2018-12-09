import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import seaborn as sns


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


def plot_feature(features, filename):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    names = ['SL', 'Time', 'BP', 'Circulation', 'HR', 'EEG']
    for i, column in enumerate(features.values.T):
        x, y = i / 3, i % 3
        axes[x, y].hist(column)
        axes[x, y].xaxis.set_major_locator(tic.MaxNLocator(4))
        axes[x, y].set_title('#' + names[i])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.savefig(filename+'.png')

def plot_pairwise_matrix(X, filename):
    sns.pairplot(X, height=2.4)
    plt.tight_layout()
    plt.savefig(filename+'.png')

def plot_correlation_heatmap(X, filename):
    fig, ax = plt.subplots()
    corr = X.corr()
    colormap = sns.color_palette('YlOrRd', 100)
    sns.heatmap(corr, cmap=colormap, annot=True, fmt='.4f')
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, '', ha='center', va='center', color='w')
    fig.tight_layout()
    plt.savefig(filename+'.png')

def plot_feature_labeling(X, y, filename):
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(20, 15))
    features = ['SL', 'Time', 'BP', 'Circulation', 'HR', 'EEG']
    labels = ['Standing', 'Walking', 'Sitting', 'Falling', 'Cramps', 'Running']
    for label in range(6):
        with_label = X[y == label]
        for feature, column in enumerate(with_label.values.T):
            axes[label, feature].hist(column)
            axes[label, feature].xaxis.set_major_locator(tic.MaxNLocator(4))
    for ax, col in zip(axes[0], features):
        ax.set_title('#' + col)
    for ax, row in zip(axes[:,0], labels):
        ax.set_ylabel(row, rotation=0, size='large', labelpad=30)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.tight_layout()
    plt.savefig(filename+'.png')


# plot_feature(X_train, 'trainset_plot')
# plot_feature(X_test, 'testset_plot')
# plot_pairwise_matrix(X_train, 'pairwise_plot')
# plot_correlation_heatmap(X_train, 'correlation_plot')
# plot_feature_labeling(X_train, y_train, 'label_wise_plot')


def outlier_detection(data):
    q1 = data['SL'].quantile(0.995)
    q6 = data['EEG'].quantile(0.005)
    return (data['SL'] > q1) | (data['EEG'] < q6)

def drop_coulmn(data):
    # recorded after 100 runs
    # return data # 0.89879
    # return data.drop(['SL'], axis=1)    # 0.873155
    # return data.drop(['Circulation'], axis=1)   # 0.895555
    # return data.drop(['Time'], axis=1)  # 0.88029
    # return data.drop(['HR'], axis=1)    # 0.895265
    # return data.drop(['SL', 'Time'], axis=1)    # 0.815865
    # return data.drop(['SL', 'HR'], axis=1)  # 0.87282
    # return data.drop(['Circulation', 'Time'], axis=1) # 0.875345
    return data.drop(['Circulation', 'HR'], axis=1) # 0.883365


def fetch_all_data():
    traindata = pd.read_csv('train.csv')
    outlier_train = outlier_detection(traindata)
    traindata = traindata[~outlier_train]
    X_train = traindata.iloc[:,:6]
    y_train = traindata.iloc[:,-1]

    X_test = pd.read_csv('test.csv')
    y_real = pd.read_csv('true_label.csv')
    # outlier_test = outlier_detection(X_test)

    # X_train = drop_coulmn(X_train)
    # X_test = drop_coulmn(X_test)
    return X_train, y_train, X_test, y_real
