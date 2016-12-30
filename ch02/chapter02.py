import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# load the iris data
data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
target_names = data['target_names']
labels = target_names[target]
plength = features[:, 2]
is_setosa = (labels == 'setosa')
max_setosa = plength[is_setosa].max()
min_not_setosa = plength[~is_setosa].min()

print "Maximum of setosa : {0}".format(max_setosa)
print "Minimum of setosa : {0}".format(min_not_setosa)

setosa = features[:, 2] < 2
print (labels[setosa] == labels[is_setosa]).mean()

features_copy = features.copy()
features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')
print virginica.shape
best_acc = -1.0
best_fi = 0
best_t = 0
for fi in xrange(features.shape[1]):
    thresh = features[:, fi].copy()
    thresh.sort()
    for t in thresh:
        pred = (features[:, fi] > t)
        acc = (pred == virginica).mean()
        if (acc > best_acc):
            best_acc = acc
            best_fi = fi
            best_t = t

print best_acc, best_fi, best_t
print feature_names
# : print 'Iris Setosa'
# else: print 'Iris Virginia or Iris Versicolour'
features = features_copy

import math


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def dimesions(num, default_rows=2):
    cols = num / default_rows
    rows = num / cols
    return rows, cols


def ploting_features():
    count = 1
    for i in range(len(feature_names)):
        rows, cols = dimesions(nCr(len(feature_names), 2), 3)
        for j in range(i + 1, len(feature_names)):
            xlabel = feature_names[i]
            ylabel = feature_names[j]
            min = 0
            plt.subplot(rows, cols, count)
            for t, marker, c in zip(xrange(3), ">ox", "rgb"):
                plt.scatter(features[target == t, i], features[target == t, j],
                            marker=marker,
                            c=c)
                plt.xlabel(feature_names[i])
                plt.ylabel(feature_names[j])

            if (i == best_fi):
                minx = features[:, j].min()
                maxx = features[:, j].max()
                yy = np.linspace(minx, maxx, 200)
                xx = np.ones(200) * best_t
                plt.plot(xx, yy, 'r-')
            elif j == best_fi:
                minx = features[:, i].min()
                maxx = features[:, i].max()
                xx = np.linspace(minx, maxx, 200)
                yy = np.ones(200) * best_t
                plt.plot(xx, yy, 'r-')

            count += 1
    plt.savefig('feature_plot.png')
    plt.close()


# ploting_features()


