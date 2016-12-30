import numpy as np
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class SeedDataSet:
    directory = "BuildingMachineLearningSystemsWithPython"
    data_path = "../" + directory + "/ch02/data/seeds.tsv"

    def __init__(self, k):
        """
        The class constructor
        """
        self.data, self.labels = self.readfile(self.data_path)
        self.k = k
        self.classifier = KNeighborsClassifier()

    def readfile(self, filename, delimit="\t"):
        """
        :param self:
        :param filename:
        :param delimit:
        """
        features = []
        labels = []
        with open(filename) as ifile:
            for line in ifile:
                tokens = line.split('\t')
                features.append([float(tk) for tk in tokens[:-1]])
                labels.append(tokens[-1].strip())
        return np.array(features), np.array(labels)

    def leave_one_out(self):
        n = len(self.data)
        correct = 0.0
        for ei in range(n):
            training = np.ones(n, bool)
            training[ei] = 0
            testing = ~training
            self.classifier.fit(self.data[training], self.labels[training])
            pred = self.classifier.predict(self.data[ei])
            correct += (pred == self.labels[ei])
        print('Result of leave-one-out: {}'.format(correct / n))

    def cross_validation(self, n_folds):
        self.kf = KFold(len(self.data), n_folds, shuffle=True)
        self.means = []
        for train, test in self.kf:
            self.classifier.fit(self.data[train], self.labels[train])
            prediction = self.classifier.predict(self.data[test])
            test_labels = self.labels.copy()
            test_labels = test_labels[test]
            # print prediction.shape, test_labels.shape
            curmean = np.mean(prediction == test_labels)
            self.means.append(curmean)
        print('Result of cross-validation using KFold: {}'.format(self.means))

    def scaling_features(self):
        crossed = cross_val_score(self.classifier, self.data, self.labels, cv=10)
        print('Result of cross-validation using cross_val_score: {}'.format(crossed))
        classifier = Pipeline([('norm', StandardScaler()), ('knn', self.classifier)])
        crossed = cross_val_score(classifier, self.data, self.labels, cv=10)
        print('Result with prescaling: {}'.format(crossed))

    def confusionmatrix(self):
        names = list(set(self.labels))
        labels = np.array([names.index(ell) for ell in self.labels])
        preds = labels.copy()
        preds[:] = -1
        for train, test in self.kf:
            self.classifier.fit(self.data[train], labels[train])
            preds[test] = self.classifier.predict(self.data[test])
            print  test
        print preds.shape, labels.shape
        cmat = confusion_matrix(labels, preds)
        print()
        print('Confusion matrix: [rows represent true outcome, columns predicted outcome]')
        print(cmat)
        acc = cmat.trace() / float(cmat.sum())
        print('Accuracy: {0:.1%}'.format(acc))

obj = SeedDataSet(5)
obj.leave_one_out()
obj.cross_validation(10)
obj.scaling_features()
