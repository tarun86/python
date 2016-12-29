import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

directory = "BuildingMachineLearningSystemsWithPython"
data_path = "../" + directory + "/ch01/data/web_traffic.tsv"


def read_data(path):
    data = sp.genfromtxt(path, delimiter="\t")
    return clean_data(data)


def clean_data(data):
    print data.shape
    x = data[:, 0]
    y = data[:, 1]
    x = x[~sp.isnan(y)]
    y = y[~sp.isnan(y)]
    return np.array([x, y])


def plot(x, y, func):
    plt.scatter(x, y)
    plt.title("Web Traffic Data for the Month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks([w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])
    plt.autoscale(tight=True)
    plt.grid()
    fx = sp.linspace(0, x[-1], 1000)
    plt.plot(fx, func(fx), linewidth=2)
    err = error(func, x, y)
    plt.legend(["d=%i error=%.2f" % (func.order, err)], loc="upper left")
    plt.savefig('plot_order_%i.png' % func.order)
    plt.close()


def error(func, x, y):
    return sp.sum((func(x) - y) ** 2)


def model(x, y, dim):
    return sp.polyfit(x, y, dim, full=True)
    # return fp1, residuals, rank, sv, rcond


def ploting_error(x, y, maxD):
    xx = np.array(range(1, maxD));
    yy = np.zeros(maxD - 1);
    for dim in range(1, maxD):
        fp1, residuals, rank, sv, rcond = model(x, y, dim)
        # print "model parameters : %s" % fp1
        f1 = sp.poly1d(fp1)
        err = error(f1, x, y)
        # print "error : %.2f" % err
        yy[dim - 1] = err
    # print yy.shape
    # print xx.shape
    plt.plot(xx, yy, 'ro-')
    plt.xlabel("Degree of Polynomial")
    plt.ylabel("Error")
    plt.grid()
    plt.autoscale(tight=True)
    plt.savefig('error.png')
    plt.close()


def multi_order_training(x, y, dim):
    fp1, residuals, rank, sv, rcond = model(x, y, dim)
    print "model parameters : %s" % fp1
    f1 = sp.poly1d(fp1)
    plot(x, y, f1)


if __name__ == '__main__':
    data = read_data(data_path)
    x = data[0]
    y = data[1]
    multi_order_training(x, y, 1)
    multi_order_training(x, y, 2)
    ploting_error(x, y, 50)
