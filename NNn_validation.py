from NN_numpy import *  # noqa
import matplotlib.pyplot as plt
import math
import copy
import csv


def kfoldvalidation(X_trn, Y_trn, net, k=4, graph=False, ** kwargs):
    if len(X_trn) != len(Y_trn):
        print "X_trn and Y_trn different lengths"
        sys.exit(0)
    perm = np.random.permutation(len(Y_trn))
    num_trn = len(X_trn)
    y_predict = np.empty_like(Y_trn)
    for k_idx in xrange(k):
        train_net = copy.deepcopy(net)
        val_idx = perm[math.floor(k_idx * 1.0 / k * num_trn):
                       math.floor((k_idx + 1) * 1.0 / k * num_trn)]
        if k_idx == k - 1:
            val_idx = perm[math.floor(k_idx * 1.0 / k * num_trn):]
        trn_idx = [i for i in perm if i not in val_idx]
        xval = X_trn[val_idx]
        xtrn = X_trn[trn_idx]
        ytrn = Y_trn[trn_idx]
        train_net.train(xtrn, ytrn, **kwargs)
        y_predict[val_idx] = train_net.predict(xval)
    sum_error = np.sum(np.abs(y_predict - Y_trn))
    train_net = copy.deepcopy(net)
    train_net.train(X_trn, Y_trn, **kwargs)
    if graph:
        plt.plot(Y_trn, label='Validation Values')
        plt.plot(y_predict, label='Predicted Values')
        plt.legend(bbox_to_anchor=(0., 0.01, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        plt.title('k-fold Validating. Mean Accuracy = ' +
                  str(sum_error / num_trn))
        plt.show()
    return [train_net, sum_error / num_trn, y_predict]


def test_input_significance_stocks(StockNN, X_trn, Y_trn,
                                   cols, verb=0, lookback=0):
    x_remove_error = []
    for i in range(len(X_trn[0])):
        x_trn_remove = X_trn[:, [col for col in range(len(X_trn[0]))
                                 if col != i]]
        StockNN_rem = StockNN.copy_structure()
        StockNN_rem.remove_input()
        [_, abs_error, _] = kfoldvalidation(
            X_trn=x_trn_remove, Y_trn=Y_trn,
            net=StockNN_rem, k=5, graph=False,
            LR=.1, epochs=100, verb=verb,
            re_init=3, re_init_d=20)
        x_remove_error.append(abs_error)
        print abs_error
    with open('results/remove_cols.csv', 'wb') as csvfile:
        remove_csv = csv.writer(csvfile, delimiter=',')
        remove_csv.writerow([" S" + str(i + 1) + " "
                             for i in range(cols)])
        for r in range(lookback):
            remove_csv.writerow(["%.2f" % x for x in x_remove_error[
                                r * cols:(r + 1) * (cols)]])
        remove_csv.writerow(['    '] +
                            ["%.2f" % x for x in x_remove_error[-9:]])
    plt.plot(x_remove_error)
    plt.title('mean error for removing each column')
    plt.show()
    print x_remove_error


def validate_model(StockNN_standard, X_trn, Y_trn,
                   verb=0, **kwargs):
    StockNN = copy.deepcopy(StockNN_standard)
    [_, abs_error, Y_predict] = kfoldvalidation(
        X_trn=X_trn, Y_trn=Y_trn, net=StockNN, verb=verb, **kwargs)
    if verb > -1:
        print "mean abs error: ", abs_error
    plt.plot(Y_trn, label='Validation Values')
    plt.plot(Y_predict, label='Predicted Values')
    plt.legend(bbox_to_anchor=(0., 0.01, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.title('Validating Labeled Data. Mean Error = ' +
              str(abs_error))
    plt.show()


def predict_future(net, X_trn, Y_trn, X_tst,
                   cols, lookback, dat, **kwargs):
    StockNN = copy.deepcopy(net)
    StockNN.train(X=X_trn, y=Y_trn, epochs=100, re_init=5, re_init_d=30)
    y_tst = []
    for idx in xrange(len(X_tst)):
        y = StockNN.predict(np.array([X_tst[idx]]))
        y_tst.append(y[0])
        for j in xrange(lookback):
            if idx + j + 1 < len(X_tst):
                X_tst[idx + j + 1][(lookback - j - 1) * cols] = y[0][0]
    y_tst = np.array(y_tst)
    with open('results/Predictions.csv', 'wb') as csvfile:
        predict_csv = csv.writer(csvfile, delimiter=',')
        predict_csv.writerow(["Date", "Value"])
        for r in range(len(y_tst)):
            predict_csv.writerow([dat[50 + r, 0], "%.2f" % y_tst[r][0]])
        cum_change = sum(y_tst)
        print "cumulative change is ", cum_change
    plt.plot(y_tst, label='Predicted Future Values')
    plt.title('Predicted Values')
    plt.show()
