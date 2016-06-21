from NN_numpy import *  # noqa
import matplotlib.pyplot as plt
import time

def test_xor(verb=0, re_init=10, netstruc='feed_forward'):
    tNN = NNn([2, 100, 100, 1], reg=10 ** -8, netstruc=netstruc)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # X = np.tile(X, (100, 1))
    y = np.array([np.array([x[0] ^ x[1]]) for x in X])
    tNN.train(X, y, epochs=1000, verb=verb, batch_type=group,
              re_init=re_init, re_init_d=100)
    y_predict = tNN.predict(X)
    if verb > 0:
        print np.around(y_predict, 2)
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print np.mean(np.square(y_predict - y))
    return np.mean(np.square(y_predict - y))
# test_xor()


def test_xor3(verb=0, netstruc='cascade'):
    tNN = NNn([3, 500, 500, 500, 1], netstruc=netstruc, reg=10 * 8 - 10)
    tNN.Lmap[0][-1] = 0
    X = np.array([[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([np.array([x[0] ^ x[1] ^ x[2]]) for x in X])
    tNN.train(X, y, epochs=100, verb=verb, re_init=10, re_init_d=20)
    y_predict = tNN.predict(X)
    if verb > 0:
        print y_predict
        print np.mean(np.square(y_predict - y))
    return np.mean(np.square(y_predict - y))
# test_xor3()


def test_sine(verb=0, netstruc='feed_forward'):
    total_error = 0
    # testing out varying regularization. Trashy
    for r in [4]:
        tNN = NNn([1, 1000, 1], perf_fcn='mse', trans='tanh',
                  reg=10 ** -8, netstruc=netstruc)
        # tNN.Lmap[0][-1] = tNN.Lmap[0][-2] = tNN.Lmap[1][-1] = 0
        X = np.array([np.array([x]) for x in np.arange(0, 2 * np.pi, .01)])
        y = np.array([np.sin(x) + np.random.randn(x.shape[0]) * .1
                      for x in X])
        if verb > 0:
            start = time.time()
        tNN.train(X, y, epochs=20, verb=verb, batch_type=2)
        if verb > 0:
            time2 = time.time()
            print "time length is: " + str(time2 - start)
        X_test = np.arange(0, 2 * np.pi, .009)
        y_test = np.sin(X_test)
        plt.plot(X, y)
        plt.plot(X_test, y_test)
        y_predict = tNN.predict(X_test)
        if y_test.ndim == 1:
            y_predict = y_predict.flatten()
        if verb > 0:
            print r, np.mean(np.square(y_predict - y_test))
        total_error += np.mean(np.square(y_predict - y_test))
        # for idx in xrange(len(y_predict)):
        #    print y_predict[idx], y_test[idx]
        if verb > 0:
            plt.plot(X_test, y_predict)
            plt.show()
    if verb > 0:
        print total_error
    return total_error
# test_sine(1)


def test_1times2squared(verb=0, netstruc='feed_forward'):
    np.seterr(all='raise')
    tNN = NNn([2, 200, 200, 1], perf_fcn='mse', netstruc=netstruc,
              trans='sigmoid', reg=10 ** (-5))
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 0],
                  [1, 1],
                  [1, 2],
                  [2, 0],
                  [2, 1],
                  [2, 2]])
    y = np.array([np.array([x[0] * x[1] ** 2]) for x in X])
    tNN.train(X, y, epochs=500, verb=verb, re_init=10, re_init_d=50)
    y_predict = tNN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        print cat([X.T, np.atleast_2d(y).T, y_predict.T]).T
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print 'error: ' + str(np.mean(np.square(y_predict - y)))
        print 'weights values squared: ' + \
              str(np.mean([np.mean([np.mean(np.square(c))
              for c in x]) for x in tNN.weights]))
    return np.mean(np.square(y_predict - y))
# test_1times2squared(1)


def test_1times2(verb=0, netstruc='feed_forward'):
    np.seterr(all='raise')
    tNN = NNn([2, 100, 100, 1], perf_fcn='mse', netstruc=netstruc,
              trans='tanh', reg=10 ** (-10))
    tNN.trans[-1] = sigmoid
    X = np.array([[0, 0],
                  [0, 1],
                  [0, 2],
                  [1, 0],
                  [1, 1],
                  [1, 2],
                  [2, 0],
                  [2, 1],
                  [2, 2]])
    y = np.array([np.array([x[0] * x[1]]) for x in X])
    tNN.train(X, y, epochs=100, verb=verb)
    y_predict = tNN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        print cat([X.T, np.atleast_2d(y).T, y_predict.T]).T
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print np.mean(np.square(y_predict - y))
    return np.mean(np.square(y_predict - y))
# test_1times2()


def test_1times2cubed(verb=0, netstruc='feed_forward'):
    np.seterr(all='raise')
    tNN = NNn([2, 1000, 1], perf_fcn='mse', trans='sigmoid', reg=10 ** (-10),
              netstruc='feed_forward')
    X = np.array([np.array([x, c]) for x in np.arange(0, 2, .2)
                 for c in np.arange(0, 2, .2)])
    y = np.array([np.array([x[0] * x[1] ** 3]) for x in X])
    tNN.train(X, y, epochs=100, verb=verb, re_init=10, re_init_d=50)
    y_predict = tNN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        for row in cat([X.T, np.atleast_2d(y).T, y_predict.T]).T:
            print np.around(row, 2)
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print 'error: ' + str(np.mean(np.square(y_predict - y)))
        print 'weights values squared: ' + \
              str(np.mean([np.mean([np.mean(np.square(c))
              for c in x]) for x in tNN.weights]))
    return np.mean(np.square(y_predict - y))
# print test_1times2cubed(1)


def test1plus21minus2(verb=0, netstruc='feed_forward'):
    np.seterr(all='raise')
    tNN = NNn([2, 1000, 2], perf_fcn='mse', trans='tanh', reg=10 ** (-3),
              netstruc=netstruc)
    X = np.array([np.array([x, c]) for x in np.arange(-1, 2, 1)
                 for c in np.arange(-1, 2, 1)])
    y = np.array([np.array([x[0] + x[1], x[0] - x[1]]) for x in X])
    # y = np.array([np.array([x[0] + x[1]]) for x in X])
    tNN.train(X, y, epochs=100, verb=verb)
    y_predict = tNN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        for row in cat([X.T, np.atleast_2d(y).T, y_predict.T]).T:
            print np.around(row, 1)
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print 'error: ' + str(np.mean(np.square(y_predict - y)))
        print 'weights values squared: ' + \
              str(np.mean([np.mean([np.mean(np.square(c)) for c in x])
              for x in tNN.weights]))
    return np.mean(np.mean(np.square(y_predict - y)))
# print test1plus21minus2()


# Inverse still has some trouble. I don't know what NN's seem to
# struggle to find this function.
# I do not believe it has local minima but might be a very flat region
def testinv(verb=0, netstruc='cascade'):
    np.seterr(all='raise')
    tNN = NNn([1, 500, 500, 500, 1], perf_fcn='mse',
              trans='sigmoid', reg=10 ** -10, netstruc=netstruc)
    X = np.array([np.array([x]) for x in np.arange(.1, 2, .1)
                 if abs(x) > 10 ** -8])
    y = np.array([np.array([1 / x[0]]) for x in X])
    tNN.train(X, y, epochs=100, verb=verb, re_init=10, re_init_d=50)
    y_predict = tNN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        for row in cat([X.T, np.atleast_2d(y).T, y_predict.T]).T:
            print np.around(row, 2)
        print 'error: ' + str(np.mean(np.square(y_predict - y)))
        print 'weights values squared: ' + \
              str(np.mean([np.mean([np.mean(np.square(c)) for c in x])
              for x in tNN.weights]))
    return np.mean(np.square(y_predict - y))
# testinv(1)


def test_NN(fcn, max_err=.1, num_retries=3, **kwargs):
    err_arr = []
    for idx in range(num_retries):
        err = fcn(**kwargs)
        err_arr.append(err)
        if err < max_err:
            return err, idx
        else:
            print "Warning. Function failed ", str(idx + 1), \
                " times. ", fcn, " ", err_arr, " ", kwargs
    print "error. Function failed: ", fcn, " ", err_arr, " ", kwargs
    sys.exit(0)


def regression_test(**kwargs):
    np.random.seed(12345)
    xor_error = test_NN(test_xor, re_init=5, **kwargs)
    print "xor squared error: " + str(np.around(xor_error, 4))
    xor3_error = test_NN(test_xor3, max_err=.4, **kwargs)
    print "xor3_squared error: " + str(np.around(xor3_error, 4))
    sine_error = test_NN(test_sine, max_err=.5, **kwargs)
    print "sine_squared error: " + str(np.around(sine_error, 4))
    test_1t2s_error = test_NN(test_1times2squared, max_err=1, **kwargs)
    print "test_1t2s_squared error: " + str(np.around(test_1t2s_error, 4))
    test_1t2_error = test_NN(test_1times2, **kwargs)
    print "test_1t2_squared error: " + str(np.around(test_1t2_error, 4))
    test_1t2c_error = test_NN(test_1times2cubed, max_err=.5, **kwargs)
    print "test_1t2c_squared error: " + str(np.around(test_1t2c_error, 4))
    test_1p21m2_error = test_NN(test1plus21minus2, **kwargs)
    print "test_1p21m2_squared error: " + str(np.around(test_1p21m2_error, 4))
    test_inv_error = test_NN(testinv, max_err=2, **kwargs)
    print "test_inv_squared error: " + str(np.around(test_inv_error, 4))
regression_test(verb=0, netstruc='cascade')
