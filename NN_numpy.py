# The following is a Neural Network class using Numpy.
# It should test clean using flake8 for all coding
# standards except for E128 because that requried too much
# indenting for visual purposes.
# Ex: flake8 NN_numpy.py --ignore E128



import numpy as np
import matplotlib.pyplot as plt
import sys
import time
cat = np.concatenate

# names of batch training types. I forget how to do this in python.
solo = -1
group = 0
# gb means global best
# LR means learning rate
# Lmap means layer map
# pt replaced future. it means the idx'd array points to the future layer


def softmax(w, t=1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


def softplus(x):
    # print np.log(1 + np.exp(x))
    return np.log(1 + np.exp(np.clip(x, -50, 50)))


def softplus_p(x):
    # print 1.0 / (1.0 + np.exp(-x))
    return 1.0 / (1.0 + np.exp(np.clip(-x, -50, 50)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -50, 50)))


def sigmoid_p(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_p(x):
    return 1.0 - x ** 2


def cross_entropy(y, a):
    return -1 * y * np.log(a + (10 ** -100)) \
        - (1 - y) * np.log(1 - a + (10 ** -100))


def cross_entropy_p(y, a):
    return (y - a)  # * np.abs(y - a) ** 2


def mse(y, a):
    return np.sum(np.square(y - a))


def mse_p(y, a):
    return (y - a) * np.abs(y - a) ** 1


def copy_weights(weights):
    return [[np.array([z for z in w]) if (len(w) != 1 or w[0] != 0)
             else [0] for w in x] for x in weights]


def zero_weights(weights):
    return [[np.array([np.zeros_like(z) for z in w]) if
             (len(w) != 1 or w[0] != 0) else [0] for w in x] for x in weights]


class NNn:

    def __init__(self, layers, trans='sigmoid', perf_fcn='cross_entropy',
                 reg=0, netstruc='feed_forward'):
        self.norm = np.ones(layers[-1])
        self.size = layers
        self.weights = []
        self.act_vals = []
        if netstruc == 'feed_forward':
            self.Lmap = cat([np.zeros([1, len(layers)]),
                             cat([np.eye(len(layers) - 1),
                             np.zeros([1, len(layers) - 1])]).T]).T
        elif netstruc == 'cascade':
            self.Lmap = \
                cat([cat([np.zeros([1, 1 + row]),
                np.ones([1, len(layers) - row - 1])], axis=1)
                    for row in xrange(len(layers))])
        else:
            print 'net structure does not exist: %s', netstruc
            sys.exit(0)
        self.init_weights()
        for idx in np.arange(len(layers) - 1):
            self.act_vals.append(np.empty([layers[idx]]))
        self.act_vals.append(np.empty([layers[-1]]))
        self.trans = [0] * (len(layers))
        self.trans_p = [0] * (len(layers))
        if isinstance(trans, str):
            for idx in xrange(1, len(layers)):
                if trans == 'sigmoid':
                    self.trans[idx] = sigmoid
                    self.trans_p[idx] = sigmoid_p
                elif trans == 'tanh':
                    self.trans[idx] = tanh
                    self.trans_p[idx] = tanh_p
                elif trans == 'softplus':
                    self.trans[idx] = softplus
                    self.trans_p[idx] = softplus_p
        elif isinstance(trans, list):
            if len(trans) < len(layers) - 1:
                print 'wrong number of transfer functions listed'
                sys.exit(0)
            for idx in xrange(1, len(layers)):
                if trans[idx] == 'sigmoid':
                    self.trans[idx] = sigmoid
                    self.trans_p[idx] = sigmoid_p
                elif trans[idx] == 'tanh':
                    self.trans[idx] = tanh
                    self.trans_p[idx] = tanh_p
                elif trans[idx] == 'softplus':
                    self.trans[idx] = softplus
                    self.trans_p[idx] = softplus_p
        if perf_fcn == 'cross_entropy' and self.trans[-1] != sigmoid:
            print 'cross_entropy requires sigmoid trans to output layer'
            sys.exit(0)
        if perf_fcn == 'cross_entropy':
            self.perf_fcn = cross_entropy
            self.perf_fcn_p = cross_entropy_p
        elif perf_fcn == 'mse':
            self.perf_fcn = mse
            self.perf_fcn_p = mse_p
        self.reg = reg
        self.deltas = np.copy(self.act_vals)
        self.best_perf = np.inf

    def init_weights(self, div=2):
        self.weights = [[[0] for _ in xrange(len(self.Lmap))]
                        for _ in xrange(len(self.Lmap[0]))]
        for r_idx in xrange(len(self.weights)):
            for c_idx in xrange(len(self.weights[r_idx])):
                if self.Lmap[r_idx][c_idx]:
                    self.weights[r_idx][c_idx] = \
                        np.random.randn(self.size[r_idx] + 1,
                        self.size[c_idx]) / div

    def run_epoch(self, X, y):
        perm = np.random.permutation(self.samples)
        for batch_idx in np.arange(self.num_batches):
            batch = perm[self.batch_size * batch_idx:self.batch_size *
                         (batch_idx + 1)]
            if batch_idx == self.num_batches - 1:
                batch = perm[self.batch_size * batch_idx:]
            self.batch_deltas = zero_weights(self.weights)
            for i in batch:
                self.train_one_sample(X[i], y[i])
            for row in xrange(len(self.weights)):
                for col in xrange(len(self.weights[0])):
                    if self.Lmap[row][col]:
                        self.weights[row][col] += self.batch_deltas[row][col]
            self.test_batch(X, y)

    def train_one_sample(self, Xi, yi):
        self.act_vals[0] = Xi
        self.propogate_forward()
        self.deltas[-1] = self.perf_fcn_p(
            yi, self.act_vals[-1]) / self.samples
        for idx in np.arange(len(self.deltas) - 2, 0, -1):
            self.deltas[idx] = np.sum(
                self.deltas[pt].dot(self.weights[idx][pt][:-1].T *
                self.trans_p[idx](self.act_vals[idx]))
                for pt in xrange(len(self.Lmap)) if self.Lmap[idx][pt])
        for row in xrange(len(self.weights)):
            for col in xrange(len(self.weights[0])):
                if self.Lmap[row][col]:
                    self.batch_deltas[row][col] += self.LR * (
                        np.atleast_2d(cat((self.act_vals[row], [1]))).T
                        .dot(np.atleast_2d(self.deltas[col])) -
                        self.reg * self.weights[row][col] *
                        (col < len(self.weights[0]) - 1)) / self.size[row]

    def test_batch(self, X, y):
        batch_perf = 0
        for i in xrange(len(X)):
            self.act_vals[0] = X[i]
            self.propogate_forward()
            batch_perf += self.perf_fcn(y[i], self.act_vals[-1]) / self.samples
        if batch_perf < self.best_perf * 1.000000001:
            self.LR *= 1.05
            self.best_perf = batch_perf
            self.best_weights = copy_weights(self.weights)
        else:
            self.LR *= .7
            self.weights = copy_weights(self.best_weights)
        if self.verb > 0:
            print batch_perf, self.LR

    def propogate_forward(self):
        for idx in range(1, len(self.Lmap[0])):
            self.act_vals[idx] = self.trans[idx](
                np.sum(np.dot(cat([self.act_vals[prev], [1]]),
                self.weights[prev][idx])for prev in xrange(len(self.Lmap))
                    if self.Lmap[prev][idx]))

    def train(self, X, y, LR=1, epochs=10,
              batch_type=group, verb=0, re_init=3, re_init_d=10):
        self.samples = X.shape[0]
        self.verb = verb
        self.LR = LR
        self.norm = np.amax(np.abs(np.vstack((self.norm, y * 1.1))))
        y = y / self.norm
        if X.ndim < 2:
            X = np.atleast_2d(X).T
        if X.shape[1] != self.size[0]:
            print "input size %d, needed %d" % (X.shape[1], self.size[0])
            return
        if batch_type == -1:
            self.num_batches = self.samples
            self.batch_size = 1
        if batch_type == 0:
            self.num_batches = 1
            self.batch_size = self.samples
        if batch_type > 0:
            self.num_batches = np.floor(self.samples ** (1.0 / batch_type))
            self.batch_size = np.ceil(self.samples / self.num_batches)
        self.best_weights = copy_weights(self.weights)

        self.gb_weights = copy_weights(self.best_weights)
        self.gb_perf = np.copy(self.best_perf)
        self.gb_LR = self.LR
        for init_idx in xrange(re_init):
            self.init_weights(2 * 10 ** ((1.0 * init_idx) / re_init))
            self.best_perf = np.inf
            self.LR = LR
            for _ in xrange(re_init_d):
                self.run_epoch(X, y)
                if self.best_perf < self.gb_perf:
                    self.gb_perf = self.best_perf
                    self.gb_weights = copy_weights(self.best_weights)
                    self.gb_LR = self.LR
        self.best_perf = self.gb_perf
        self.best_weights = copy_weights(self.gb_weights)
        self.weights = copy_weights(self.best_weights)
        self.LR = self.gb_LR
        for _ in range(epochs):
            self.run_epoch(X, y)
        self.weights = copy_weights(self.best_weights)

    def predict(self, x):
        self.samples = x.shape[0]
        output = np.empty([self.samples, self.act_vals[-1].shape[0]])
        for d_idx in np.arange(self.samples):
            self.act_vals[0] = np.atleast_1d(x[d_idx])
            self.propogate_forward()
            output[d_idx] = self.act_vals[-1]
        return output * self.norm


def test_xor(verb=0, re_init=10):
    tNN = NNn([2, 100, 100, 1], reg=10 ** -8)
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


def test_xor3(verb=0):
    tNN = NNn([3, 500, 500, 500, 1], netstruc='cascade', reg=10 * 8 - 10)
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


def test_sine(verb=0):
    total_error = 0
    # testing out varying regularization. Trashy
    for r in [4]:
        tNN = NNn([1, 1000, 1], perf_fcn='mse', trans='tanh', reg=10 ** -8)
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


def test_1times2squared(verb=0):
    np.seterr(all='raise')
    tNN = NNn([2, 200, 200, 1], perf_fcn='mse',
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


def test_1times2(verb=0):
    np.seterr(all='raise')
    tNN = NNn([2, 100, 100, 1], perf_fcn='mse',
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


def test_1times2cubed(verb=0):
    np.seterr(all='raise')
    tNN = NNn([2, 1000, 1], perf_fcn='mse', trans='sigmoid', reg=10 ** (-10))
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


def test1plus21minus2(verb=0):
    np.seterr(all='raise')
    tNN = NNn([2, 1000, 2], perf_fcn='mse', trans='tanh', reg=10 ** (-3))
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
def testinv(verb=0):
    np.seterr(all='raise')
    tNN = NNn([1, 500, 500, 500, 1], perf_fcn='mse',
              trans='sigmoid', reg=10 ** -10, netstruc='cascade')
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


def regression_test(verb=0):
    np.random.seed(12345)
    xor_error = test_NN(test_xor, verb=verb, re_init=5)
    print "xor squared error: " + str(np.around(xor_error, 4))
    xor3_error = test_NN(test_xor3, verb=verb, max_err=.4)
    print "xor3_squared error: " + str(np.around(xor3_error, 4))
    sine_error = test_NN(test_sine, verb=verb, max_err=.5)
    print "sine_squared error: " + str(np.around(sine_error, 4))
    test_1t2s_error = test_NN(test_1times2squared, verb=verb, max_err=1)
    print "test_1t2s_squared error: " + str(np.around(test_1t2s_error, 4))
    test_1t2_error = test_NN(test_1times2, verb=verb)
    print "test_1t2_squared error: " + str(np.around(test_1t2_error, 4))
    test_1t2c_error = test_NN(test_1times2cubed, verb=verb, max_err=.5)
    print "test_1t2c_squared error: " + str(np.around(test_1t2c_error, 4))
    test_1p21m2_error = test_NN(test1plus21minus2, verb=verb)
    print "test_1p21m2_squared error: " + str(np.around(test_1p21m2_error, 4))
    test_inv_error = test_NN(testinv, verb=verb, max_err=2)
    print "test_inv_squared error: " + str(np.around(test_inv_error, 4))
regression_test(verb=0)
