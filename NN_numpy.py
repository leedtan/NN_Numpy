import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import random

# names of batch training types. I forget how to do this in python.
solo = -1
group = 0

def softmax(w, t=1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist
def softplus(x):
    # print np.log(1 + np.exp(x))
    return np.log(1 + np.exp(np.clip(x, -50, 50)))
def softplus_prime(x):
    # print 1.0 / (1.0 + np.exp(-x))
    return 1.0 / (1.0 + np.exp(np.clip(-x, -50, 50)))
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -50, 50)))
def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
def tanh(x):
    return np.tanh(x)
def tanh_prime(x):
    return 1.0 - x ** 2
def cross_entropy(y, a):
    return -1 * y * np.log(a + (10 ** -100)) - (1 - y) * np.log(1 - a + (10 ** -100))
def cross_entropy_prime(y, a):
    return (y - a)  # * np.abs(y - a) ** 2
def mse(y, a):
    return np.sum(np.square(y - a))
def mse_prime(y, a):
    return (y - a) * np.abs(y - a) ** 1
def copy_weights(weights):
    return [[np.array([z for z in w]) if (len(w) != 1 or w[0] != 0) else [0] for w in x] for x in weights]
def zero_weights(weights):
    return [[np.array([np.zeros_like(z) for z in w]) if (len(w) != 1 or w[0] != 0) else [0] for w in x] for x in weights]
class Lee_NN:
    def __init__(self, layers, trans_fcn='sigmoid', perf_fcn='cross_entropy', \
            reg=0, connections='feed_forward'):
        self.normalization = np.ones(layers[-1]);
        self.size = layers
        self.weights = []
        self.act_vals = []
        if connections == 'feed_forward':
            self.connections = np.concatenate([np.zeros([1, len(layers)]), np.concatenate(\
                    [np.eye(len(layers) - 1), np.zeros([1, len(layers) - 1])]).T]).T
        elif connections == 'cascade':
            self.connections = np.concatenate([np.concatenate([\
                    np.zeros([1, 1 + row]), np.ones([1, len(layers) - row - 1])], axis=1) \
                    for row in xrange(len(layers))])
        else:
            print 'connection type does not exist: %s', connections
            sys.exit(0)
        self.init_weights()
        for idx in np.arange(len(layers) - 1):
            # self.weights.append(np.random.randn(layers[idx] + 1, layers[idx + 1]))
            self.act_vals.append(np.empty([layers[idx]]))
        self.act_vals.append(np.empty([layers[-1]]))
        self.trans_fcn = [0] * (len(layers))
        self.trans_fcn_prime = [0] * (len(layers))
        if type(trans_fcn) == type('a'):
            for idx in xrange(1, len(layers)):
                if trans_fcn == 'sigmoid':
                    self.trans_fcn[idx] = sigmoid
                    self.trans_fcn_prime[idx] = sigmoid_prime
                elif trans_fcn == 'tanh':
                    self.trans_fcn[idx] = tanh
                    self.trans_fcn_prime[idx] = tanh_prime
                elif trans_fcn == 'softplus':
                    self.trans_fcn[idx] = softplus
                    self.trans_fcn_prime[idx] = softplus_prime
        elif type(trans_fcn) == type([]):
            if len(trans_fcn) < len(layers) - 1:
                print 'wrong number of transfer functions listed'
                sys.exit(0)
            for idx in xrange(1, len(layers)):
                if trans_fcn[idx] == 'sigmoid':
                    self.trans_fcn[idx] = sigmoid
                    self.trans_fcn_prime[idx] = sigmoid_prime
                elif trans_fcn[idx] == 'tanh':
                    self.trans_fcn[idx] = tanh
                    self.trans_fcn_prime[idx] = tanh_prime
                elif trans_fcn[idx] == 'softplus':
                    self.trans_fcn[idx] = softplus
                    self.trans_fcn_prime[idx] = softplus_prime
        if perf_fcn == 'cross_entropy' and self.trans_fcn[-1] != sigmoid:
            print 'cross_entropy requires sigmoid transfer fcn to output layer'
            sys.exit(0)
        if perf_fcn == 'cross_entropy':
            self.perf_fcn = cross_entropy
            self.perf_fcn_prime = cross_entropy_prime
        elif perf_fcn == 'mse':
            self.perf_fcn = mse
            self.perf_fcn_prime = mse_prime
        self.reg = reg
        self.deltas = np.copy(self.act_vals)
        self.best_perf = np.inf
    def init_weights(self, div=2):
        self.weights = [[[0] for _ in xrange(len(self.connections))] \
                        for _ in xrange(len(self.connections[0]))]
        for r_idx in xrange(len(self.weights)):
            for c_idx in xrange(len(self.weights[r_idx])):
                if self.connections[r_idx][c_idx]:
                    self.weights[r_idx][c_idx] = np.random.randn(self.size[r_idx] + 1, self.size[c_idx]) / div
    def run_epoch(self, X, y):
        perm = np.random.permutation(X.shape[0])
        for batch_idx in np.arange(self.num_batches):
            batch = perm[self.batch_size * batch_idx:self.batch_size * (batch_idx + 1)]
            if batch_idx == self.num_batches - 1:
                batch = perm[self.batch_size * batch_idx:]
            self.batch_deltas = zero_weights(self.weights)  # np.array([np.zeros_like(x) for x in self.deltas])
            for i in batch:
                self.act_vals[0] = X[i]
                self.propogate_forward()
                self.deltas[-1] = self.perf_fcn_prime(y[i], self.act_vals[-1]) / X.shape[0]
                for idx in np.arange(len(self.deltas) - 2, 0, -1):
                    # deltas[prev] = delta[future] * weight * trans_prime(act_val[prev])
                    self.deltas[idx] = np.sum(
                        self.deltas[future].dot(self.weights[idx][future][:-1].T\
                        *self.trans_fcn_prime[idx](self.act_vals[idx])) \
                        for future in xrange(len(self.connections)) if self.connections[idx][future] == True)
                # self.batch_deltas += self.deltas
                for row in xrange(len(self.weights)):
                    for col in xrange(len(self.weights[0])):
                        if self.connections[row][col]:
                            self.batch_deltas[row][col] += self.learning_rate * (\
                                np.atleast_2d(np.concatenate((self.act_vals[row], [1]))).T\
                                .dot(np.atleast_2d(self.deltas[col])) - \
                                self.reg * self.weights[row][col] * (col < len(self.weights[0]) - 1)) / self.size[row]
            for row in xrange(len(self.weights)):
                for col in xrange(len(self.weights[0])):
                    if self.connections[row][col]:
                        self.weights[row][col] += self.batch_deltas[row][col]
            self.test_batch(X, y)
    def test_batch(self, X, y):
        batch_perf = 0;
        for i in xrange(len(X)):
            self.act_vals[0] = X[i]
            self.propogate_forward()
            batch_perf += self.perf_fcn(y[i], self.act_vals[-1]) / X.shape[0];
        if batch_perf < self.best_perf * 1.000000001:
            self.learning_rate *= 1.05
            self.best_perf = batch_perf
            self.best_weights = copy_weights(self.weights)
        else:
            self.learning_rate *= .7
            self.weights = copy_weights(self.best_weights)
        if self.verb > 0:
            print batch_perf, self.learning_rate

    def propogate_forward(self):
        for idx in range(1, len(self.connections[0])):
            self.act_vals[idx] = self.trans_fcn[idx](\
                np.sum(np.dot(np.concatenate([self.act_vals[prev], [1]]), self.weights[prev][idx])\
                for prev in xrange(len(self.connections)) if self.connections[prev][idx] == True))
    def train(self, X, y, learning_rate=1, epochs=10, batch_type=group, verb=0, re_init_tries=3, re_init_depth=10):
        self.verb = verb
        self.learning_rate = learning_rate
        self.normalization = np.amax(np.abs(np.vstack((self.normalization, y * 1.1))))
        y = y / self.normalization
        if X.ndim < 2:
            X = np.atleast_2d(X).T
        if X.shape[1] != self.size[0]:
            print "input size %d, needed %d" % (X.shape[1], self.size[0])
            return
        if batch_type == -1:
            self.num_batches = X.shape[0]
            self.batch_size = 1  # X.shape[0]
        if batch_type == 0:
            self.num_batches = 1
            self.batch_size = X.shape[0]
        if batch_type > 0:
            self.num_batches = np.floor(X.shape[0] ** (1.0 / batch_type))
            self.batch_size = np.ceil(X.shape[0] / self.num_batches)
        self.best_weights = copy_weights(self.weights)

        self.global_best_weights = copy_weights(self.best_weights)
        self.global_best_perf = np.copy(self.best_perf)
        self.global_best_learning_rate = self.learning_rate
        for init_idx in xrange(re_init_tries):
            self.init_weights(2 * 10 ** ((1.0 * init_idx) / re_init_tries))
            self.best_perf = np.inf
            self.learning_rate = learning_rate
            for _ in xrange(re_init_depth):
                self.run_epoch(X, y)
                if self.best_perf < self.global_best_perf:
                    self.global_best_perf = self.best_perf
                    self.global_best_weights = copy_weights(self.best_weights)
                    self.global_best_learning_rate = self.learning_rate
        self.best_perf = self.global_best_perf
        self.best_weights = copy_weights(self.global_best_weights)
        self.weights = copy_weights(self.best_weights)
        self.learning_rate = self.global_best_learning_rate
        for _ in range(epochs):
            self.run_epoch(X, y)
        self.weights = copy_weights(self.best_weights)
    def predict(self, x):
        output = np.empty([x.shape[0], self.act_vals[-1].shape[0]])
        for d_idx in np.arange(x.shape[0]):
            self.act_vals[0] = np.atleast_1d(x[d_idx])
            self.propogate_forward()
            output[d_idx] = self.act_vals[-1]
        return output * self.normalization
def test_xor(verb=0):
    trial_NN = Lee_NN([2, 100, 100, 1], reg=10 ** -8);
    X = np.array([[0, 0], \
                  [0, 1], \
                  [1, 0], \
                  [1, 1]])
    # X = np.tile(X, (100, 1))
    y = np.array([np.array([x[0] ^ x[1]]) for x in X])
    trial_NN.train(X, y, epochs=1000, verb=verb, batch_type=group, re_init_tries=10, re_init_depth=100)
    y_predict = trial_NN.predict(X)
    if verb > 0:
        print np.around(y_predict, 2)
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print np.mean(np.square(y_predict - y))
    return np.mean(np.square(y_predict - y))
# test_xor()

def test_xor3(verb=0):
    trial_NN = Lee_NN([3, 500, 500, 500, 1], connections='cascade', reg=10 * 8 - 10);
    trial_NN.connections[0][-1] = 0
    X = np.array([[0, 0, 0], \
                  [0, 1, 0], \
                  [1, 0, 0], \
                  [1, 1, 0], \
                  [0, 0, 1], \
                  [0, 1, 1], \
                  [1, 0, 1], \
                  [1, 1, 1]])
    y = np.array([np.array([x[0] ^ x[1] ^ x[2]]) for x in X])
    trial_NN.train(X, y, epochs=100, verb=verb, re_init_tries=10, re_init_depth=20)
    y_predict = trial_NN.predict(X)
    if verb > 0:
        print y_predict
        print np.mean(np.square(y_predict - y))
    return np.mean(np.square(y_predict - y))
# test_xor3()


def test_sine(verb=0):
    total_error = 0
    # testing out varying regularization. Trashy
    for r in [4]:
        trial_NN = Lee_NN([1, 1000, 1], perf_fcn='mse', trans_fcn='tanh', reg=10 ** -8);
        # trial_NN.connections[0][-1] = trial_NN.connections[0][-2] = trial_NN.connections[1][-1] = 0
        X = np.array([np.array([x]) for x in np.arange(0, 2 * np.pi, .01)])
        y = np.array([np.sin(x) + np.random.randn(x.shape[0]) * .1 for x in X])
        if verb > 0:
            start = time.time()
        trial_NN.train(X, y, epochs=20, verb=verb, batch_type=2)
        if verb > 0:
            time2 = time.time()
            print "time length is: " + str(time2 - start)
        X_test = np.arange(0, 2 * np.pi, .009)
        y_test = np.sin(X_test)
        plt.plot(X, y)
        plt.plot(X_test, y_test);
        y_predict = trial_NN.predict(X_test);
        if y_test.ndim == 1:
            y_predict = y_predict.flatten()
        if verb > 0:
            print r, np.mean(np.square(y_predict - y_test))
        total_error += np.mean(np.square(y_predict - y_test))
        # for idx in xrange(len(y_predict)):
        #    print y_predict[idx], y_test[idx]
        if verb > 0:
            plt.plot(X_test, y_predict);
            plt.show()
    if verb > 0:
        print total_error
    return total_error
# test_sine()
def test_1times2squared(verb=0):
    np.seterr(all='raise')
    trial_NN = Lee_NN([2, 200, 200, 1], perf_fcn='mse', trans_fcn='sigmoid', reg=10 ** (-5));
    X = np.array([[0, 0], \
                  [0, 1], \
                  [0, 2], \
                  [1, 0], \
                  [1, 1], \
                  [1, 2], \
                  [2, 0], \
                  [2, 1], \
                  [2, 2]])
    # y = np.array([0, 0, 0, 0, 1, 4, 0, 2, 8])
    y = np.array([np.array([x[0] * x[1] ** 2]) for x in X])
    trial_NN.train(X, y, epochs=500, verb=verb, re_init_tries=10, re_init_depth=50)
    y_predict = trial_NN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        print np.concatenate([X.T, np.atleast_2d(y).T, y_predict.T]).T
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print 'error: ' + str(np.mean(np.square(y_predict - y)))
        print 'weights values squared: ' + str(np.mean([np.mean([np.mean(np.square(y)) for y in x]) for x in trial_NN.weights]))
    return np.mean(np.square(y_predict - y))
# test_1times2squared(1)
def test_1times2(verb=0):
    np.seterr(all='raise')
    trial_NN = Lee_NN([2, 100, 100, 1], perf_fcn='mse', trans_fcn='tanh', reg=10 ** (-10));
    trial_NN.trans_fcn[-1] = sigmoid
    X = np.array([[0, 0], \
                  [0, 1], \
                  [0, 2], \
                  [1, 0], \
                  [1, 1], \
                  [1, 2], \
                  [2, 0], \
                  [2, 1], \
                  [2, 2]])
    y = np.array([np.array([x[0] * x[1]]) for x in X])
    trial_NN.train(X, y, epochs=100, verb=verb)
    y_predict = trial_NN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        print np.concatenate([X.T, np.atleast_2d(y).T, y_predict.T]).T
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print np.mean(np.square(y_predict - y))
    return np.mean(np.square(y_predict - y))
# test_1times2()

# doesnt work that well:
def test_1times2cubed(verb=0):
    np.seterr(all='raise')
    trial_NN = Lee_NN([2, 1000, 1], perf_fcn='mse', trans_fcn='sigmoid', reg=10 ** (-10));
    X = np.array([np.array([x, y]) for x in np.arange(0, 2, .2) for y in np.arange(0, 2, .2)])
    y = np.array([np.array([x[0] * x[1] ** 3]) for x in X])
    trial_NN.train(X, y, epochs=100, verb=verb, re_init_tries=10, re_init_depth=50)
    y_predict = trial_NN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        for row in np.concatenate([X.T, np.atleast_2d(y).T, y_predict.T]).T:
            print np.around(row, 2)
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print 'error: ' + str(np.mean(np.square(y_predict - y)))
        print 'weights values squared: ' + str(np.mean([np.mean([np.mean(np.square(y)) for y in x]) for x in trial_NN.weights]))
    return np.mean(np.square(y_predict - y))
# print test_1times2cubed(1)
def test1plus21minus2(verb=0):
    np.seterr(all='raise')
    trial_NN = Lee_NN([2, 1000, 2], perf_fcn='mse', trans_fcn='tanh', reg=10 ** (-3));
    X = np.array([np.array([x, y]) for x in np.arange(-1, 2, 1) for y in np.arange(-1, 2, 1)])
    y = np.array([np.array([x[0] + x[1], x[0] - x[1]]) for x in X])
    # y = np.array([np.array([x[0] + x[1]]) for x in X])
    trial_NN.train(X, y, epochs=100, verb=verb)
    y_predict = trial_NN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        for row in np.concatenate([X.T, np.atleast_2d(y).T, y_predict.T]).T:
            print np.around(row, 1)
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > 0:
        print 'error: ' + str(np.mean(np.square(y_predict - y)))
        print 'weights values squared: ' + str(np.mean([np.mean([np.mean(np.square(y)) for y in x]) for x in trial_NN.weights]))
    return np.mean(np.mean(np.square(y_predict - y)))
# print test1plus21minus2()
####Inverse still has some trouble. I don't know what NN's seem to struggle to find this function.
# I do not believe it has local minima but might be a very flat region
def testinv(verb=0):
    np.seterr(all='raise')
    trial_NN = Lee_NN([1, 500, 500, 500, 1], perf_fcn='mse', trans_fcn='sigmoid', reg=10 ** -10, connections='cascade');
    X = np.array([np.array([x]) for x in np.arange(.1, 2, .1) if abs(x) > 10 ** -8])
    y = np.array([np.array([1 / x[0]]) for x in X])
    trial_NN.train(X, y, epochs=100, verb=verb, re_init_tries=10, re_init_depth=50)
    y_predict = trial_NN.predict(X)
    y_predict = np.around(y_predict, 2)
    if verb > 0:
        for row in np.concatenate([X.T, np.atleast_2d(y).T, y_predict.T]).T:
            print np.around(row, 2)
    if verb > 0:
        print 'error: ' + str(np.mean(np.square(y_predict - y)))
        print 'weights values squared: ' + str(np.mean([np.mean([np.mean(np.square(y)) for y in x]) for x in trial_NN.weights]))
    return np.mean(np.square(y_predict - y))
# testinv(1)

def test_until_success(fcn, args, max_err=.1, num_retries=3):
    err_arr = []
    for idx in range(num_retries):
        err = fcn(args);
        err_arr.append(err)
        if err < max_err:
            return err, idx
        else:
            print "Warning. Function failed ", str(idx + 1), " times. ", fcn, " ", err_arr, " ", args
    print "error. Function failed: ", fcn, " ", err_arr, " ", args
    sys.exit(0)



def regression_test(verb=0):
    np.random.seed(12345)
    xor_error = test_until_success(test_xor, args=verb)
    print "xor squared error: " + str(xor_error)
    xor3_error = test_until_success(test_xor3, args=verb, max_err=.4)
    print "xor3_squared error: " + str(xor3_error)
    sine_error = test_until_success(test_sine, args=verb, max_err=.5)
    print "sine_squared error: " + str(sine_error)
    test_1t2s_error = test_until_success(test_1times2squared, args=verb, max_err=1)
    print "test_1t2s_squared error: " + str(test_1t2s_error)
    test_1t2_error = test_until_success(test_1times2, args=verb)
    print "test_1t2_squared error: " + str(test_1t2_error)
    test_1t2c_error = test_until_success(test_1times2cubed, args=verb, max_err=.5)
    print "test_1t2c_squared error: " + str(test_1t2c_error)
    test_1p21m2_error = test_until_success(test1plus21minus2, args=verb)
    print "test_1p21m2_squared error: " + str(test_1p21m2_error)
    test_inv_error = test_until_success(testinv, args=verb, max_err=2)
    print "test_inv_squared error: " + str(test_inv_error)
    # print "%d %d %d %d %d %d %d %d" % (xor_error[0], xor3_error[0], sine_error[0], \
    #        test_1t2s_error[0], test_1t2_error[0], test_1t2c_error[0], test_1p21m2_error[0], test_inv_error[0])
regression_test(verb=0)
