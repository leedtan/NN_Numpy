import numpy as np
import matplotlib.pyplot as plt
import sys

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
    return a - y
def mse(y, a):
    return np.sum(np.square(y - a))
def mse_prime(y, a):
    return y - a
# TODO: implement weights class
def copy_weights(weights):
    return [[np.array([z for z in w]) if (len(w) != 1 or w[0] != 0) else [0] for w in x] for x in weights]
class Lee_NN:
    def __init__(self, layers, trans_fcn='sigmoid', perf_fcn='cross_entropy', \
            reg=0, connections='feed_forward'):
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
        self.weights = [[[0] for _ in xrange(len(self.connections))] \
                        for row in xrange(len(self.connections[0]))]
        for r_idx in xrange(len(self.weights)):
            for c_idx in xrange(len(self.weights[r_idx])):
                if self.connections[r_idx][c_idx]:
                    self.weights[r_idx][c_idx] = np.random.randn(layers[r_idx] + 1, layers[c_idx])
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
        if True:  # if type(reg) in [type(1), type(1.)]:
            self.reg = reg
        self.deltas = np.copy(self.act_vals)
    def propogate_forward(self):
        for idx in range(1, len(self.connections[0])):
            self.act_vals[idx] = self.trans_fcn[idx](\
                np.sum(np.dot(np.concatenate([self.act_vals[prev], [1]]), self.weights[prev][idx])\
                for prev in xrange(len(self.connections)) if self.connections[prev][idx] == True))
    def train(self, X, y, learning_rate=1, epochs=100, batch_type=-1, verb=0):
        if False:
            for idx in xrange(len(y)):
                if y[idx] == 0:
                    y[idx] = -1
                if y[idx] == 1:
                    y[idx] = 2
        if X.ndim < 2:
            X = np.atleast_2d(X).T
        if X.shape[1] != self.size[0]:
            print "input size %d, needed %d" % (X.shape[1], self.size[0])
            return
        if batch_type == -1:
            num_batches = 1
            batch_size = X.shape[0]
        best_perf = np.inf
        best_weights = copy_weights(self.weights)
        for _ in range(epochs):
            perm = np.random.permutation(X.shape[0])
            for batch_idx in np.arange(num_batches):
                batch_perf = 0;
                batch = perm[batch_size * batch_idx:batch_size * (batch_idx + 1)]
                if batch_idx == num_batches - 1:
                    batch = perm[batch_size * batch_idx:]
                for i in batch:
                    # i = np.random.randint(X.shape[0])
                    self.act_vals[0] = X[i]
                    self.propogate_forward()
                    self.deltas[-1] = self.perf_fcn_prime(y[i], self.act_vals[-1]) / batch_size  # self.deltas[-1] = self.perf_fcn_prime(y[i], self.act_vals[-1])/X.shape[0]
                    # we need to begin at the second to last layer
                    # (a layer before the output layer)
                    for idx in np.arange(len(self.deltas) - 2, 0, -1):
                        # weights and act_vals are indexed deltas + 1 because there is no deltas[1st nodes]
                        self.deltas[idx] = np.sum(
                            self.deltas[future].dot(self.weights[idx][future][:-1].T)\
                            *self.trans_fcn_prime[idx](self.act_vals[idx]) \
                            for future in xrange(len(self.connections)) if self.connections[idx][future] == True)
                    for row in xrange(len(self.weights)):
                        for col in xrange(len(self.weights[0])):
                            if self.connections[row][col]:
                                self.weights[row][col] += learning_rate * (\
                                    np.atleast_2d(np.concatenate((self.act_vals[row], [1]))).T\
                                    .dot(np.atleast_2d(self.deltas[col])) - \
                                    self.reg * self.weights[row][col] * (col < len(self.weights[0]) - 1))
            batch_perf = 0;
            for i in xrange(len(X)):
                self.act_vals[0] = X[i]
                self.propogate_forward()
                batch_perf += self.perf_fcn(y[i], self.act_vals[-1]) / X.shape[0];
            if batch_perf < best_perf * 1.000000001:
                learning_rate *= 1.05
                best_perf = batch_perf
                best_weights = copy_weights(self.weights)  # best_weights = [[np.array([z for z in w]) if (len(w) != 1 or w[0] != 0) else [0] for w in x] for x in self.weights]
            else:
                learning_rate *= .7
                self.weights = copy_weights(best_weights)  # self.weights = [[np.array([z for z in w]) if (len(w) != 1 or w[0] != 0) else [0] for w in x] for x in best_weights]
            if verb > 0:
                print batch_perf, learning_rate
        self.weights = copy_weights(best_weights)  # self.weights = [[np.array([z for z in w]) if (len(w) != 1 or w[0] != 0) else [0] for w in x] for x in best_weights]
    def predict(self, x):
        output = np.empty([x.shape[0], self.act_vals[-1].shape[0]])
        for d_idx in np.arange(x.shape[0]):
            self.act_vals[0] = np.atleast_1d(x[d_idx])
            self.propogate_forward()
            output[d_idx] = self.act_vals[-1]
        return output


def test_xor():
    trial_NN = Lee_NN([2, 2, 1], reg=10 ** -8);
    X = np.array([[0, 0], \
                  [0, 1], \
                  [1, 0], \
                  [1, 1]])
    y = np.array([0, 1, 1, 0])
    trial_NN.train(X, y, epochs=10000, verb=1)
    y_predict = trial_NN.predict(X)
    print y_predict
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    print np.sum(np.square(y_predict - y))
# test_xor()

def test_xor3():
    trial_NN = Lee_NN([3, 100, 100, 1], connections='cascade');
    trial_NN.connections[0][-1] = 0
    X = np.array([[0, 0, 0], \
                  [0, 1, 0], \
                  [1, 0, 0], \
                  [1, 1, 0], \
                  [0, 0, 1], \
                  [0, 1, 1], \
                  [1, 0, 1], \
                  [1, 1, 1]])
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    trial_NN.train(X, y, epochs=100)
    y_predict = trial_NN.predict(X)
    print y_predict
    print np.sum(np.square(y_predict - y))
    y_predict
# test_xor3()


def test_sine():
    total_error = 0
    for r in [4] * 5:  # xrange(4, 9):
        trial_NN = Lee_NN([1, 20, 20, 1], perf_fcn='mse', trans_fcn='tanh', reg=10 ** (-1 * r));
        # trial_NN.connections[0][-1] = trial_NN.connections[0][-2] = trial_NN.connections[1][-1] = 0
        X = np.arange(0, 2 * np.pi, .01)
        y = np.sin(X) + np.random.randn(X.shape[0]) * 0
        trial_NN.train(X, y / 2, epochs=40)
        X_test = np.arange(0, 2 * np.pi, .009)
        y_test = np.sin(X_test)
        plt.plot(X, y)
        plt.plot(X_test, y_test);
        y_predict = trial_NN.predict(X_test) * 2;
        if y_test.ndim == 1:
            y_predict = y_predict.flatten()
        print r, np.sum(np.square(y_predict - y_test))
        total_error += np.sum(np.square(y_predict - y_test))
        # for idx in xrange(len(y_predict)):
        #    print y_predict[idx], y_test[idx]
        plt.plot(X_test, y_predict);
        plt.show()
    # non-cascade: 500, cascade: 270 FF: 900 2270 cascade
    print total_error
# test_sine()
def test_1times2squared():
    np.seterr(all='raise')
    trial_NN = Lee_NN([2, 500, 400, 500, 1], perf_fcn='mse', trans_fcn='sigmoid', reg=10 ** (-5));
    X = np.array([[0, 0], \
                  [0, 1], \
                  [0, 2], \
                  [1, 0], \
                  [1, 1], \
                  [1, 2], \
                  [2, 0], \
                  [2, 1], \
                  [2, 2]])
    y = np.array([0, 0, 0, 0, 1, 4, 0, 2, 8])
    trial_NN.train(X, y / 8.0, epochs=100, verb=1)
    y_predict = trial_NN.predict(X) * 8
    y_predict = np.around(y_predict, 2)
    print np.concatenate([X.T, np.atleast_2d(y), y_predict.T]).T
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    print 'error: ' + str(np.sum(np.square(y_predict - y)))
    if True:
        print 'weights values squared: ' + str(np.mean([np.mean([np.mean(np.square(y)) for y in x]) for x in trial_NN.weights]))
test_1times2squared()
def test_1times2():
    np.seterr(all='raise')
    trial_NN = Lee_NN([2, 3000, 300, 3000, 1], perf_fcn='mse', \
        trans_fcn='tanh', reg=10 ** (-10));
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
    y = np.array([0, 0, 0, 0, 1, 2, 0, 2, 4])
    trial_NN.train(X, y / 8.0, epochs=100, verb=1)
    y_predict = trial_NN.predict(X) * 8
    y_predict = np.around(y_predict, 2)
    print np.concatenate([np.atleast_2d(y), y_predict.T]).T
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    print np.sum(np.square(y_predict - y))
# test_1times2()
"""
X = np.ones([100]);
y = np.ones([100]);
for idx in xrange(10):
    X[idx] = np.random.rand()
    y[idx] = 1-X[idx]**2
"""

