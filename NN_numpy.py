# The following is a Neural Network class using Numpy.
# It should test clean using flake8 for all coding
# standards except for E128 and F405
# Ex: flake8 NN_numpy.py --ignore E128,F405

from NNn_helper import *  # noqa
import sys

# gb means global best
# LR means learning rate
# Lmap means layer map
# pt replaced future. it means the idx'd array points to the future layer


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
                        self.weights[row][col] += self.batch_deltas[row][col]\
                            / (col - row)
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
