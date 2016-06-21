import numpy as np
# names of batch training types:
solo = -1
group = 0

cat = np.concatenate


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
    return (y - a)


def mse(y, a):
    return np.sum(np.square(y - a))


def mse_p(y, a):
    return (y - a)


def copy_weights(weights):
    return [[np.array([z for z in w]) if (len(w) != 1 or w[0] != 0)
             else [0] for w in x] for x in weights]


def zero_weights(weights):
    return [[np.array([np.zeros_like(z) for z in w]) if
             (len(w) != 1 or w[0] != 0) else [0] for w in x] for x in weights]

