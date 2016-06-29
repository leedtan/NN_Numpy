from NNn_validation import *  # noqa
import time


def NNn_optimize(fcn, opt_vals, X_trn, Y_trn, optimize,
                 k=10, hidden_layers=[100], default_net=0,
                 netstruc='feed_forward', reg=10 * 8 - 10,
                 lookback=0, LR=.1, epochs=100, verb=0,
                 re_init=5, re_init_d=20, trans='sigmoid'):
    k = 4
    global_accuracy = []
    if len(optimize) == 1:
        for hyper_val in opt_vals:
            if optimize[0] == 'net width':
                hidden = [math.floor(hyper_val) for _ in hidden_layers]
                layers = [len(X_trn[0])]
                for h in hidden:
                    layers.append(h)
                layers.append(len(Y_trn[0]))
                net = NNn(layers, netstruc=netstruc, reg=reg, trans=trans)
                start = time.time()
                [_, error, _] = fcn(X_trn=X_trn, Y_trn=Y_trn, k=k,
                                    net=net,
                                    LR=LR, epochs=epochs, verb=verb,
                                    re_init=re_init, re_init_d=re_init_d)
                end = time.time()
                global_accuracy.append(error)
                print optimize[0], ' ', hyper_val, '  error: ', error, \
                    ' And it took ', end - start, ' seconds.'
            if optimize[0] == 'net depth':
                hidden = [math.floor(hidden_layers[0])
                          for _ in range(hyper_val)]
                layers = [len(X_trn[0])]
                for h in hidden:
                    layers.append(h)
                layers.append(len(Y_trn[0]))
                net = NNn(layers, netstruc=netstruc, reg=reg, trans=trans)
                start = time.time()
                [_, error, _] = fcn(X_trn=X_trn, Y_trn=Y_trn, k=k,
                                 net=net,
                                 LR=LR, epochs=epochs, verb=verb,
                                 re_init=re_init, re_init_d=re_init_d)
                end = time.time()
                global_accuracy.append(error)
                print optimize[0], ' ', hyper_val, '  error: ', error, \
                    ' And it took ', end - start, ' seconds.'
            if optimize[0] == 'reg':
                hidden = copy.deepcopy(hidden_layers)
                layers = [len(X_trn[0])]
                for h in hidden:
                    layers.append(h)
                layers.append(len(Y_trn[0]))
                net = NNn(layers, netstruc=netstruc,
                          reg=hyper_val, trans=trans)
                start = time.time()
                [_, error, _] = fcn(X_trn=X_trn, Y_trn=Y_trn, k=k,
                                 net=net,
                                 LR=LR, epochs=epochs, verb=verb,
                                 re_init=re_init, re_init_d=re_init_d)
                end = time.time()
                global_accuracy.append(error)
                print optimize[0], ' ', hyper_val, '  error: ', error, \
                    ' And it took ', end - start, ' seconds.'
            if optimize[0] == 'epochs':
                start = time.time()
                [_, error, _] = fcn(X_trn=X_trn, Y_trn=Y_trn, k=k,
                                 net=default_net,
                                 LR=LR, epochs=hyper_val, verb=verb,
                                 re_init=re_init, re_init_d=re_init_d)
                end = time.time()
                global_accuracy.append(error)
                print optimize[0], ' ', hyper_val, '  error: ', error, \
                    ' And it took ', end - start, ' seconds.'
            if optimize[0] == 're init':
                start = time.time()
                [_, error, _] = fcn(X_trn=X_trn, Y_trn=Y_trn, k=k,
                                 net=default_net,
                                 LR=LR, epochs=epochs, verb=verb,
                                 re_init=hyper_val, re_init_d=re_init_d)
                end = time.time()
                global_accuracy.append(error)
                print optimize[0], ' ', hyper_val, '  error: ', error, \
                    ' And it took ', end - start, ' seconds.'
            if optimize[0] == 're init depth':
                start = time.time()
                [_, error, _] = fcn(X_trn=X_trn, Y_trn=Y_trn, k=k,
                                 net=default_net,
                                 LR=LR, epochs=epochs, verb=verb,
                                 re_init=re_init, re_init_d=hyper_val)
                end = time.time()
                global_accuracy.append(error)
                print optimize[0], ' ', hyper_val, '  error: ', error, \
                    ' And it took ', end - start, ' seconds.'
    # plt.plot(opt_vals, global_accuracy)
    # plt.show()
    return global_accuracy
