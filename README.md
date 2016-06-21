# NN_Numpy

See NN_numpy for Neural Network Class, NNn_helper for helper functions,
NNn_test for test functions.

I have implemented a few performance functions, a few transfer functions, 
a feedforward or cascade forward (or customizable) network structure, 
regularization, random reinitializations of weights, 
stochastic batch or mini batch training, and an adaptive learning rate.

Make sure to use sigmoid on the output layer for only-positive outputs.
Use tanh for positive/negative outputs. It is not necessary to normalize the data, 
but centerring the outputs may be benefitial if using the tanh transfer function to the output layer.

This project is clean by all flake8 coding standards except for E128 and F405
Ex: flake8 NN_numpy.py --ignore E128,F405