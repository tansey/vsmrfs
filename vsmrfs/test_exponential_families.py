import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
from exponential_families import *
from utils import *

def test_bernoulli():
    expfam = Bernoulli()
    
    assert(expfam.num_params == 1)

    p = np.array([0.1, 0.5, 0.9, 0.25])
    eta = np.log(p / (1-p))

    g = expfam.grad_log_partition(eta)
    grad_eta = np.exp(eta) / (np.exp(eta) + 1)

    # Test gradient
    assert_array_almost_equal(g, grad_eta)

    # Test sum of gradients
    assert_array_almost_equal(g.sum(), grad_eta.sum())

    print 'eta: {0} Grad: {1} Sum: {2}'.format(pretty_str(eta), pretty_str(g), g.sum())


if __name__ == '__main__':
    x = np.array([0, 0.25, 0.1, 0.]) # observations
    
    eta = np.array([[5., 10., 2.0], [0.1, 2.0, 4.0], [0.15, 0.5, 1.0], [0.9, 4.0, 2.0]]).T
    
    # Get natural params
    eta[1] -= 1
    eta[2] *= -1
    
    zig = ZeroInflated(Gaussian())
    print 'Sufficient Statistics'
    print zig.sufficient_statistics(x)
    print ''

    print 'Log Base Measure'
    print zig.log_base_measure(x)
    print ''

    print 'Log-partition'
    print zig.log_partition(eta)
    print ''

    print 'Grad log partition'
    print zig.grad_log_partition(eta)
    print ''

    print 'Diagonal Hessian log partition'
    print zig.diagonal_hessian_log_partition(eta)
    print ''

    print 'Eta constraints'
    print zig.eta_constraints(eta.T)
    print ''

    print 'Grad eta constraints'
    print zig.grad_eta_constraints(eta.T)
    print ''

    print 'Diagonal Hessian eta constraints'
    print zig.diagonal_hessian_eta_constraints(eta.T)
    print ''