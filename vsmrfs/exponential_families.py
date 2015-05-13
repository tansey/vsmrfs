'''
A suite of common exponential family distributions.
'''
import numpy as np
import scipy.sparse as sps
from scipy.special import psi, polygamma, gammaln, gammasgn
from copy import deepcopy
from utils import pretty_str, safe_exp, safe_sq
import csv

def get_node(name):
    if name == 'bernoulli' or name == 'b':
        return Bernoulli()
    if name == 'gaussian' or name == 'normal' or name == 'n':
        return Gaussian()
    if name == 'gamma' or name == 'g':
        return Gamma()
    if name.startswith('dirichlet') or name.startswith('d') or name.startswith('dir'):
        num_params = int(name.replace('dirichlet', '').replace('dir', '').replace('d',''))
        return Dirichlet(num_params)
    if name.startswith('zi') or name.startswith('zeroinflated'):
        name = name[len('zi'):] if name.startswith('zi') else name[len('zeroinflated'):]
        return ZeroInflated(get_node(name))

def get_node_from_file(target, filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        line = reader.next()
        return get_node(line[target])

def load_nodes(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        return [get_node(x) for x in reader.next()]

def save_nodes(nodes, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(nodes)

class ExponentialFamily:
    def log_likelihood(self, eta, x):
        if sps.issparse(x):
            # CSC format
            y = x.multiply(eta.T).sum()
        else:
            y = (eta.T * x).sum()
        return y + self.log_base_measure(x).sum() - self.log_partition(eta).sum()

    def log_base_measure(self, x):
        pass

    def sufficient_statistics(self, x):
        pass

    def log_partition(self, eta):
        pass

    def grad_log_partition(self, eta):
        pass

    def hessian_log_partition(self, eta):
        pass

    def diagonal_hessian_log_partition(self, eta):
        pass

    def sample(self, eta, count=1):
        pass

    def starting_eta(self):
        pass

    def starting_x(self):
        pass

class Bernoulli(ExponentialFamily):
    def __init__(self):
        self.num_params = 1
        self.domain_size = 1

    def sufficient_statistics(self, x):
        if type(x) is not np.ndarray or len(x.shape) == 1:
            return np.array([x]).T
        return np.copy(x)

    def log_base_measure(self, x):
        return np.zeros(x.shape)

    def log_partition(self, eta):
        return np.log(1 + safe_exp(eta))

    def grad_log_partition(self, eta):
        exp_eta = safe_exp(eta)
        return exp_eta / (exp_eta + 1.0)

    def hessian_log_partition(self, eta):
        exp_eta = safe_exp(eta)
        return -exp_eta / safe_sq(exp_eta + 1)

    def diagonal_hessian_log_partition(self, eta):
        return self.hessian_log_partition(eta)

    def sample(self, eta, count=1):
        exp_eta = safe_exp(eta)
        p = exp_eta / (1 + exp_eta)
        return np.random.random(size=count) < p

    def eta_constraints(self, eta):
        return np.array([[]])

    def grad_eta_constraints(self, eta):
        return np.array([[]])

    def diagonal_hessian_eta_constraints(self, eta):
        return np.array([[]])

    def starting_x(self):
        return np.zeros(1)

    def __repr__(self):
        return 'Bernoulli'

class Gamma(ExponentialFamily):
    def __init__(self):
        self.num_params = 2
        self.domain_size = 1

    def sufficient_statistics(self, x):
        return np.array([np.log(x), x]).T

    def log_base_measure(self, x):
        return np.zeros(x.shape)

    def log_partition(self, eta):
        #assert np.all(gammasgn(eta[0]+1) == 1)
        return gammaln(eta[0] + 1) - (eta[0] + 1) * np.log(-eta[1])

    def grad_log_partition(self, eta):
        return np.array([psi(eta[0] + 1) - np.log(-eta[1]), -(eta[0] + 1) / eta[1]])

    def hessian_log_partition(self, eta):
        return np.array([[polygamma(1, eta[0] + 1), -1.0 / eta[1]],
                         [-1.0 / eta[1], (eta[0] + 1) / safe_sq(eta[1])]])

    def diagonal_hessian_log_partition(self, eta):
        return np.array([polygamma(1, eta[0] + 1), (eta[0] + 1) / safe_sq(eta[1])])

    def sample(self, eta, count=1):
        return np.random.gamma(eta[0] + 1, -1.0 / eta[1], size=count)

    def eta_constraints(self, eta):
        if len(eta.shape) == 1:
            return np.array([-1 - eta[0], eta[1]])
        return np.array([-1 - eta[:,0], eta[:,1]])

    def grad_eta_constraints(self, eta):
        if len(eta.shape) == 1:
            return np.array([-1., 1.])
        return np.array([np.zeros(eta.shape[0]) - 1., np.ones(eta.shape[0])])

    def diagonal_hessian_eta_constraints(self, eta):
        if len(eta.shape) == 1:
            return np.array([0., 0.])
        return np.array([np.zeros(eta.shape[0]), np.zeros(eta.shape[0])])

    def starting_x(self):
        return np.ones(1)

    def __repr__(self):
        return 'Gamma'

class Gaussian(ExponentialFamily):
    def __init__(self):
        self.num_params = 2
        self.domain_size = 1

    def sufficient_statistics(self, x):
        return np.array([x, safe_sq(x)]).T

    def log_base_measure(self, x):
        return np.repeat(np.log(1./np.sqrt(2*np.pi)), x.shape[0])

    def log_partition(self, eta):
        return -safe_sq(eta[0]) / (4*eta[1]) - 0.5 * np.log(-2 * eta[1])

    def grad_log_partition(self, eta):
        return np.array([-0.5 * eta[0] / eta[1], (safe_sq(eta[0]) - 2*eta[1]) / (4 * safe_sq(eta[1]))])

    def hessian_log_partition(self, eta):
        return np.array([[-0.5 / eta[1],0.5 * eta[0] / safe_sq(eta[1])],
                         [0.5 * eta[0] / safe_sq(eta[1]),(eta[1] - safe_sq(eta[0])) / (2*eta[1]**3)]])

    def diagonal_hessian_log_partition(self, eta):
        return np.array([-0.5 / eta[1], (eta[1] - safe_sq(eta[0])) / (2*eta[1]**3)])

    def sample(self, eta, count=1):
        variance = -1. / (2. * eta[1])
        mu = eta[0] * variance
        sigma = np.sqrt(variance)
        return np.random.normal(mu, sigma, size=count)

    def eta_constraints(self, eta):
        if len(eta.shape) == 1:
            return np.array([eta[1]])
        return np.array([np.zeros(0),eta[:,1]])

    def grad_eta_constraints(self, eta):
        if len(eta.shape) == 1:
            return np.array([1.])
        return np.array([np.zeros(0),np.ones(eta.shape[0])])

    def diagonal_hessian_eta_constraints(self, eta):
        if len(eta.shape) == 1:
            return np.array([0., 0.])
        return np.array([np.zeros(0),np.zeros(eta.shape[0])])

    def starting_x(self):
        return np.zeros(1)

    def __repr__(self):
        return 'Gaussian'

class Dirichlet(ExponentialFamily):
    def __init__(self, num_params):
        self.num_params = num_params
        self.domain_size = num_params

    def sufficient_statistics(self, x):
        if len(x.shape) == 1:
            return np.array([np.log(x)])
        return np.log(x)

    def log_base_measure(self, x):
        return np.zeros(x.shape[0])

    def log_partition(self, eta):
        p = eta+1
        np.log(p.min())
        return gammaln(p).sum(axis=0) - gammaln(p.sum(axis=0))

    def grad_log_partition(self, eta):
        p = eta+1
        np.log(p.min())
        return psi(p) - psi(p.sum(axis=0))

    def hessian_log_partition(self, eta):
        pass

    def diagonal_hessian_log_partition(self, eta):
        p = eta+1
        np.log(p.min())
        return polygamma(1, p) - polygamma(1, p.sum(axis=0))

    def sample(self, eta, count=1):
        return np.random.dirichlet(eta+1, size=count)

    def eta_constraints(self, eta):
        return (-eta - 1.).T

    def grad_eta_constraints(self, eta):
        return np.zeros(eta.shape).T - 1.

    def diagonal_hessian_eta_constraints(self, eta):
        return np.zeros(eta.shape).T

    def starting_x(self):
        return np.ones(self.num_params) / float(self.num_params)

    def __repr__(self):
        return 'Dirichlet'

class ZeroInflated(ExponentialFamily):
    def __init__(self, base_model):
        self.base_model = base_model
        self.num_params = 1 + base_model.num_params
        self.domain_size = 1 
        # TODO: generalize to multivariate and arbitrary points that may be in the domain of the base model

    def sufficient_statistics(self, x):
        ss = np.zeros((x.shape[0], 3))
        ss[x == 0, 0] = 1
        ss[x != 0, 1:] = self.base_model.sufficient_statistics(x[x != 0])
        return ss

    def log_base_measure(self, x):
        if sps.issparse(x):
            # TODO: handle sparse data better
            x = x.todense()
        result = np.zeros(x.shape[0])
        idx = np.where(x[:,0]==0)[0][0]
        result[idx] = self.base_model.log_base_measure(x[:,1:][idx])
        return result

    def log_partition(self, eta):
        return np.log(safe_exp(eta[0]) + safe_exp(self.base_model.log_partition(eta[1:])))

    def grad_log_partition(self, eta):
        exp_base_log_partition = safe_exp(self.base_model.log_partition(eta[1:]))
        exp_x0 = safe_exp(eta[0])
        denominator = exp_base_log_partition + exp_x0
        w = (exp_base_log_partition / denominator)
        return np.concatenate(((exp_x0 / denominator)[:,np.newaxis].T,
                               self.base_model.grad_log_partition(eta[1:]) * w), axis=0)

    def hessian_log_partition(self, eta):
        pass

    def diagonal_hessian_log_partition(self, eta):
        base_log_partition = self.base_model.log_partition(eta[1:])
        exp_base_log_partition = safe_exp(base_log_partition)
        exp_x0 = safe_exp(eta[0])
        exp_sum = safe_exp(eta[0] + base_log_partition)
        sum_exp = exp_base_log_partition + exp_x0
        sq_sum_exp = safe_sq(sum_exp)
        diag_hess_base = self.base_model.diagonal_hessian_log_partition(eta[1:])
        sq_grad_base = safe_sq(self.base_model.grad_log_partition(eta[1:]))
        numerator = np.zeros(diag_hess_base.shape)
        numerator[:,sq_sum_exp != np.inf] = (sum_exp[sq_sum_exp != np.inf] * diag_hess_base[:, sq_sum_exp != np.inf] + exp_x0[sq_sum_exp != np.inf] * sq_grad_base[:, sq_sum_exp != np.inf]) / sq_sum_exp[sq_sum_exp != np.inf]
        return np.concatenate(((exp_sum / sq_sum_exp)[:,np.newaxis].T,
                                exp_base_log_partition * numerator), axis=0)

    def sample(self, eta, count=1):
        exp_x0 = safe_exp(eta[0])
        prob_x0 = exp_x0 / (exp_x0 + safe_exp(self.base_model.log_partition(eta[1:])))
        results = np.zeros(count)
        nonzero = np.random.random(size=count) > prob_x0
        results[nonzero] = self.base_model.sample(eta[1:], count=nonzero.sum())
        return results

    def eta_constraints(self, eta):
        base_constraints = self.base_model.eta_constraints(eta[:,1:]) if len(eta.shape) > 1 else self.base_model.eta_constraints(eta[1:])
        return np.array([np.array([])] + [x for x in base_constraints])

    def grad_eta_constraints(self, eta):
        base_constraints = self.base_model.grad_eta_constraints(eta[:,1:]) if len(eta.shape) > 1 else self.base_model.grad_eta_constraints(eta[1:])
        return np.array([np.array([])] + [x for x in base_constraints])

    def diagonal_hessian_eta_constraints(self, eta):
        base_constraints = self.base_model.diagonal_hessian_eta_constraints(eta[:,1:]) if len(eta.shape) > 1 else self.base_model.diagonal_hessian_eta_constraints(eta[1:])
        return np.array([np.array([])] + [x for x in base_constraints])

    def starting_x(self):
        return np.zeros(1)

    def __repr__(self):
        return 'Zero-Inflated {0}'.format(self.base_model)


