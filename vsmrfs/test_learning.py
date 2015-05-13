import matplotlib
matplotlib.use('Agg')
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import argparse
import csv
import sys
from node_learning import *
from exponential_families import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the maximum likelihood estimation (MLE) algorithm for a single node-conditional.')

    # Generic settings
    parser.add_argument('data_file', help='The file containing the sufficient statistics.')
    #parser.add_argument('neighbors_file', help='The file containing the neighbors partition.')
    parser.add_argument('--verbose', type=int, default=2, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')
    parser.add_argument('--distribution', default='b', help='The distribution of the target node-conditional.')
    parser.add_argument('--target', type=int, help='The target node.')
    #parser.add_argument('--postprocess', dest='generate_data', action='store_true', help='Re-runs the MLE to find the right weight values once nonzeros are detected. TODO: p >> n, so how do we do this? adaptive lasso?')
    
    parser.add_argument('--true_weights', help='The true weights file.')

    # Data saving settings
    parser.add_argument('--save_weights', help='The file where the resulting weights will be saved.')
    parser.add_argument('--save_edges', help='The file where the resulting edges will be saved.')
    parser.add_argument('--save_metrics', help='The file where the resulting metrics such as BIC, DoF, etc. will be saved.')
    
    # Plotting settings
    parser.add_argument('--plot_results', help='The file to which the results will be plotted.')
    parser.add_argument('--plot_edges', help='The file to which the estimated signal distribution will be plotted.')
    parser.add_argument('--plot_path', help='The file to which the solution path of the penalty (lambda) will be plotted.')
    parser.add_argument('--plot_final', help='The file to which the results of the final solution will be plotted.')
    
    # Solution path and lambda settings
    parser.add_argument('--solution_path', dest='solution_path', action='store_true', help='Use the solution path of the generalized lasso to find a good value for the penalty weight (lambda).')
    parser.add_argument('--min_lambda1', type=float, default=0.2, help='The minimum amount the lambda1 penalty can take in the solution path.')
    parser.add_argument('--max_lambda1', type=float, default=1.5, help='The maximum amount the lambda1 penalty can take in the solution path.')
    parser.add_argument('--min_lambda2', type=float, default=0.2, help='The minimum amount the lambda2 penalty can take in the solution path.')
    parser.add_argument('--max_lambda2', type=float, default=1.5, help='The maximum amount the lambda2 penalty can take in the solution path.')
    parser.add_argument('--penalty_bins', type=int, default=30, help='The number of lambda penalty values in the solution path.')
    parser.add_argument('--dof_tolerance', type=float, default=1e-4, help='The threshold for calculating the degrees of freedom.')
    parser.add_argument('--lambda1', type=float, default=0.3, help='The lambda1 penalty that controls the sparsity of edges (only used if --solution_path is not specified).')
    parser.add_argument('--lambda2', type=float, default=0.3, help='The lambda2 penalty that controls the sparsity of individual weights (only used if --solution_path is not specified).')

    # Convergence settings
    parser.add_argument('--rel_tol', type=float, default=1e-6, help='The convergence threshold for the main optimization loop.')
    parser.add_argument('--edge_tol', type=float, default=1e-3, help='The convergence threshold for the edge definition criteria.')
    parser.add_argument('--max_steps', type=int, default=300, help='The maximum number of steps for the main optimization loop.')
    parser.add_argument('--newton_rel_tol', type=float, default=1e-6, help='The convergence threshold for the inner loop of Newton\'s method.')
    parser.add_argument('--newton_max_steps', type=int, default=100, help='The maximum number of steps for the inner loop of Newton\'s method.')
    
    # ADMM settings
    parser.add_argument('--admm_alpha', type=float, default=1.8, help='The step size value for the ADMM solver.')
    parser.add_argument('--admm_inflate', type=float, default=2., help='The inflation/deflation rate for the ADMM step size.')
    

    parser.set_defaults()

    # Get the arguments from the command line
    args = parser.parse_args()
    

    # Load the data
    #data = sp.lil_matrix(np.loadtxt(args.data_file, delimiter=','))
    header = get_numeric_header(args.data_file)
    data = np.loadtxt(args.data_file, delimiter=',', skiprows=1)

    # Rearrange the data so that sufficient statistics of this node come first
    target_cols = np.where(header == args.target)[0]
    neighbors_partition = np.hstack([[args.target], np.delete(header, target_cols)])
    c = np.hstack([target_cols, np.delete(np.arange(data.shape[1]), target_cols)])
    print 'c: {0} target: {1}'.format(c,target_cols)
    data = data[:, c]

    true_weights = np.loadtxt(args.true_weights, delimiter=',')

    '''Bernoulli testing'''
    # Count
    # counts = np.zeros((2,2,2))
    # for row in data.astype(int):
    #     counts[row[0],row[1],row[2]] += 1

    # print 'X=0 Counts'
    # print pretty_str(counts[0])
    # print 'X=1 Counts'
    # print pretty_str(counts[1])


    # probs = counts[1] / (counts[0] + counts[1])

    # # normalize
    # print ''
    # print 'Empirical:'
    # print 'p(X0=1 | X1=0, X2=0) = {0}'.format(probs[0,0])
    # print 'p(X0=1 | X1=1, X2=0) = {0}'.format(probs[1,0])
    # print 'p(X0=1 | X1=0, X2=1) = {0}'.format(probs[0,1])
    # print 'p(X0=1 | X1=1, X2=1) = {0}'.format(probs[1,1])

    # print ''
    # print 'Truth: '
    # eta1 = true_weights.dot(np.array([1, 0, 0]))
    # print 'p(X0=1 | X1=0, X2=0) = {0}'.format(np.exp(eta1) / (1 + np.exp(eta1)))

    # eta1 = true_weights.dot(np.array([1, 1, 0]))
    # print 'p(X0=1 | X1=1, X2=0) = {0}'.format(np.exp(eta1) / (1 + np.exp(eta1)))

    # eta1 = true_weights.dot(np.array([1, 0, 1]))
    # print 'p(X0=1 | X1=0, X2=1) = {0}'.format(np.exp(eta1) / (1 + np.exp(eta1)))

    # eta1 = true_weights.dot(np.array([1, 1, 1]))
    # print 'p(X0=1 | X1=1, X2=1) = {0}'.format(np.exp(eta1) / (1 + np.exp(eta1)))

    '''Dirichlet testing'''
    means = data.mean(axis=0)
    print 'Averages: {0}'.format(means.reshape(len(means)/3, 3))
    print 'Truth:    {0}'.format(true_weights[:,0] / true_weights[:,0].sum())

    # print ''
    # five_cols = np.where(header == 5)[0]
    # for i in xrange(3):
    #     for j in five_cols:
    #         print '{0}->{1}: {2}'.format(i, j, np.correlate(data[:,i], data[:,j]))

    print ''
    print np.cov(data.T)[0:3][:, np.where(header == 5)]
