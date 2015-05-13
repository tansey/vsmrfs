import matplotlib
matplotlib.use('Agg')
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import argparse
import csv
import sys
from node_learning import *
from exponential_families import *
from utils import *


FIG_FONTSIZE = 18
FIG_TITLE_FONTSIZE = 28
FIG_LINE_WIDTH = 4
FIG_TICK_LABEL_SIZE = 14
FIG_BORDER_WIDTH = 2
FIG_TICK_WIDTH = 2

def save_metrics(results, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='=')
        writer.writerow(['dof', results['dof']])
        writer.writerow(['edge_count', len(results['edges'])])
        writer.writerow(['log_likelihood', results['log_likelihood']])
        writer.writerow(['aic', results['aic']])
        writer.writerow(['aicc', results['aicc']])
        writer.writerow(['bic', results['bic']])

def plot_path(results, filename):
    lambda1 = results['lambda1_grid']
    lambda2 = results['lambda2_grid']
    fig, axarr = plt.subplots(len(lambda1),4, sharex=True, figsize=(21, 5 * len(lambda1)))
    for i, lambda1_val in enumerate(lambda1):
        axarr[i,0].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
        axarr[i,1].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
        axarr[i,2].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
        axarr[i,3].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
        axarr[i,0].plot(results['lambda2_grid'], results['log_likelihood'][i], lw=FIG_LINE_WIDTH)
        axarr[i,0].axvline(results['lambda2_grid'][np.argmax(results['log_likelihood'][i])], ymin=results['log_likelihood'][i].min(), ymax=results['log_likelihood'][i].max(), color='r', linestyle='--')
        axarr[i,1].plot(results['lambda2_grid'], results['dof'][i], lw=FIG_LINE_WIDTH)
        axarr[i,1].axvline(results['lambda2_grid'][np.argmin(results['dof'][i])], ymin=results['dof'][i].min(), ymax=results['dof'][i].max(), color='r', linestyle='--')
        axarr[i,2].plot(results['lambda2_grid'], results['aic'][i], lw=FIG_LINE_WIDTH)
        axarr[i,2].axvline(results['lambda2_grid'][np.argmin(results['aic'][i])], ymin=results['aic'][i].min(), ymax=results['aic'][i].max(), color='r', linestyle='--')
        axarr[i,3].plot(results['lambda2_grid'], results['bic'][i], lw=FIG_LINE_WIDTH)
        axarr[i,3].axvline(results['lambda2_grid'][np.argmin(results['bic'][i])], ymin=results['bic'][i].min(), ymax=results['bic'][i].max(), color='r', linestyle='--')
        axarr[i,0].set_title('Log-Likelihood', fontsize=FIG_TITLE_FONTSIZE)
        axarr[i,1].set_title('Degrees of Freedom', fontsize=FIG_TITLE_FONTSIZE)
        axarr[i,2].set_title('AIC', fontsize=FIG_TITLE_FONTSIZE)
        axarr[i,3].set_title('BIC', fontsize=FIG_TITLE_FONTSIZE)
        axarr[i,0].set_xlabel('Lambda 2 (Lambda1 = {0})'.format(lambda1_val))
        axarr[i,1].set_xlabel('Lambda 2 (Lambda1 = {0})'.format(lambda1_val))
        axarr[i,2].set_xlabel('Lambda 2 (Lambda1 = {0})'.format(lambda1_val))
        axarr[i,3].set_xlabel('Lambda 2 (Lambda1 = {0})'.format(lambda1_val))
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the maximum likelihood estimation (MLE) algorithm for a single node-conditional.')

    # Generic settings
    parser.add_argument('experiment_dir', help='The directory for the experiment.')
    parser.add_argument('--experiment_label', default='', help='An extra label to prepend to all of the output files. Useful if running lots of comparison experiments on the same dataset.')
    parser.add_argument('--verbose', type=int, default=1, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')
    parser.add_argument('--target', type=int, help='The ID of the node to perform maximum likelihood expectation on its neighborhood.')
    parser.add_argument('--sample_weights', help='The name of an optional file containing sample weights. If unspecified, all samples are assumed to be equally weighted.')
    
    # Data storage settings
    parser.add_argument('--corpus', help='An optional meta-file containing lists of all the data documents to load. The format is a filename followed by a series of lines to add.')
    parser.add_argument('--sparse', dest='sparse', action='store_true', help='Run using the sparse data version. This version uses sparse scipy arrays instead of dense numpy ones.')
    parser.add_argument('--file_format', choices=['dense', 'sparse'], default='dense', help='The format that the underlying file uses. If it uses a dense format and --sparse is specified, zeros will be ignored. If --sparse is not specified, this is ignored and the file is assumed to be a dense numpy array.')
    
    # Plotting settings
    parser.add_argument('--plot_results', help='The file to which the results will be plotted.')
    parser.add_argument('--plot_path', help='The file to which the solution path of the penalty (lambda) will be plotted.')
    parser.add_argument('--plot_final', help='The file to which the results of the final solution will be plotted.')
    
    # Solution path and lambda settings
    parser.add_argument('--solution_path', dest='solution_path', action='store_true', help='Use the solution path of the generalized lasso to find a good value for the penalty weight (lambda).')
    parser.add_argument('--min_lambda1', type=float, default=0.0001, help='The minimum amount the lambda1 penalty can take in the solution path.')
    parser.add_argument('--max_lambda1', type=float, default=1., help='The maximum amount the lambda1 penalty can take in the solution path.')
    parser.add_argument('--min_lambda2', type=float, default=0.0001, help='The minimum amount the lambda2 penalty can take in the solution path.')
    parser.add_argument('--max_lambda2', type=float, default=1., help='The maximum amount the lambda2 penalty can take in the solution path.')
    parser.add_argument('--lambda1_bins', type=int, default=10, help='The number of lambda1 penalty values in the solution path.')
    parser.add_argument('--lambda2_bins', type=int, default=10, help='The number of lambda1 penalty values in the solution path.')
    parser.add_argument('--quality_metric', choices=['bic', 'aic', 'aicc'], default='bic', help='The metric to use when assessing the quality of a point along the solution path.')

    # Penalty settings
    parser.add_argument('--dof_tolerance', type=float, default=1e-4, help='The threshold for calculating the degrees of freedom.')
    parser.add_argument('--lambda1', type=float, default=0.3, help='The lambda1 penalty that controls the sparsity of edges (only used if --solution_path is not specified).')
    parser.add_argument('--lambda2', type=float, default=0.3, help='The lambda2 penalty that controls the sparsity of individual weights (only used if --solution_path is not specified).')

    # Convergence settings
    parser.add_argument('--converge_tol', type=float, default=1e-4, help='The convergence threshold for the main optimization loop.')
    parser.add_argument('--rel_tol', type=float, default=1e-6, help='The general error threshold for the main optimization loop.')
    parser.add_argument('--edge_tol', type=float, default=0.01, help='The convergence threshold for the edge definition criteria.')
    parser.add_argument('--max_steps', type=int, default=100, help='The maximum number of steps for the main optimization loop.')
    parser.add_argument('--newton_rel_tol', type=float, default=1e-6, help='The convergence threshold for the inner loop Newton\'s method.')
    parser.add_argument('--newton_max_steps', type=int, default=30, help='The maximum number of steps for the inner loop Newton\'s method.')
    
    # ADMM settings
    parser.add_argument('--admm_alpha', type=float, default=100, help='The initial step size value for the ADMM solver. It is typically a good idea to keep this huge at the start and have the gap be exponentially closed in the initial iterations.')
    parser.add_argument('--admm_inflate', type=float, default=2., help='The inflation/deflation rate for the ADMM step size.')

    parser.set_defaults(solution_path=False, sparse=False)

    # Get the arguments from the command line
    args = parser.parse_args()

    print 'Running Node Learning for node {0} {1}'.format(args.target, 'using solution path' if args.solution_path else 'with fixed lambda1={0} lambda2={1}'.format(args.lambda1, args.lambda2))
    sys.stdout.flush()

    # Get the directory and subdirs for the experiment
    experiment_dir = args.experiment_dir + ('' if args.experiment_dir.endswith('/') else '/')
    data_dir = make_directory(experiment_dir, 'data')
    weights_dir = make_directory(experiment_dir, 'weights')
    edges_dir = make_directory(experiment_dir, 'edges')
    metrics_dir = make_directory(experiment_dir, 'metrics')
    args_dir = make_directory(experiment_dir, 'args')

    # Create an optional string to prepend to output files
    prepend_str = (args.experiment_label + '_') if args.experiment_label != '' else ''

    # Get the input and output filenames
    data_file = data_dir + 'sufficient_statistics.csv'
    nodes_file = data_dir + 'nodes.csv'
    args_file = args_dir + '{0}args_node{1}.txt'.format(prepend_str, args.target)
    sample_weights_file = (weights_dir + args.sample_weights) if args.sample_weights else None
    weights_outfile = weights_dir + '{0}mle_weights_node{1}.csv'.format(prepend_str, args.target)
    edges_outfile = edges_dir + '{0}mle_edges_node{1}.csv'.format(prepend_str, args.target)
    metrics_outfile = metrics_dir + '{0}mle_metrics_node{1}.txt'.format(prepend_str, args.target)

    save_args(args, args_file)

    # Load the nodes and generate the column -> node mapping header
    nodes = load_nodes(nodes_file)
    header = []
    for i,node in enumerate(nodes):
        header.extend([i]*node.num_params)
    header = np.array(header)

    # Load the data
    if args.corpus:
        data = load_sparse_corpus(experiment_dir, args.corpus, nodes, verbose=args.verbose)
    else:
        #header = get_numeric_header(data_file)
        if args.sparse:
            data = load_sparse_data_from_dense_file(data_file, verbose=args.verbose) if args.file_format == 'dense' else load_sparse_data_from_sparse_file(data_file, nodes, verbose=args.verbose)
        else:
            data = np.loadtxt(data_file, delimiter=',', skiprows=1)

    # Load the sample weights, if any are present
    sample_weights = np.loadtxt(sample_weights_file, delimiter=',') if args.sample_weights else None
    if sample_weights is not None and sample_weights.shape[0] != data.shape[0]:
        raise Exception('Sample weights must be the same length as the data. Sample length: {0} Data length: {1}'.format(sample_weights.shape[0], data.shape[0]))

    # Rearrange the data so that sufficient statistics of this node come first
    target_cols = np.where(header == args.target)[0]
    neighbors_partition = np.hstack([[args.target], np.delete(header, target_cols)]).astype(np.int32)
    c = np.hstack([target_cols, np.delete(np.arange(data.shape[1]), target_cols)])
    data = data[:, c]
    
    # Get the exponential family distribution of this node
    dist = nodes[args.target]

    sufficient_stats = data[:,0:dist.num_params]
    neighbor_stats = data[:,dist.num_params:]

    # Initialize the node conditional
    node = MixedMRFNode(dist, rel_tol=args.rel_tol,
                              edge_tol=args.edge_tol,
                              converge_tol=args.converge_tol,
                              max_steps=args.max_steps,
                              newton_max_steps=args.newton_max_steps,
                              quality_metric=args.quality_metric,
                              verbose=args.verbose,
                              admm_alpha=args.admm_alpha,
                              admm_inflate=args.admm_inflate)

    # Set the data and cache whatever we can now
    node.set_data(sufficient_stats, neighbor_stats, neighbors_partition, sample_weights=sample_weights)

    if args.solution_path:
        path_results = node.solution_path(lambda1_range=(args.min_lambda1,args.max_lambda1),
                                          lambda2_range=(args.min_lambda2,args.max_lambda2),
                                          lambda1_bins=args.lambda1_bins,
                                          lambda2_bins=args.lambda2_bins)
        results = path_results['best']

        if args.plot_path:
            if args.verbose:
                print 'Plotting solution path to {0}'.format(args.plot_path)
            plot_path(path_results, args.plot_path)
    else:
        results = node.mle(lambda1=args.lambda1, lambda2=args.lambda2)

    theta = results['theta']
    edges = results['edges']

    np.savetxt(weights_outfile, theta, delimiter=',')
    #np.savetxt(edges_outfile, edges, fmt='%1i', delimiter=',')
    save_pseudoedges(edges, edges_outfile)
    save_metrics(results, metrics_outfile)

    print 'Done!'
