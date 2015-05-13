'''
Generate synthetic exponential family MRF data.
'''
import numpy as np
import scipy.sparse as sp
from exponential_families import *
from mixedmrf import MixedMRF
from utils import *
import argparse
import csv

def set_bias_weights(node, weights):
    if isinstance(node, Gamma) or isinstance(node, Gaussian):
        weights[0,0] = 2.
        weights[1,0] = -2.
    elif isinstance(node, Dirichlet):
        weights[:,0] = 1.
    elif isinstance(node, ZeroInflated):
        weights[0,0] = np.random.random() * 2 - 1
        set_bias_weights(node.base_model, weights[1:]) # Recursively set the weights for the wrapper model
    else:
        weights[:,0] = np.random.random(size=node.num_params) * 2 - 1

def create_edges(n, edge_sparsity):
    edges = (np.random.random(size=(n,n)) > edge_sparsity).astype(int)
    edges[np.triu_indices(n)] = edges.T[np.triu_indices(n)]
    edges[np.diag_indices(n)] = 1
    return edges

def create_mrf(node_types, edge_sparsity, param_sparsity):
    nodes = [get_node(name) for name in node_types]
    edges = create_edges(len(node_types), edge_sparsity)
    degrees = edges.sum(axis=0)
    weights = []
    neighbor_partitions = []
    total_params = sum([x.num_params for x in nodes])
    for i, node in enumerate(nodes):
        rows = node.num_params # one set of weights per sufficient statistic
        cols = total_params - rows + 1 # one weight per neighbor param plus a bias
        
        # Create the matrix of node weights
        node_weights = np.zeros((rows, cols))

        # Create a partition map to back out which weights belong to which nodes
        node_neighbor_partition = np.zeros(cols, dtype=int)
        node_neighbor_partition[0] = i
        node_neighbor_partition[1:] = np.hstack([np.repeat(j, neighbor.num_params) for j,neighbor in enumerate(nodes) if j != i])
        
        # Copy weights we've already generated (weights are symmetric)
        for j, (neighbor_weights, neighbor_partition) in enumerate(zip(weights, neighbor_partitions)):
            node_idx = node_neighbor_partition == j
            neighbor_idx = neighbor_partition == i
            node_weights[:,node_idx] = neighbor_weights.T[neighbor_idx]

        # Create the bias weights
        set_bias_weights(node, node_weights)

        for j in xrange(i+1, len(nodes)):
            node_idx = node_neighbor_partition == j
            if edges[i,j]:
                node_weights[:,node_idx] = np.random.random(size=(rows,node_idx.sum())) * 2 - 1
                node_weights[:,node_idx] = node_weights[:,node_idx] * (np.random.random(size=(rows,node_idx.sum())) > param_sparsity)
                max_norm = 1.0 / max(degrees[i], degrees[j])
                while np.linalg.norm(node_weights[:,node_idx]) > max_norm:
                   node_weights[:,node_idx] *= 0.9

        weights.append(node_weights)
        neighbor_partitions.append(node_neighbor_partition)

    return MixedMRF(nodes, weights, neighbor_partitions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic data as drawn from an MRF via Gibbs sampling.')

    # I/O settings
    parser.add_argument('outdir', help='The directory to which all of the output will be saved.')
    parser.add_argument('--verbose', type=int, default=0, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')
    
    parser.add_argument('--nodes', nargs='+', help='The distribution of each node-conditional. Supported options: b (bernoulli), g (gamma), n (normal / gaussian), d# (dirichlet with # replaced with number of params), ziX (zero-inflated version of the X distribution, recursively defined)')
    parser.add_argument('--edge_sparsity', type=float, default=0.25, help='The probability of any two nodes having an edge in the MRF.')
    parser.add_argument('--param_sparsity', type=float, default=0.8, help='The probability of any two nodes with an edge having non-zero weight on a given parameter.')

    # Gibbs sampling settings
    parser.add_argument('--iterations', type=int, default=5000, help='The number of gibbs iterations to run.')
    parser.add_argument('--burn', type=int, default=100, help='The number of iterations to treat as burn-in.')
    parser.add_argument('--thin', type=int, default=10, help='The number of iterations to skip between samples.')

    parser.add_argument('--noraw', action='store_true', help='Do not output the raw data file, only the sufficient statistics file.')

    parser.set_defaults(noraw=False)

    # Get the arguments from the command line
    args = parser.parse_args()

    # Create the directory structure for storing the results
    output_dir = args.outdir + ('' if args.outdir.endswith('/') else '/')
    args_dir = make_directory(output_dir, 'args')
    data_dir = make_directory(output_dir, 'data')
    weights_dir = make_directory(output_dir, 'weights')
    edges_dir = make_directory(output_dir, 'edges')
    metrics_dir = make_directory(output_dir, 'metrics')

    # Create all the filenames
    nodes_file = data_dir + 'nodes.csv'
    args_file = args_dir + 'generate_data_args.txt'
    data_file = data_dir + 'data.csv'
    sufficient_statistics_file = data_dir + 'sufficient_statistics.csv'
    edges_file = edges_dir + 'edges.csv'
    weights_file = weights_dir + 'weights_node{0}.csv'

    # Save the parameters of the data
    save_args(args, args_file)

    save_nodes(args.nodes, nodes_file)

    np.seterr(all='raise')
    
    while True:
        mrf = create_mrf(args.nodes, args.edge_sparsity, args.param_sparsity)

        try:
            samples = []
            ss_samples = []

            # Draw the initial sample to start from
            sample = mrf.gibbs_sample(verbose=args.verbose > 1)

            for trial in xrange(args.iterations):
                # Draw another sample from the gibbs sampler
                sample = mrf.gibbs_sample(sample, verbose=args.verbose > 1)

                # Save the sample
                if trial > args.burn and trial % args.thin == 0:
                    samples.append(sample)
                    ss_samples.append(mrf.calc_sufficient_statistics(sample))

                if trial % 10000 == 0 and args.verbose:
                    print '\tSample #{0}'.format(trial)
        except Exception as ex:
            if args.verbose:
                print 'Exception thrown: {0}'.format(ex)
            continue

        print 'Saving weights and edges'
        mrf.save_weights(weights_file)
        mrf.save_edges(edges_file)

        break

    if not args.noraw:
        print 'Saving data'
        with open(data_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(mrf.ss_partitions)
            writer.writerows(samples)
    #np.savetxt(data_file, np.array(samples), delimiter=',')
    print 'Saving sufficient statistics'
    with open(sufficient_statistics_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(mrf.ss_partitions)
        writer.writerows(ss_samples)
    #np.savetxt(sufficient_statistics_file, np.array(ss_samples), delimiter=',', header=','.join([str(x) for x in mrf.ss_partitions]))
    print 'Finished!'


