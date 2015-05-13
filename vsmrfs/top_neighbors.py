import numpy as np
import os
import csv
import argparse
from utils import *


def print_top(target, maxtop, infile, all_nodes, skip=None):
    neighbors = []
    weights = []
    with open(infile, 'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            n1 = int(line[0])
            n2 = int(line[1])
            
            if n1 != target and n2 != target:
                continue

            n = n1 if n1 != target else n2

            if skip and n in skip:
                continue            
            
            w = np.array([float(x) for x in line[2:]])
            magnitude = np.linalg.norm(w)
            sign = np.sign(w.sum())
            
            neighbors.append((magnitude, n, sign))

    neighbors = sorted(neighbors)
    neighbors.reverse()

    for i,(mag, nid, sign) in enumerate(neighbors[0:maxtop]):
        name = all_nodes[nid]
        print '{0}.\t{1}\t{2}'.format(i+1, name, mag*sign)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gets the top K edges for a given node.')

    parser.add_argument('experiment_dir', help='The directory for the experiment.')
    parser.add_argument('node', type=int, help='The ID of the node to look at.')
    parser.add_argument('top', type=int, help='The number of top edges.')
    parser.add_argument('--verbose', type=int, default=1, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')
    parser.add_argument('--skip', nargs='+', type=int, help='List of nodes to remove from the graph.')
    
    parser.set_defaults()

    # Get the arguments from the command line
    args = parser.parse_args()

    experiment_dir = args.experiment_dir + ('' if args.experiment_dir.endswith('/') else '/')
    data_dir = make_directory(experiment_dir, 'data')
    edges_dir = make_directory(experiment_dir, 'edges')
    weights_dir = make_directory(experiment_dir, 'weights')
    plots_dir = make_directory(experiment_dir, 'plots')

    nodenames_infile = data_dir + 'nodenames.csv'
    mle_weights_infile = weights_dir + 'mle_weights_node{0}.csv'
    and_edges_infile = edges_dir + 'and_mle_edges.csv'
    or_edges_infile = edges_dir + 'or_mle_edges.csv'
    and_graph_outfile = plots_dir + 'and_graph.dot'
    or_graph_outfile = plots_dir + 'or_graph.dot'

    if args.verbose:
        print 'Loading node names'

    all_nodes = {}
    with open(nodenames_infile, 'rb') as nnf:
        reader = csv.reader(nnf)
        for line in reader:
            idx = int(line[0])
            label = line[1]
            #extra = (',' + ','.join(line[2:])) if len(line) > 2 else ''
            all_nodes[idx] = label

    if args.verbose:
        print 'Top for {0} (AND graph)'.format(all_nodes[args.node])

    print_top(args.node, args.top, and_edges_infile, all_nodes, args.skip)

    # if args.verbose:
    #     print 'Top for OR graph'

    # print_top(args.node, args.top, or_edges_infile, all_nodes)



