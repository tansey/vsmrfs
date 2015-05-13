import numpy as np
import os
import csv
import argparse
from utils import *
from exponential_families import *

def merge_pseudoedges(w1, w2, strat):
    if strat == 'avg':
        return (w1+w2) / 2.0
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    if strat == 'min':
        return w1 if norm1 < norm2 else w2
    if strat == 'max':
        return w1 if norm1 > norm2 else w2
    raise Exception('Unknown merge strategy: {0}'.format(strat))

def calc_tpr_fpr(truth, estimate, total):
    total_positive = float(len(truth))
    total_negative = total - total_positive
    tpr = len(truth & estimate) / max(total_positive, 1.0)
    fpr = len(estimate - truth) / max(total_negative, 1.0)
    return (tpr, fpr)

def recovered_parameters(edges, rel_tol):
    '''Gets the set of edge parameters that are non-zero.'''
    p = []
    for (i,j), w in edges:
        # Skip the bias weights
        if i == j:
            continue
        for k, x in enumerate(w):
            if np.abs(x) > rel_tol:
                p.append((i,j,k))
    return set(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stitches together a graph from a set of local node neighborhoods.')

    parser.add_argument('experiment_dir', help='The directory for the experiment.')
    parser.add_argument('--nodes', nargs='+', help='The node types in the graph.')
    parser.add_argument('--experiment_label', default='', help='An extra label to prepend to all of the output files. Useful if running lots of comparison experiments on the same dataset.')
    parser.add_argument('--verbose', type=int, default=1, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')
    parser.add_argument('--evaluate',  dest='evaluate', action='store_true', help='Evaluate the resulting graph against the true underlying model.')
    parser.add_argument('--mergestrat', choices=['avg', 'min', 'max'], default='avg', help='The strategy to use to form a single edge weight from two pseudo-edge weights.')
    parser.add_argument('--rel_tol', type=float, default=0.005, help='The threshold for a parameter to be considered present.')
    parser.set_defaults(evaluate=False)

    # Get the arguments from the command line
    args = parser.parse_args()

    experiment_dir = args.experiment_dir + ('' if args.experiment_dir.endswith('/') else '/')
    edges_dir = make_directory(experiment_dir, 'edges')

    # Create an optional string to prepend to output files
    prepend_str = (args.experiment_label + '_') if args.experiment_label != '' else ''

    edges_infile = edges_dir + prepend_str + 'mle_edges_node{0}.csv'
    and_edges_outfile = edges_dir + '{0}and_mle_edges.csv'.format(prepend_str)
    or_edges_outfile = edges_dir + '{0}or_mle_edges.csv'.format(prepend_str)

    if args.verbose:
        print 'Loading node edges'

    # Figure out the number of nodes
    all_node_edges = []
    while os.path.exists(edges_infile.format(len(all_node_edges))):
        #node_edges = np.loadtxt(edges_infile.format(len(all_node_edges)), delimiter=',', dtype=int)
        node_edges, node_weights = load_pseudoedges(edges_infile.format(len(all_node_edges)))
        if type(node_edges) != np.ndarray or len(node_edges.shape) == 0:
            node_edges = np.array([node_edges])
        all_node_edges.append((node_edges, node_weights))

    num_nodes = len(all_node_edges)

    if args.verbose:
        print 'Found {0} nodes. Stitching graph together. Trying both ANDing and ORing.'.format(num_nodes)

    # Calculate the edges by ANDing and ORing the node neighborhoods together
    and_edges = []
    or_edges = []
    for i, (node_edges, node_weights) in enumerate(all_node_edges):
        print 'Node {0}: {1}'.format(i, node_edges)
        for j, w in zip(node_edges, node_weights):
            # If we've already looked at this edge, skip it
            if j < i:
                continue
            # Super slow search that's O(n^2) for a full graph
            found = False
            opp_edges, opp_weights = all_node_edges[j]
            for opp_i, opp_w in zip(opp_edges, opp_weights):
                # If this is the corresponding pseudo-edge in the opposite direction, merge the 2 weights and add the edge
                if opp_i == i:
                    merged_w = merge_pseudoedges(w, opp_w, args.mergestrat)
                    edge = ((i,j), merged_w)
                    and_edges.append(edge)
                    or_edges.append(edge)
                    found = True
                    break
            if not found:
                # If we didn't find a corresponding pseudo-edge in the opposite direction, then only add this to the OR edges
                or_edges.append(((i,j),w))

    if args.evaluate:
        if args.verbose:
            print 'Evaluating'
        nodes = [get_node(name) for name in args.nodes]
        num_params = 0.
        num_edges = 0.
        for i,n in enumerate(nodes):
            for j,m in enumerate(nodes):
                if i >= j:
                    continue
                num_params += n.num_params * m.num_params
                num_edges += 1.
        simple_and = set([edge for edge,w in and_edges if edge[0] != edge[1]])
        simple_or = set([edge for edge,w in or_edges if edge[0] != edge[1]])
        true_edges_file = edges_dir + 'edges.csv'
        true_edges = load_edges(true_edges_file)
        simple_true_edges = set([edge for edge,w in true_edges if edge[0] != edge[1]])
        and_tpr, and_fpr = calc_tpr_fpr(simple_true_edges, simple_and, num_edges)
        or_tpr, or_fpr = calc_tpr_fpr(simple_true_edges, simple_or, num_edges)
        if args.verbose:
            print '    ======= Edges ======='
            print '        TPR      FPR     '
            print '    -----------------------'
            print 'AND |  {0:.4f}  |  {1:.4f}  |'.format(and_tpr, and_fpr)
            print '    |----------------------'
            print 'OR  |  {0:.4f}  |  {1:.4f}  |'.format(or_tpr, or_fpr)
            print '    -----------------------'
            print ''
        metrics_dir = make_directory(experiment_dir, 'metrics')
        roc_metrics_file = metrics_dir + prepend_str + 'edge_roc_metrics.csv'
        with open(roc_metrics_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['AND', and_tpr, and_fpr])
            writer.writerow(['OR', or_tpr, or_fpr])
        # Now evaluate the parameter sparsity
        true_params = recovered_parameters(true_edges, rel_tol=args.rel_tol)
        and_params = recovered_parameters(and_edges, rel_tol=args.rel_tol)
        or_params = recovered_parameters(or_edges, rel_tol=args.rel_tol)
        and_tpr, and_fpr = calc_tpr_fpr(true_params, and_params, num_params)
        or_tpr, or_fpr = calc_tpr_fpr(true_params, or_params, num_params)
        if args.verbose:
            print '    ===== Parameters ====='
            print '        TPR      FPR     '
            print '    -----------------------'
            print 'AND |  {0:.4f}  |  {1:.4f}  |'.format(and_tpr, and_fpr)
            print '    |----------------------'
            print 'OR  |  {0:.4f}  |  {1:.4f}  |'.format(or_tpr, or_fpr)
            print '    -----------------------'
            print ''
        roc_metrics_file = metrics_dir + prepend_str + 'param_roc_metrics.csv'
        with open(roc_metrics_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['AND', and_tpr, and_fpr])
            writer.writerow(['OR', or_tpr, or_fpr])

    if args.verbose:
        print 'Saving edges to {0} and {1}'.format(and_edges_outfile, or_edges_outfile)

    and_edges.sort(key=lambda s: s[0])
    or_edges.sort(key=lambda s: s[0])

    save_edges(and_edges, and_edges_outfile)
    save_edges(or_edges, or_edges_outfile)
