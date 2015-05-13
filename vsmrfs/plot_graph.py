import numpy as np
import os
import csv
import argparse
from utils import *

def create_graph(infile, outfile, weights_infile, all_nodes, skip_self, ignored, weight_choice, top):
    FILE_START = 'graph G {\ngraph [overlap=scale];\n'
    FILE_END = '}\n'
    NODE_FORMAT = 'n{0} [label="{1}"{2}];\n'
    EDGE_FORMAT = 'n{0} -- n{1}[color={2};penwidth={3}];\n'
    nodes_used = set()
    edges = []
    edge_weights = []
    with open(infile, 'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            n1 = int(line[0])
            n2 = int(line[1])
            
            # Skip self-references
            if skip_self and n1 == n2:
                continue

            if ignored and (n1 in ignored or n2 in ignored):
                continue
            
            w = np.array([float(x) for x in line[2:]])
            magnitude = np.linalg.norm(w)
            sign = np.sign(w.sum())

            ########################
            # TEMP for ICML graphs
            # First parameter is prob(zero) (smaller = higher)
            # Second parameter is prob(high outlier) (larger = higher)
            # Third parameter is shape (larger = higher)
            # Fourth parameter is rate (smaller = higher)
            # PIGamma-PIGamma
            if len(w) == 16:
                w = w.reshape((4,4))
                print '{0} <-> {1}: {2} {3} {4} {5}'.format(all_nodes[n1][0], all_nodes[n2][0], np.linalg.norm(w[0])**2, np.linalg.norm(w[1])**2, np.linalg.norm(w[2])**2, -np.linalg.norm(w[3])**2)
                #w = np.linalg.norm(w[0])**2 + np.linalg.norm(w[1])**2 + -np.linalg.norm(w[2])**2 * np.sign(w[2].sum()) + np.linalg.norm(w[3])**2
                w = np.linalg.norm(w[0])**2 + np.linalg.norm(w[1])**2 + np.linalg.norm(w[2])**2 + -np.linalg.norm(w[3])**2
                #w = np.linalg.norm(w[2]) * np.linalg.norm(w[3]) * np.sign(w[3].sum())
                magnitude = np.abs(w)
            # Bernoulli-PIGamma
            elif len(w) == 4:
                if all_nodes[n1][0] == 'protein':
                    print '{0} <-> {1}: {2} {3} {4} {5}'.format(all_nodes[n1][0], all_nodes[n2][0], w[0], w[1], w[2], w[3])
                magnitude = np.linalg.norm(w)
                w = w[2]**2 + -w[3]**2
                #w = np.abs(w[2]) * (w[3]+0.001)
                
                #w = np.linalg.norm(w)
                #magnitude = np.abs(w)
            elif len(w) == 1:
                w = w
                magnitude = np.abs(w)
            else:
                raise Exception('Weird size: {0}'.format(len(w)))
            ########################

            nodes_used.add(n1)
            nodes_used.add(n2)
            edges.append((magnitude,n1,n2,w,sign))
            edge_weights.append(w)

    if top:
        edges = sorted(edges)
        edges.reverse()
        edges = edges[0:top]
        #edges = edges[0:top/2] + edges[-top/2:]
        nodes_used = set([n1 for m, n1, n2, w, s in edges] + [n2 for m, n1, n2, w, s in edges])
        edge_weights = [w for m, n1, n2, w, s in edges]
            
    # Calculate the thickness of each line
    edge_weights = np.abs(np.array(edge_weights))
    quantiles = np.array([np.percentile(edge_weights, i*10) for i in xrange(11)])

    with open(outfile, 'wb') as f:
        f.write(FILE_START)
        for nid in nodes_used:
            node = all_nodes[nid]
            f.write(NODE_FORMAT.format(nid, node[0], (node[1]+',fontsize=60') if node[1] != '' else ',fillcolor=green,style=filled,fontsize=44'))
        for m,n1,n2,w,s in edges:
            color = 'blue' if w < 0 else 'orange'
            print 'w: {0} n1: {1} n2: {2}'.format(w, all_nodes[n1][0], all_nodes[n2][0])
            #color = 'blue'
            bucket = np.where(quantiles <= np.abs(w))[0].max() + 1
            f.write(EDGE_FORMAT.format(n1, n2, color, bucket*2))
        f.write(FILE_END)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots an undirected food graph.')

    parser.add_argument('experiment_dir', help='The directory for the experiment.')
    parser.add_argument('--verbose', type=int, default=1, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')
    parser.add_argument('--noself', action='store_true', dest='noself', help='Ignore self-edges.')
    parser.add_argument('--skip', nargs='+', type=int, help='List of nodes to remove from the graph.')
    parser.add_argument('--top', type=int, help='Plot only the top K edges for a given hub node.')
    
    parser.set_defaults(noself=False)

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
            extra = (',' + ','.join(line[2:])) if len(line) > 2 else ''
            all_nodes[idx] = (label, extra)

    if args.verbose:
        print 'Creating AND graph'

    create_graph(and_edges_infile, and_graph_outfile, mle_weights_infile, all_nodes, args.noself, args.skip, 'avg', args.top)

    # if args.verbose:
    #     print 'Creating OR graph'

    # create_graph(or_edges_infile, or_graph_outfile, mle_weights_infile, all_nodes, args.noself, args.skip, 'avg', args.top)



