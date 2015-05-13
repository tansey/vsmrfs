import numpy as np
from utils import *
from exponential_families import load_nodes
import csv
import scipy.sparse as sps

def get_node_map(nodes):
    '''Calculate the mapping from data column to node.'''
    cols = []
    for i,node in enumerate(nodes):
        cols.extend([i]*node.num_params)
    return np.array(cols)

class MixedMRF:
    def __init__(self, nodes, weights, neighbor_partitions):
        self.nodes = nodes
        self.weights = weights
        self.neighbor_partitions = neighbor_partitions
        
        self.obs_partitions = np.hstack([np.repeat(i, node.domain_size) for i, node in enumerate(nodes)])
        self.ss_partitions = np.hstack([np.repeat(i, node.num_params) for i, node in enumerate(nodes)])

    def calc_sufficient_statistics(self, data):
        return np.hstack([node.sufficient_statistics(data[self.obs_partitions == i])[0] for i, node in enumerate(self.nodes)])

    def gibbs_sample(self, start=None, verbose=False):
        if start is None:
            start = np.hstack([node.starting_x() for node in self.nodes])

        if verbose:
            print 'Starting: {0}'.format(pretty_str(start))

        # Get the vector of sufficient statistics
        sufficient_statistics = self.calc_sufficient_statistics(start)

        if verbose:
            print 'suff statistics: {0}'.format(pretty_str(sufficient_statistics))
            
        # Create the resulting vector
        result = np.copy(start)
        
        for i,(node, weights) in enumerate(zip(self.nodes, self.weights)):
            if verbose:
                print ''
                print 'Node #{0}: {1}'.format(i, node)
                print 'Weights: {0}'.format(pretty_str(weights))
            
            # Calculate the natural parameters
            eta = weights[:,1:].dot(sufficient_statistics[self.ss_partitions != i]) + weights[:,0]

            if verbose:
                print 'Theta: {0}\nNatural params: {1}'.format(pretty_str(weights), pretty_str(eta))

            sample = node.sample(eta)
            result[self.obs_partitions == i] = sample
            sufficient_statistics[self.ss_partitions == i] = node.sufficient_statistics(sample)

        if verbose:
            print 'gibbs sample: {0}'.format(pretty_str(result))

        return result

    def save_neighbors(self, filename):
        save_list_of_arrays(filename, self.neighbor_partitions, fmt='%1i')
    
    def save_weights(self, filename):
        save_list_of_arrays(filename, self.weights)

    def save_edges(self, filename, rel_tol=1e-4):
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            for i, (weights, neigh_part) in enumerate(zip(self.weights,self.neighbor_partitions)):
                for j in xrange(i,len(self.weights)):
                    w = weights[:,neigh_part == j]
                    if np.abs(w).max() > rel_tol:
                        writer.writerow([i,j] + list(w.flatten()))

class SparsePseudoMixedMRF:
    '''A sparse mixed MRF that uses the pseudo-likelihood to calculate the joint log-partition.'''
    def __init__(self, nodes, edges):
        self.nodes = nodes # list of ExponentialFamily objects
        self.edges = edges # a list where each element is a tuple ((i,j), weight)
        self.node_map = get_node_map(nodes)
        self.jll = None

        # Calculate the neighbors of each node and create the weight vectors
        self.neighbors = [[] for _ in nodes]
        self.weights = [[[0.] for _ in xrange(node.num_params)] for node in nodes] # Reserve the first weight for the bias term
        self.weights_map = {}
        for (i,j), w in edges:
            inode = nodes[i]

            # If this is the bias term
            if i == j:
                for k in xrange(inode.num_params):
                    self.weights[i][k][0] = w[k]
                self.weights_map[(i,j)] = w
                continue

            # If this is an edge between neighbors
            jnode = nodes[j]
            w = w.reshape((inode.num_params, jnode.num_params))
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)
            for k in xrange(inode.num_params):
                self.weights[i][k].extend(w[k])
            for k in xrange(jnode.num_params):
                self.weights[j][k].extend(w.T[k])
            self.weights_map[(i,j)] = w
            self.weights_map[(j,i)] = w.T

        self.neighbors = [np.array(x) for x in self.neighbors]
        self.weights = [np.array(x) for x in self.weights]

    def set_data(self, data):
        '''Set the sufficient statistics data to be used.'''
        assert(data.shape[1] == len(self.node_map))
        self.data = data
        self.dirty = True

    def joint_log_likelihood(self):
        '''Calculates the joint log-pseudo-likelihood'''
        if self.dirty:
            result = 0.
            for i, node in enumerate(self.nodes):
                result += self.node_log_likelihood(i, node)
            self.jll = result
        return self.jll

    def node_log_likelihood(self, node_id, node):
        neighbors = self.neighbors[node_id]
        
        # Figure out which columns can safely be deleted since they're unused
        target_cols = np.where(self.node_map == node_id)[0]
        neighbor_cols = np.where(np.in1d(self.node_map, neighbors))[0]
        unused = np.delete(np.arange(self.data.shape[1]), np.hstack([target_cols, neighbor_cols]))
        
        # Rearrange the data so that sufficient statistics of this node come first and any non-neighbor nodes are removed
        neighbors_partition = np.hstack([[node_id], np.delete(self.node_map, unused)]).astype(np.int32)
        c = np.delete(np.arange(self.data.shape[1]), unused)
        data_subset = self.data[:, c]

        weights = self.weights[node_id] # p x q+1
        sufficient_stats = data_subset[:,0:node.num_params] # n x p
        neighbor_stats = data_subset[:,node.num_params:] # n x q

        # The results matrix is n x p, where n = # examples and p = # natural parameters for this node
        results = np.zeros((sufficient_stats.shape[0], node.num_params))

        # Add all the natural parameters
        for i, w in enumerate(weights):
            # Handle nodes without edges
            if w.shape[0] < 2:
                continue
            if sps.issparse(sufficient_stats):
                ss = sufficient_stats[:,i].A[:,0]
            else:
                ss = sufficient_stats[:,i]
            results[:,i] += sps.diags(ss, 0).dot(neighbor_stats).dot(w[1:][:,np.newaxis])[:,0]
            results[:,i] += ss * w[0]

        # Calculate the log-partition function for each example
        a = node.log_partition(results.T)

        return results.sum() - a.sum()


def load_sparse_pseudomrf(experiment_dir, edges_path='edges/and_mle_edges.csv'):
    if not experiment_dir.endswith('/'):
        experiment_dir += '/'
    nodes = load_nodes(experiment_dir + 'data/nodes.csv')
    edges = load_edges(experiment_dir + edges_path)
    return SparsePseudoMixedMRF(nodes, edges)



        












