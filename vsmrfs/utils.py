import numpy as np
import scipy.sparse as sps
import os
import csv

def backtracking_line_search(x, f, delta_x, grad_fx, fx=None, alpha=0.2, beta=0.5, verbose=False, verbose_prefix=''):
    if fx is None:
        fx = f(x)
    grad_fx_dot_delta_x = grad_fx.flatten().dot(delta_x.flatten())
    t = 1.0
    if verbose:
        print '{0}Starting backtracking line search'.format(verbose_prefix)
        print '{1}x:       {0}'.format(pretty_str(x).replace('\n', '\n{0}'.format(verbose_prefix)), verbose_prefix)
        print '{1}delta_x: {0}'.format(pretty_str(delta_x).replace('\n', '\n{0}'.format(verbose_prefix)), verbose_prefix)
        print '{4}t={0} fx={1} f(x + t*delta_x)={2} fx + alpha * t * grad_fx.dot(delta_x)={3}'.format(t, fx, f(x + t*delta_x), fx + alpha * t * grad_fx_dot_delta_x, verbose_prefix)
    while t > 1e-9 and (np.isnan(f(x + t*delta_x)) or f(x + t*delta_x) > fx + alpha * t * grad_fx_dot_delta_x):
        t *= beta
        if verbose:
            print '{4}t={0} fx={1} f(x + t*delta_x)={2} fx + alpha * t * grad_fx.dot(delta_x)={3}'.format(t, fx, f(x + t*delta_x), fx + alpha * t * grad_fx_dot_delta_x, verbose_prefix)
    if verbose:
        if t > 1e-9:
            print '{1}Found t={0}'.format(t, verbose_prefix)
        else:
            print 'No suitable t found. Using t=0'
            
    return t if t > 1e-9 else 0.

def save_list_of_arrays(filename, list_of_arrays, fmt='%.18e'):
    for i, arr in enumerate(list_of_arrays):
        np.savetxt(filename.format(i), arr, delimiter=',', fmt=fmt)

def pretty_str(p, decimal_places=2, join_str='\n'):
    '''Pretty-print a matrix or vector.'''
    if type(p) == list:
        return join_str.join([pretty_str(x) for x in p])
    if len(p.shape) == 1:
        return vector_str(p, decimal_places)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([vector_str(a, decimal_places) for a in p]))

def vector_str(p, decimal_places=2):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([style.format(a) for a in p]))

def make_directory(base, subdir):
    if not base.endswith('/'):
        base += '/'
    directory = base + subdir
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not directory.endswith('/'):
        directory = directory + '/'
    return directory

def save_args(args, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='=')
        for k,v in vars(args).iteritems():
            writer.writerow([k,v])

def get_numeric_header(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        line = reader.next()
        return np.array([float(x) for x in line])

def load_sparse_data_from_dense_file(filename, verbose=False):
    data = []
    row_coords = []
    col_coords = []
    if verbose:
        print 'Loading sparse data from dense file {0}'.format(filename)
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        # skip header
        reader.next()
        row = 0
        for line in reader:
            for col,val in enumerate(map(float,line)):
                if val == 0:
                    continue
                data.append(val)
                row_coords.append(row)
                col_coords.append(col)
            row += 1
            if verbose and row % 10000 == 0:
                print 'Row #{0}'.format(row)
    results = sps.coo_matrix((data, (row_coords, col_coords)))
    return results.tocsc()

def load_sparse_data_from_sparse_file(filename, nodes, verbose=False):
    '''
    Loads data from a sparse CSV file stored in the following format:
    ColumnID:Value,ColumnID:Value,...
    '''
    data = []
    row_coords = []
    col_coords = []
    col_count = sum([node.num_params for node in nodes])
    if verbose:
        print 'Loading sparse data from sparse file {0}'.format(filename)
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        # skip header
        row = 0
        for line in reader:
            for col,val in map(lambda x: (int(x.split(':')[0]), float(x.split(':')[1])),line):
                data.append(val)
                row_coords.append(row)
                col_coords.append(col)
            row += 1
            if verbose and row % 10000 == 0:
                print 'Row #{0}'.format(row)
    results = sps.coo_matrix((data, (row_coords, col_coords)), shape=(row,col_count))
    x = results.tocsc()
    
    if verbose:
        print 'Loaded sparse matrix with shape: {0}'.format(x.shape)
    # nutrients = x[:,0:64].todense()
    # print 'Max from first 48 columns:\n{0}'.format(nutrients.max(axis=0))
    # print 'Medians from first 48 columns:'
    # for i in np.arange(16):
    #     a = []
    #     for j in xrange(nutrients.shape[0]):
    #         if nutrients[j,i*4+3] != 0:
    #             a.append(nutrients[j,i*4+3])
    #     print '{0}: {1}'.format(i, np.median(np.array(a)))
    # raise Exception
    return x

def safe_exp(x):
    return np.exp(x.clip(-500.,500.))

def safe_sq(x):
    results = x.copy()
    results[np.abs(x) > 1e30] = np.inf
    return results ** 2

def save_pseudoedges(edges, filename):
    '''Saves pseudo-edges (a single node's edges)'''
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for edge,weights in edges:
            writer.writerow([edge] + list(weights.flatten()))

def load_pseudoedges(filename):
    '''Loads pseudo-edges (a single node's edges)'''
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        nodes = []
        weights = []
        for line in reader:
            nodes.append(int(line[0]))
            weights.append(np.array([float(x) for x in line[1:]]))
    return (np.array(nodes), weights)

def save_edges(edges, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for edge,weights in edges:
            writer.writerow([edge[0], edge[1]] + list(weights))

def load_edges(filename):
    '''
    Loads the edges for an MRF. The weights are loaded as a flat vector. The vector for
    the edge (i,j) is of the form [i1j1, i1j2, ... iNjM-1, iNjM].
    '''
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        edges = [((int(x[0]),int(x[1])),np.array([float(xi) for xi in x[2:]])) for x in reader]
    return edges

def get_node_map(nodes):
    '''Calculate the mapping from data column to node.'''
    cols = []
    for i,node in enumerate(nodes):
        cols.extend([i]*node.num_params)
    return np.array(cols)












