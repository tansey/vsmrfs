import numpy as np
import scipy.sparse as sps
from utils import *
from copy import deepcopy
from functools import partial
import sys


class MixedMRFNode:
    def __init__(self, expfam, 
                    rel_tol=1e-6,
                    edge_tol=1e-3,
                    converge_tol=1e-4,
                    max_steps=100,
                    newton_max_steps=30,
                    admm_alpha = 100.,
                    admm_inflate = 2.,
                    quality_metric='bic',
                    verbose=1):
        self.expfam = expfam

        self.rel_tol = rel_tol
        self.edge_tol = edge_tol
        self.converge_tol = converge_tol
        self.max_steps = max_steps
        self.newton_max_steps = newton_max_steps
        self.admm_alpha = admm_alpha
        self.admm_inflate = admm_inflate
        self.quality_metric = quality_metric
        self.verbose = verbose
        self.admm_alpha_starting = admm_alpha

    def set_data(self, sufficient_stats, neighbor_stats, neighbors, sample_weights=None):
        if self.verbose:
            print 'Setting data and caching relevant statistics.'

        self.sufficient_stats = sufficient_stats # n x p
        self.neighbor_stats = neighbor_stats # n x q
        self.neighbors = neighbors
        self.sample_weights = sample_weights # n x 1
        self.n = sample_weights.sum() if self.sample_weights is not None else float(sufficient_stats.shape[0])

        # Handle sparse datasets (e.g. Ising models)
        # Note: We assume the data is in scipy's CSC format
        self.sparse_data = sps.issparse(neighbor_stats)

        self.num_params = self.expfam.num_params
        self.theta_rows = self.expfam.num_params
        self.theta_cols = self.neighbor_stats.shape[1] + 1

        if self.sparse_data:
            self.sq_neighbor_stats = self.neighbor_stats.multiply(self.neighbor_stats) # n x q
            self.edge_sizes = np.array([(neighbors == i).sum() * self.num_params for i in np.unique(neighbors).astype(np.int32)], dtype=np.int32) # q x 1
            # CSC format
            self.mean_B = np.zeros((self.num_params, self.theta_cols), dtype=np.float32) # p x q+1
            if self.sample_weights is not None:
                self.mean_B[:,0] = weighted_column_sum(self.sufficient_stats, self.sample_weights) / self.n
                # for j in xrange(self.sq_neighbor_stats.shape[1]):
                #     self.sq_neighbor_stats[:,j] = self.sq_neighbor_stats[:,j].T.multiply(self.sample_weights).T
            else:
                self.mean_B[:,0] = self.sufficient_stats.mean(axis=0)
            for i in xrange(self.num_params):
                for j in xrange(self.theta_cols-1):
                    if self.verbose:
                        print 'Averaging B[{0},{1}]'.format(i,j+1)
                    if self.sample_weights is not None:
                        self.mean_B[i,j+1] = self.sufficient_stats[:,i].T.dot(self.neighbor_stats[:,j].T.multiply(self.sample_weights).T)[0,0] / self.n
                    else:    
                        self.mean_B[i,j+1] = self.sufficient_stats[:,i].T.dot(self.neighbor_stats[:,j])[0,0] / self.n
        else:
            self.sq_neighbor_stats = self.neighbor_stats * self.neighbor_stats
            self.edge_sizes = np.array([(neighbors == i).sum() * self.expfam.num_params for i in np.unique(neighbors)])
            self.mean_B = np.zeros((self.expfam.num_params, self.neighbor_stats.shape[1] + 1))
            total = 0.0
            for i,(node_stats, neighbor_stats) in enumerate(zip(self.sufficient_stats, self.neighbor_stats)):
                w = 1.0 if self.sample_weights is None else self.sample_weights[i]
                total += w
                update_pct = w / total
                remainder_pct = (total-w) / total
                self.mean_B[:,0] = update_pct * node_stats + remainder_pct * self.mean_B[:,0]
                self.mean_B[:,1:] = update_pct * node_stats[:,np.newaxis] * np.tile(neighbor_stats, (self.expfam.num_params, 1)) + remainder_pct * self.mean_B[:,1:]


    def log_likelihood(self, theta):
        eta = self.natural_params(theta).T # n x p
        return self.expfam.log_likelihood(eta, self.sufficient_stats)

    def reset(self):
        self.admm_alpha = self.admm_alpha_starting

    def solution_path(self, lambda1_range=(0.0001,1000), lambda2_range=(0.0001,1000), lambda1_bins=50, lambda2_bins=50):
        '''Performs a grid search for the best lambda values via BIC'''
        # initialize grid search
        lambda1_grid = np.exp(np.linspace(np.log(lambda1_range[1]), np.log(lambda1_range[0]), lambda1_bins))
        lambda2_grid = np.exp(np.linspace(np.log(lambda2_range[1]), np.log(lambda2_range[0]), lambda2_bins))
        aic_trace = np.zeros((lambda1_bins, lambda2_bins)) # The AIC score for each lambda value
        aicc_trace = np.zeros((lambda1_bins, lambda2_bins)) # The AICc score for each lambda value (correcting for finite sample size)
        bic_trace = np.zeros((lambda1_bins, lambda2_bins)) # The BIC score for each lambda value
        dof_trace = np.zeros((lambda1_bins, lambda2_bins)) # The degrees of freedom of each final solution
        edge_count_trace = np.zeros((lambda1_bins, lambda2_bins)) # The number of edges found in each solution
        log_likelihood_trace = np.zeros((lambda1_bins, lambda2_bins))
        edges_trace = [] # The edges found in each solution
        best_lambda1 = None
        best_lambda2 = None
        best_score = None
        best_results = None
        lambda1_warmstart = None
        starting_max_steps = self.max_steps
        starting_admm_inflate = self.admm_inflate

        # for each lambda1, lambda2 pair
        for i,lambda1 in enumerate(lambda1_grid):
            results = lambda1_warmstart
            edges_trace.append([])
            for j,lambda2 in enumerate(lambda2_grid):
                print 'Lambda1={0} Lambda2={1}'.format(lambda1, lambda2)
                sys.stdout.flush()

                # Reset any adaptive fields like admm_alpha
                self.reset()

                # TODO: TEMP for speedup
                # if j == 0:
                #     self.max_steps = starting_max_steps
                #     self.admm_inflate = starting_admm_inflate
                #     self.admm_alpha = self.admm_alpha_starting
                # else:
                #     self.max_steps = 50
                #     self.admm_inflate = 1. # TEMP
                #     self.admm_alpha = 1.67 # TEMP

                # run the MLE procedure with warm starts
                results = self.mle(lambda1=lambda1, lambda2=lambda2, initial_values=results)

                # save trace of results
                edges_trace[i].append(results['edges'])
                edge_count_trace[i,j] = results['edge_count']

                # save trace of various metrics
                aic_trace[i,j] = results['aic']
                aicc_trace[i,j] = results['aicc']
                bic_trace[i,j] = results['bic']
                dof_trace[i,j] = results['dof']
                log_likelihood_trace[i,j] = results['log_likelihood']

                # Get the score for these penalty settings
                if self.quality_metric == 'bic':
                    score = bic_trace[i,j]
                elif self.quality_metric == 'aic':
                    score = aic_trace[i,j]
                elif self.quality_metric == 'aicc':
                    score = aicc_trace[i,j]
                else:
                    raise Exception('Unknown quality metric: {0}'.format(self.quality_metric))

                if self.verbose:
                    print 'Log-Likelihood: {0}'.format(results['log_likelihood'])
                    print 'DoF: {0}'.format(results['dof'])
                    print 'AIC: {0}'.format(results['aic'])
                    print 'AICc: {0}'.format(results['aicc'])
                    print 'BIC: {0}'.format(results['bic'])

                # Track the best results
                if best_score is None or score < best_score:
                    best_score = score
                    best_results = results
                    best_lambda1 = lambda1
                    best_lambda2 = lambda2

                # Use a cross-cutting warm start
                if j == 0:
                    lambda1_warmstart = results

        if self.verbose:
            print '--- Solution Path Finished ---'
            print 'Best values: lambda1 = {0} lambda2 = {1}'.format(best_lambda1, best_lambda2)
            print 'Edges from best: {0}'.format(best_results['edges'])
            
        # return trace results and best point via BIC
        return {'best': best_results,
                'lambda1': best_lambda1,
                'lambda2': best_lambda2,
                'lambda1_grid': lambda1_grid,
                'lambda2_grid': lambda2_grid,
                'dof': dof_trace,
                'edges': edges_trace,
                'edge_count': edge_count_trace,
                'log_likelihood': log_likelihood_trace,
                'aic': aic_trace,
                'aicc': aicc_trace,
                'bic': bic_trace}

    def natural_params(self, theta):
        # eta size = p
        # neighbors = q
        # samples = n
        # data is n x p+q
        # theta is p x 1+q
        # sufficient_stats is n x p
        # neighbor_stats is n x q
        return (self.neighbor_stats.dot(theta[:,1:].T) + theta[:,0]) # n x p
        

    def feasible_start_f(self, t, theta, constraints, s_idx, theta_idx, x):
        # Get the parameters from the x vector
        s = x[s_idx]
        theta = x[theta_idx[0]:theta_idx[-1]].reshape(theta.shape)
        
        # Get the natural parmaeters and constraints
        eta = self.natural_params(theta)
        constraints = np.hstack(self.expfam.eta_constraints(eta))

        # Calculate the objective function
        np.seterr(all='raise')
        try:
            return t*s - np.log(-constraints + s).sum()
        except FloatingPointError:
            return np.inf

    def find_feasible_point(self):
        '''Find a feasible starting point for theta via gradient descent.'''
        if self.verbose:
            print 'Searching for a feasible starting x that satisfies all constraints.'

        # Initialize the weight matrix
        theta = np.zeros((self.theta_rows, self.theta_cols))

        # Initialize the parameter and gradient vectors. Our variables are s and theta
        x = np.zeros(1 + theta.shape[0]*theta.shape[1])
        grad_fx = np.zeros(x.shape)
        diag_hess_fx = np.zeros(x.shape) # use a diagonal approximation of the hessian
        delta_x = np.zeros(x.shape)

        # Initialize some useful indices
        s_idx = 0
        theta_idx = np.array([theta.shape[1] * i + 1 for i in xrange(theta.shape[0]+1)])

        # Get the natural parameter version (B.theta)
        eta = self.natural_params(x[theta_idx[0]:theta_idx[-1]].reshape(theta.shape))

        # Get the constraints
        constraints = self.expfam.eta_constraints(eta)

        # Initialize the barrier method parameters
        m = constraints.shape[0] * constraints.shape[1] if len(constraints.shape) == 2 else constraints.shape[0]
        mu = 30.
        t = m / 100. # start with a duality gap of 100. See p. 571 of Boyd.
        outer_steps = 0

        # Handle the edge case where there are no parameter constraints
        if m == 0:
            theta = x[theta_idx[0]:theta_idx[-1]].reshape(theta.shape)
            if self.verbose:
                print 'No constraints apply to this exponential family distribution.'
                print 'Starting theta: {0}'.format(pretty_str(theta))
                print 'Checking... (will crash if not valid)'
                np.seterr(all='raise')
                eta = self.natural_params(theta).T
                self.expfam.log_partition(eta)
                print 'passed.'
            return theta
        
        while m/t >= 1e-6:
            # Get the natural parameter version (B.theta)
            eta = self.natural_params(x[theta_idx[0]:theta_idx[-1]].reshape(theta.shape))

            # Get the constraints
            constraints = self.expfam.eta_constraints(eta)

            # Find the maximum starting value
            x[s_idx] = np.hstack(constraints).max()+1

            converged = self.rel_tol+1.0
            steps = 0

            if self.verbose > 1:
                print '\tOuter step #{0}'.format(outer_steps)
                print '\tStarting x: {0}'.format(x)

            while converged > self.rel_tol and x[s_idx] >= -0.00001:
                # Create the objective function
                f = partial(self.feasible_start_f, t, theta, constraints, s_idx, theta_idx)

                if self.verbose > 1:
                    print '\t\tStep #{0}'.format(steps)
                    
                # Get the natural parameters (B.theta) and their constraints
                eta = self.natural_params(x[theta_idx[0]:theta_idx[-1]].reshape(theta.shape))
                constraints = self.expfam.eta_constraints(eta)
                
                # Get the f_i(x) values
                fx_i = constraints - x[0]
                grad_constraints = self.expfam.grad_eta_constraints(eta)
                diag_hess_constraints = self.expfam.diagonal_hessian_eta_constraints(eta)
                sq_grad_constraints = grad_constraints * grad_constraints
                sq_fx_i = fx_i * fx_i

                grad_fx[0] = t + (1.0 / np.hstack(fx_i)).sum() # s gradient
                diag_hess_fx[0] = (1.0 / np.hstack(sq_fx_i)).sum() # s hessian
                delta_x[0] = -grad_fx[0] / diag_hess_fx[0]
                for j,c in enumerate(grad_constraints): # for every natural parameter j
                    # Skip this parameter if we have no constraints on it.
                    if c.shape[0] == 0:
                        continue
                    grad_base = grad_constraints[j] / -fx_i[j]
                    hess_base = sq_grad_constraints[j] / sq_fx_i[j] - diag_hess_constraints[j] / fx_i[j]
                    grad_fx[theta_idx[j]] = grad_base.sum() # bias gradient
                    diag_hess_fx[theta_idx[j]] = hess_base.sum() # bias hessian
                    grad_fx[theta_idx[j]+1:theta_idx[j+1]] = weighted_column_sum(self.neighbor_stats, grad_base)
                    diag_hess_fx[theta_idx[j]+1:theta_idx[j+1]] = weighted_column_sum(self.sq_neighbor_stats, hess_base).clip(self.rel_tol)
                    delta_x[theta_idx[j]:theta_idx[j+1]] = -grad_fx[theta_idx[j]:theta_idx[j+1]] / diag_hess_fx[theta_idx[j]:theta_idx[j+1]]

                converged = grad_fx.dot(-delta_x) / 2.

                if self.verbose > 1:
                    print '\t\tConvergence Criteria: {0}'.format(converged)
                
                if converged <= self.rel_tol:
                    break

                inner_t = backtracking_line_search(x, f, delta_x, grad_fx, verbose=self.verbose>1)

                x += inner_t*delta_x

                steps += 1

                if self.verbose > 1:
                    print '\t\tConvergence Criteria: {0}'.format(converged)
                    
            # Update the t precision variable
            t *= mu

            if self.verbose > 1:
                print '\tConverged! s: {0}'.format(x[s_idx])

            outer_steps += 1

            if x[s_idx] < 0:
                theta = x[theta_idx[0]:theta_idx[-1]].reshape(theta.shape)
                if self.verbose:
                    print 'Found a solution satisfying the constraint matrix of shape {0} (s={1})'.format(constraints.shape, x[s_idx])
                    print 'Starting theta: {0}'.format(pretty_str(theta))
                    print 'Checking... (will crash if not valid)'
                    np.seterr(all='raise')
                    eta = self.natural_params(theta).T
                    self.expfam.log_partition(eta)
                    print 'passed.'
                return theta

        print 'Infeasible problem. No solution is possible.'
        return None


    def mle(self, lambda1=0.3, lambda2=0.3,
                  initial_values=None):
        '''Maximum likelihood estimation for a node-wise conditional via ADMM.'''
        # initialize values
        if initial_values:
            theta = initial_values['theta']
            z = initial_values['z']
            u = initial_values['u']
        else:
            theta = self.find_feasible_point()
            if theta is None:
                raise Exception("No feasible starting point.")
            z = np.zeros(theta.shape)
            u = np.zeros(theta.shape)

        converged = self.converge_tol+1.0
        steps = 0

        # while not converged
        while converged > self.converge_tol and steps < self.max_steps:
            if self.verbose:
                print 'ADMM step #{0}'.format(steps)
                print '\tUpdating theta...'

            # update theta
            theta = self.theta_update(theta, z, u)

            if self.verbose:
                print '\tUpdating z...'

            z_old = np.copy(z) # track so we can calculate the dual residual
            y = self.admm_alpha * (theta + u) # cache for use in the z-update

            # Over-relaxation extension
            # relaxation = 1.67
            # theta_accel = relaxation * theta + (1 - relaxation) * z
            # y = self.admm_alpha * (theta_accel + u)

            # update z
            for j,p in enumerate(self.edge_sizes):
                if self.neighbors[0] == j:
                    z[:,self.neighbors == j] = theta[:,self.neighbors==j] + u[:,self.neighbors == j]
                else:
                    s_j = _soft_threshold(y[:,self.neighbors == j], lambda2)
                    norm_s_j = np.linalg.norm(s_j)
                    p_lambda = np.sqrt(p) * lambda1
                    numerator = (norm_s_j - p_lambda).clip(0) * s_j
                    # Handle the edge-case where the weights truly go to zero
                    if np.allclose(numerator, 0):
                        z[:,self.neighbors == j] = 0
                        continue
                    z[:,self.neighbors == j] = numerator / (self.admm_alpha * norm_s_j + p_lambda * (1. - self.admm_alpha))

            # calculate residuals
            primal_residual = theta - z
            #primal_residual[:,1:] = theta[:,1:] - z[:,1:] # TEMP
            #primal_residual[:,0] = 0 # TEMP
            #primal_residual = theta_accel - z # over-relaxation
            dual_residual = self.admm_alpha * (z_old - z)

            if self.verbose:
                print '\tUpdating u...'

            # update u
            u += primal_residual

            # Check convergence
            # TODO: I think this should be np.sqrt(np.mean(x ** 2))
            # primal_resnorm = np.linalg.norm(primal_residual)
            # dual_resnorm = np.linalg.norm(dual_residual)
            primal_resnorm = np.sqrt(np.mean(primal_residual ** 2))
            dual_resnorm = np.sqrt(np.mean(dual_residual ** 2))
            converged = max(primal_resnorm, dual_resnorm)


            # Varying penalty parameter extension
            if primal_resnorm > 5 * dual_resnorm:
                self.admm_alpha *= self.admm_inflate
                u /= self.admm_inflate
            elif dual_resnorm > 5 * primal_resnorm:
                self.admm_alpha /= self.admm_inflate
                u *= self.admm_inflate
                # TEMP
                if self.admm_alpha < 1.67:
                    multiplier = 1.67 / self.admm_alpha
                    self.admm_alpha *= multiplier
                    u /= multiplier
            if self.verbose:
                print '\tadmm alpha: {0}'.format(self.admm_alpha)

            if self.verbose:
                print '\tdual_resnorm: {0:.6f}'.format(dual_resnorm)
                print '\tprimal_resnorm: {0:.6f}'.format(primal_resnorm)
                print '\tconvergence: {0:.6f}'.format(converged)
                print '\ttheta:\t{0}'.format(pretty_str(theta[:,:30]).replace('\n', '\n\t\t'))
                print '\tz:\t{0}'.format(pretty_str(z[:,:30]).replace('\n', '\n\t\t'))
                print '\tu:\t{0}'.format(pretty_str(u[:,:30]).replace('\n', '\n\t\t'))
                print '\tlog_likelihood: {0}'.format(self.log_likelihood(theta))

            # Update step counter
            steps += 1

        if self.verbose:
            print 'Finished!'

        # Get the final log-likelihood
        log_likelihood = self.log_likelihood(theta)

        # Force the sparsity to conform to the lambda penalty
        #theta[:,1:][np.abs(theta[:,1:]) <= lambda2] = 0

        # Calculate degrees of freedom
        #dof = np.sum(np.abs(theta) > self.edge_tol) + 1.

        # Calculate the edges
        dof = 0
        edge_count = 0
        edges = []
        for j,p in enumerate(self.edge_sizes):
            edge_weight = theta[:, self.neighbors == j]
            edge_norm = np.linalg.norm(np.abs(edge_weight))
            #if edge_norm > self.edge_tol:
            if np.abs(edge_weight).max() > self.edge_tol:
                if self.verbose:
                    print 'Norm {0}: {1}'.format(j, edge_norm)
                edges.append((j,edge_weight))
                edge_count += 1
                dof += np.sum(np.abs(edge_weight) > self.edge_tol) # Calculate degrees of freedom

        # Calculate AIC = 2k - 2ln(L)
        aic = 2. * dof - 2. * log_likelihood
        
        # Calculate AICc = AIC + 2k * (k+1) / (n - k - 1)
        aicc = aic + 2 * dof * (dof+1) / max(self.n - dof - 1., 1.)

        # Calculate BIC = -2ln(L) + k * (ln(n) - ln(2pi))
        bic = -2 * log_likelihood + dof * (np.log(self.n) - np.log(2 * np.pi))

        if self.verbose:
            print '--- Relevant statistics ---'
            print 'DoF: {0}'.format(dof)
            print '# Edges: {0} ({1})'.format(edge_count, edges)
            print 'Log-Likelihood: {0}'.format(log_likelihood)
            print 'AIC: {0}'.format(aic)
            print 'AICc: {0}'.format(aicc)
            print 'BIC: {0}'.format(bic)
            print 'Theta: {0}'.format(pretty_str(theta[:,:30], decimal_places=4))

        # return resulting weights
        return {'theta': theta, 'z': z, 'u': u, 'dof': dof, 'edge_count': edge_count, 'edges': edges, 'log_likelihood': log_likelihood, 'aic': aic, 'aicc': aicc, 'bic': bic}

    def theta_update_f(self, theta, z, u, x):
        theta = x.reshape(theta.shape)
        eta = self.natural_params(theta).T # p x n

        try:
            if self.sample_weights is None:
                mean_log_partition = self.expfam.log_partition(eta).sum() / self.n
            else:
                mean_log_partition = self.expfam.log_partition(eta).dot(self.sample_weights) / self.n
            return (-self.mean_B * theta).sum() \
                    + self.admm_alpha / 2. * np.linalg.norm(theta - z + u) ** 2 \
                    + mean_log_partition
        except FloatingPointError:
            return np.inf

    def theta_update_grad_f(self, z, u, theta):
        eta = self.natural_params(theta).T # p x n
        grad_a = self.expfam.grad_log_partition(eta) # p x n

        if self.sample_weights is not None:
            for i in xrange(grad_a.shape[0]):
                grad_a[i,:] *= self.sample_weights

        g = -self.mean_B + self.admm_alpha * (theta - z + u)
        g[:,0] += grad_a.sum(axis=1) / self.n
        if self.sparse_data:
            for i in xrange(g.shape[0]):
                g[i, 1:] += weighted_column_sum(self.neighbor_stats, grad_a[i]).A1 / self.n
        else:
            g[:,1:] += grad_a.dot(self.neighbor_stats) / self.n

        return g

    def theta_update_diagonal_hessian_f(self, theta):
        eta = self.natural_params(theta).T # p x n
        hess_a = self.expfam.diagonal_hessian_log_partition(eta) # p x n

        if self.sample_weights is not None:
            for i in xrange(hess_a.shape[0]):
                hess_a[i,:] *= self.sample_weights

        h = np.zeros(theta.shape)
        h += self.admm_alpha
        h[:,0] += hess_a.sum(axis=1) / self.n
        if self.sparse_data:
            for i in xrange(h.shape[0]):
                h[i, 1:] += weighted_column_sum(self.sq_neighbor_stats, hess_a[i]).A1 / self.n
        else:
            h[:,1:] += hess_a.dot(self.sq_neighbor_stats) / self.n
        return h

    def theta_update(self, theta, z, u):
        '''Update theta via an interior point (log-barrier + Newton) method.'''
        np.seterr(under='ignore')
        # Initialize the parameter and gradient vectors. Our variables are s and theta
        x = theta.flatten()
        grad_fx = np.zeros(x.shape)
        diag_hess_fx = np.zeros(x.shape) # use a diagonal approximation of the hessian

        # Initialize some useful indices
        theta_idx = np.array([theta.shape[1] * i for i in xrange(theta.shape[0]+1)])

        # Create the objective function
        f = partial(self.theta_update_f, theta, z, u)
        g = partial(self.theta_update_grad_f, z, u)
        h = partial(self.theta_update_diagonal_hessian_f)

        converged = self.rel_tol + 1
        steps = 0

        while converged > self.rel_tol and steps < self.newton_max_steps:
            if self.verbose > 1:
                print '\t\tInner Step #{0}'.format(steps)

            theta = x.reshape(theta.shape)

            # Get the gradient and (diagonal) hessian of the original objective, scaled for log-barrier form
            grad_fx = g(theta).flatten()
            diag_hess_fx = h(theta).flatten()

            if self.verbose > 2:
                print '\t\tgrad_fx:          {0}'.format(pretty_str(grad_fx))
                print '\t\tdiagonal hess_fx: {0}'.format(pretty_str(h(theta)))
                
            delta_x = -grad_fx / diag_hess_fx # quasi-newton delta
            #delta_x = -grad_fx # gradient descent delta

            converged = np.abs(grad_fx.dot(-delta_x) / 2.)

            if self.verbose > 1:
                print '\t\t\tGrad_fx criteria: {0}'.format(np.linalg.norm(grad_fx))
                print '\t\t\tConvergence Criteria: {0}'.format(converged)
            
            # Quasi-Newton cutoff
            if converged <= self.rel_tol:
                break

            inner_t = backtracking_line_search(x, f, delta_x, grad_fx, verbose=self.verbose>2, verbose_prefix='\t\t\t')

            # If we aren't moving, then just exit.
            if inner_t == 0:
                break

            x += inner_t*delta_x

            # Gradient descent cutoff
            # if np.linalg.norm(grad_fx) <= self.rel_tol:
            #     break

            steps += 1 

        theta = x.reshape(theta.shape)
        
        if self.verbose > 1:
            print '\t\tTheta solution found: {0}'.format(pretty_str(theta.flatten()))
            print '\t\tChecking... (will crash if not valid)'

        np.seterr(all='raise')
        eta = self.natural_params(theta).T
        self.expfam.log_partition(eta)
        np.seterr(under='ignore')

        if self.verbose > 1:
            print '\t\tpassed.'

        return theta     
            

def _soft_threshold(x, _lambda):
    return np.sign(x) * (np.abs(x) - _lambda).clip(0)

def weighted_column_sum(m, v):
    if sps.issparse(m):
        # CSC format
        data = m.data * np.take(v, m.indices)
        return sps.csc_matrix((data, m.indices, m.indptr), shape=m.shape).sum(axis=0)
    else:
        return (v[:,np.newaxis] * m).sum(axis=0)




