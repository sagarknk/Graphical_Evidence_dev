This folder consists of files required to compute the log of marginal likelihood under the G-Wishart prior. 

1. G_Wishart_graphical_evidence.m
### Description: This matlab file implements our proposed procedure to compute log-marginal under the G-Wishart prior, according to the simulation settings mentioned in the paper. 
### Usage:
%%% set seed, q, n, $\delta$.
%%% set the adjacency matrix G (G_mat_adj) which defines the constraints of zero entries in $\Omega$. It is stored in the folder G_mat.
%%% set the scale matrix V (Scale_matrix). It is stored in the folder Scale_matrix.
%%% True precision matrix is generated from the function call prior_sampling(q,100,100, G_mat_adj, scale_matrix, delta, 0).
%%% n x q data matrix is generated and stored in the variable "xx" and sample covariance matrix, S = xx'*xx.
%%% xx and S are stored in respective folders, which will be used to compute log-marginal from competing procedures.
%%% set burnin and nmc (number of saved mc samples). Following this, 25 random permutations of 1:q are generated and stored in Matrix_of_rand_order.
%%% log-marginal is computed in the for loop
%%% Function calls to "G_Wishart_Hao_wang.m", "G_Wishart_last_col_fixed.m" are made to run the required Gibbs samplers for the procedure.
%%% Function calls to "logmvgamma.m" are made to compute the log of multivariate gamma function when required. 










