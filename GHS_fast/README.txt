This folder consists of files required to compute the log of marginal likelihood under the Graphical horseshoe (GHS) prior. 

1. GHS_graphical_evidence.m
### Description: This matlab file implements our proposed procedure to compute log-marginal under the Graphical horseshoe prior, according to the simulation settings mentioned in the paper. 
### Usage:
%%% set seed, q, n, $\lambda$, burnin and nmc (number of saved mc samples).
%%% True precision matrix is generated from the function call prior_sampling(q,burnin,nmc, lambda, 0).
%%% n x q data matrix is generated and stored in the variable "xx" and sample covariance matrix, S = xx'*xx.
%%% xx and S are stored in respective folders, which will be used to compute log-marginal from competing procedures.
%%% If q=2, exact log-marginal is computed. Following this, 25 random permutations of 1:q are generated and 
%%% log-marginal is computed in the for loop
%%% Function calls to "GHS_Hao_wang.m", "GHS_last_col_fixed.m" are made to run the required Gibbs samplers for the procedure.

2
