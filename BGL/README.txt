This folder consists of files required to compute the log of marginal likelihood under the Bayesian graphical lasso (BGL) prior. 

1. BGL_graphical_evidence.m
### Description: This matlab file implements our proposed procedure to compute log-marginal under the Bayesian graphical lasso, according to the simulation settings mentioned in the paper. 
### Usage:
%%% set seed, q, n, $\lambda$, burnin and nmc (number of saved mc samples).
%%% True precision matrix is generated from the function call prior_sampling(q,burnin,nmc, lambda, 0).
%%% n x q data matrix is generated and stored in the variable "xx" and sample covariance matrix, S = xx'*xx.
%%% xx and S are stored in respective folders, which will be used to compute log-marginal from competing procedures.
%%% If q=2, exact log-marginal is computed. Following this, 25 random permutations of 1:q are generated and 
%%% log-marginal is computed in the for loop
%%% Function calls to "BGL_Hao_wang.m", "BGL_last_col_fixed.m" are made to run the required Gibbs samplers for the procedure.
%%% Function calls to "gigrnd.m" to sample inverse-Gaussian latent parameters ($\tau_{ij}$), which is a special case of sampling from Generalized Inverse-Gaussian (GIG) - gigrnd


