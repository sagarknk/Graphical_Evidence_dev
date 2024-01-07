This folder consists of files required to compute the log of marginal likelihood under the Wishart prior. 

1. Wishart_graphical_evidence.m
### Description: This matlab file implements our proposed procedure to compute log-marginal under the Wishart prior, according to the simulation settings mentioned in the paper. 
### Usage:
%%% set seed, q, n, $\alpha$, burnin and nmc (number of saved mc samples).
%%% set the scale matrix V (Scale_matrix). It is stored in the folder Scale_matrix.
%%% True precision matrix is generated from the function call wishrnd(scale_matrix, alpha).
%%% n x q data matrix is generated and stored in the variable "xx" and sample covariance matrix, S = xx'*xx.
%%% xx and S are stored in respective folders, which will be used to compute log-marginal from competing procedures.
%%% As closed form log-marginal exists for Wishart, it is computed. Following this, 25 random permutations of 1:q are generated and 
%%% log-marginal is computed for in the for loop
%%% Function calls to "Wishart_Hao_wang.m", "Wishart_last_col_fixed.m" are made to run the required Gibbs samplers for the procedure.



