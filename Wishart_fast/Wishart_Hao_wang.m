function [post_mean_omega, MC_average_Equation_9]  = ...
    Wishart_Hao_wang(S,n,burnin, nmc,dof)

%%% S: Sample covariance matrix
%%% n: sample size
%%% burnin: burn-in for MCMC
%%% nmc: number of samples to be saved after burn-in
%%% dof: degree of freedom of wishart (alpha in the main code)

[p] = size(S,1);

omega_save = zeros(p,p,nmc);

%%% The three initializations below are for computing the Normal density
%%% in the evaluation of the term IV_{p-j+1}

vec_log_normal_density = zeros(1,nmc);
inv_C_store = zeros(p-1,p-1,nmc);
mean_vec_store = zeros(p-1, nmc);

%%%% ind_noi_all stores the indicices {1,2,...p}\{i} for the i^th column

ind_noi_all = zeros(p-1,p);
for i = 1:p
    if i==1
        ind_noi = [2:p]';
    elseif i==p
        ind_noi = [1:p-1]';
    else
        ind_noi = [1:i-1,i+1:p]';
    end

    ind_noi_all(:,i) = ind_noi;
end

% set initial values
Omega = eye(p);

for iter = 1:(burnin+nmc)

    %%% Gibb's sampler for Omega with Hao-Wang's decomposition

    for i = 1:p
        ind_noi = ind_noi_all(:,i);
        s_21 = S(ind_noi,i); s_22 = S(i,i);
        %%% sample gamma and beta

        gamma = gamrnd((dof + n -p +1)/2 , 2/(s_22 + 1));
        % gamma with shape=(dof + n -p +1)/2, rate=(s_22+1)/2 or scale = 2/(s_22 + 1)
        % Sampling from the Gamma density of Equation (8) in the paper

        inv_Omega_11 = inv(Omega(ind_noi, ind_noi));
        inv_C = (s_22 + 1)*inv_Omega_11;
        inv_C_chol = chol(inv_C);
        mu_i = -inv_C\s_21;

        if  iter > burnin && i == p
            inv_C_store(:,:,iter - burnin) = inv_C;
            mean_vec_store(:,iter-burnin) = mu_i;
        end

        beta = mu_i+ inv_C_chol\randn(p-1,1);
        % Sampling from the Normal density of Equation (8) in the paper

        omega_12 = beta; omega_22 = gamma + beta'*inv_Omega_11*beta;

        %%% update Omega, Sigma, Lambda_sq, Nu
        Omega(i,ind_noi) = omega_12; Omega(ind_noi,i) = omega_12;
        Omega(i,i) = omega_22;
    end

    if iter > burnin
        omega_save(:,:,iter-burnin) = Omega;
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
post_mean_omega = mean(omega_save,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ind_noi = ind_noi_all(:,p);

for sample_index  = 1:nmc

    mean_vec = mean_vec_store(:, sample_index);
    inv_C = inv_C_store(:,:, sample_index);

    %%% mean vector of the normal densities
    %%% That it, computing each term in the sum in Equation (9) of the
    %%% paper

    vec_log_normal_density(1, sample_index)= ...
        log(mvnpdf(post_mean_omega(p,ind_noi),...
        mean_vec', inv(inv_C)));
end

MC_average_Equation_9 = log(mean(exp(vec_log_normal_density)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

