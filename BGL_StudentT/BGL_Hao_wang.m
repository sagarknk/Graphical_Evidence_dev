function [post_mean_omega,post_mean_tau_save, MC_average_Equation_9] = ...
    BGL_Hao_wang(S,n,burnin, nmc,lambda,t_dof,Matrix_2be_added_Gibbs)

%%% S: Sample covariance matrix
%%% n: sample size
%%% burnin: burn-in for MCMC
%%% nmc: number of samples to be saved after burn-in

[p] = size(S,1);

omega_save = zeros(p,p,nmc); %%% To store all posterior samples
tau_save = zeros(p,p,nmc); %%% To store all posterior samples

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

%%% set initial values
Omega = eye(p);
Sigma = inv(Omega);

TAU(1:p,1:p) = 1;


%%%%%%%%%%%%%%%%%
S_orig = S;
%%%%%%%%%%%%%%%%%
for iter = 1:(burnin+nmc)
    
    shape_posterior_u_ = (n*p + t_dof)/2;
    scale_posterior_u_ = 2/(t_dof + trace(S_orig * Omega));
    posterior_u_ = gamrnd(shape_posterior_u_, scale_posterior_u_);
    
    S = posterior_u_ .* S_orig;
    %%% Gibb's sampler for Omega with Hao-Wang's decomposition
    for i = 1:p

        ind_noi = ind_noi_all(:,i);

        Sigma_11 = Sigma(ind_noi,ind_noi); sigma_12 = Sigma(ind_noi,i);
        sigma_22 = Sigma(i,i);

        s_21 = S(ind_noi,i); s_22 = S(i,i);
        vec_2be_added_21 = Matrix_2be_added_Gibbs(ind_noi,i);
        tau_12  = TAU(ind_noi,i);
        %%% sample gamma and beta

        gamma_param = gamrnd(n/2 + 1 , 2/(s_22 + lambda));
        % Gamma with shape=n/2 + 1, rate=(s_22+lambda)/2
        % or scale = 2/(s_22 + lambda)
        % Sampling from the Gamma density of Equation (15) in the paper

        inv_Omega_11 = Sigma_11 - sigma_12*sigma_12'/sigma_22;

        inv_C = diag(1./tau_12) + (s_22 + lambda)*inv_Omega_11;
        inv_C_chol = chol(inv_C);
        mu_i = -inv_C\(s_21  + vec_2be_added_21./tau_12);

        if  iter > burnin && i == p
            inv_C_store(:,:,iter - burnin) = inv_C;
            mean_vec_store(:,iter-burnin) = mu_i;
        end

        beta = mu_i+ inv_C_chol\randn(p-1,1);
        % Sampling from the Normal density of Equation (15) in the paper

        omega_12 = beta; omega_22 = gamma_param + beta'*inv_Omega_11*beta;

        %%% update Omega
        Omega(i,ind_noi) = omega_12; Omega(ind_noi,i) = omega_12;
        Omega(i,i) = omega_22;

        temp = inv_Omega_11*beta;
        Sigma_11 = inv_Omega_11 + temp*temp'/gamma_param;
        sigma_12 = -temp/gamma_param; sigma_22 = 1/gamma_param;
        Sigma(ind_noi,ind_noi) = Sigma_11; Sigma(i,i) = sigma_22;
        Sigma(i,ind_noi) = sigma_12; Sigma(ind_noi,i) = sigma_12;

        %%% TAU
        %%%% This is inverse Gaussian with mu' = sqrt(lambda^2/omega_ij^2)
        %%%% and lambda' = lambda^2

        mu_prime = sqrt(lambda^2./...
            ((omega_12 + vec_2be_added_21).*(omega_12 + vec_2be_added_21)));
        lambda_prime = lambda^2;


        %%% sampler for inverse-Gaussian from Generalized Inverse-Gaussian
        %%% This sampler is more efficient and stable than the sampler
        %%% for inverse-Gaussian from wiki

        a_gig_tau = lambda_prime./(mu_prime.^2);
        b_gig_tau = lambda_prime;
        u_12 = zeros(p-1,1);
        for tau_idx = 1:p-1
            u_12(tau_idx,1) = gigrnd(-1/2,a_gig_tau(tau_idx,1), b_gig_tau,1);
        end
        tau_12 = 1./u_12;

        TAU(i,ind_noi) = tau_12;
        TAU(ind_noi,i) = tau_12;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if iter > burnin
        omega_save(:,:,iter-burnin) = Omega;
        tau_save(:,:,iter-burnin) = TAU;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
post_mean_omega = mean(omega_save,3);
post_mean_tau_save = mean(tau_save,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

