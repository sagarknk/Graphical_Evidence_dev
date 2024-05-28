function [post_mean_omega,post_mean_tau_save, MC_average_Equation_9] = ...
    GHS_Hao_wang(S,n,burnin, nmc,lambda,t_dof,Matrix_2be_added_Gibbs)

%%% S: Sample covariance matrix
%%% n: sample size
%%% burnin: burn-in for MCMC
%%% nmc: number of samples to be saved after burn-in

[p] = size(S,1);

omega_save = zeros(p,p,nmc);
tau_sq_save = zeros(p,p,nmc);

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
Sigma = inv(Omega);

TAU_sq(1:p,1:p) = 1;
Nu(1:p,1:p) = 1;
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
        tau_sq_12  = TAU_sq(ind_noi,i);
        nu_12 = Nu(ind_noi,i);
        %%% sample gamma and beta
        
        gamma_param = gamrnd(n/2 + 1 , 2/(s_22 + 1/lambda));   
        %%% gamma with shape=n/2 + 1, rate=(s_22+1/lambda)/2 or scale = 2/(s_22 + 1/lambda)
        % Sampling from the Gamma density of Equation (15) in the paper

        inv_Omega_11 = Sigma_11 - sigma_12*sigma_12'/sigma_22;
        
        inv_C = diag(1./(tau_sq_12*lambda*lambda)) + (s_22 + 1/lambda)*inv_Omega_11; 
        inv_C_chol = chol(inv_C);
        mu_i = -inv_C\(s_21  + vec_2be_added_21./(tau_sq_12*lambda*lambda));

        if  iter > burnin && i == p
            inv_C_store(:,:,iter - burnin) = inv_C;
            mean_vec_store(:,iter-burnin) = mu_i;
        end

        beta = mu_i+ inv_C_chol\randn(p-1,1);
        % Sampling from the Normal density of Equation (15) in the paper

        omega_12 = beta; omega_22 = gamma_param + beta'*inv_Omega_11*beta;
        
        %%% sample tau_sq and nu
        rate = omega_12.^2/(2*lambda^2)+1./nu_12;
        tau_sq_12 = 1./gamrnd(1,1./rate);    % random inv gamma with shape=1, rate=rate
        nu_12 = 1./gamrnd(1,1./(1+1./tau_sq_12));    % random inv gamma with shape=1, rate=1+1/lambda_sq_12
      
        %%% update Omega
        Omega(i,ind_noi) = omega_12; Omega(ind_noi,i) = omega_12;
        Omega(i,i) = omega_22;
        
        temp = inv_Omega_11*beta;
        Sigma_11 = inv_Omega_11 + temp*temp'/gamma_param;
        sigma_12 = -temp/gamma_param; sigma_22 = 1/gamma_param;
        Sigma(ind_noi,ind_noi) = Sigma_11; Sigma(i,i) = sigma_22;
        Sigma(i,ind_noi) = sigma_12; Sigma(ind_noi,i) = sigma_12;
        
        TAU_sq(i,ind_noi) = tau_sq_12; TAU_sq(ind_noi,i) = tau_sq_12;
        Nu(i,ind_noi) = nu_12; Nu(ind_noi,i) = nu_12;
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if iter > burnin
        omega_save(:,:,iter-burnin) = Omega;
        tau_sq_save(:,:,iter-burnin) = TAU_sq;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
post_mean_omega = mean(omega_save,3);
post_mean_tau_save = mean(tau_sq_save,3);
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

