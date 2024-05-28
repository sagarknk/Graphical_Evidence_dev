function [post_mean_omega_22_2ndGibbs, MC_average_Equation_11] = ...
    GHS_last_col_fixed(S,n,burnin,nmc,lambda,t_dof,fixed_last_col,...
    Matrix_2be_added_Gibbs, post_mean_omega, post_mean_tau_sq_save)

%%% S: Sample covariance matrix
%%% n: sample size
%%% burnin: burn-in for MCMC
%%% nmc: number of samples to be saved after burn-in
%%% fixed_last_col: \omega_12^*

[p] = size(S,1);
omega_save = zeros(p,p,nmc);
posterior_u_save = zeros(1,nmc);
%tau_sq_save = zeros(p,p,nmc);

%%% S_reduced is sample covariance matrix for first p-1 columns, as the
%%% lastone remains fixed

S_reduced = S(1:(p-1), 1:(p-1));
p_reduced = p-1;
Matrix_2be_added_Gibbs_reduced = Matrix_2be_added_Gibbs(1:p_reduced, 1:p_reduced);

%%%% ind_noi_all stores the indicices {1,2,...p}\{i} for the i^th column

if p_reduced ~=1
    ind_noi_all = zeros(p_reduced-1,p_reduced);
    for i = 1:p_reduced
        if i==1
            ind_noi = [2:p_reduced]';
        elseif i==p_reduced
            ind_noi = [1:p_reduced-1]';
        else
            ind_noi = [1:i-1,i+1:p_reduced]';
        end
        
        ind_noi_all(:,i) = ind_noi;
    end
else
    % do nothing 
end

Omega_reduced = post_mean_omega(1:p_reduced, 1:p_reduced);

TAU_sq_reduced(1:p_reduced, 1:p_reduced) = ...
    post_mean_tau_sq_save(1:p_reduced, 1:p_reduced);
Nu_reduced(1:p_reduced,1:p_reduced) = 1;
omega_pp = [];

full_Omega = post_mean_omega;
S_orig = S;
S_reduced_orig = S_reduced;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iter = 1:(burnin + nmc)
    
    %%%%%%%%%%%%%%%%%%%%%%
    
    shape_posterior_u_ = (n*p + t_dof)/2;
    scale_posterior_u_ = 2/(t_dof + trace(S_orig * full_Omega));
    posterior_u_ = gamrnd(shape_posterior_u_, scale_posterior_u_);
    
    S = posterior_u_ .* S_orig;
    S_reduced = posterior_u_ .* S_reduced_orig;
    
    %%%%%%%%%%%%%%%%%%%%%%

    %if(mod(iter,1000)==0)
        %fprintf('iter = %d \n',iter);
    %end
    
    %%% First we update omega_pp which is nothing but sampling \omega_22
    %%% with \omega_12^* held fixed
    
    inv_Omega_11 = inv(Omega_reduced);
    s_22 = S(p,p);
    
    %%%%% sample omega_22 %%%%%%%%%%%%%
    gamma_param = gamrnd(n/2 + 1,2/(s_22+1/lambda));
    % Sampling from the Gamma density of Equation (16) in the paper
    omega_pp =  gamma_param + ...
        fixed_last_col'*inv_Omega_11*fixed_last_col ; 
    
    if p_reduced~=1
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Omega_reduced_tilda = Omega_reduced - ...
            (1/omega_pp)*(fixed_last_col*fixed_last_col');
        
        temp_matrix_2be_added = (1/omega_pp)*(fixed_last_col*fixed_last_col');
        
        %%%%% sample Omega_11_tilde %%%%%%%%%%%%%
        for i=1:p_reduced
            
            ind_noi = ind_noi_all(:,i);
            s_21_tilda = S_reduced(ind_noi,i); s_22_tilda = S_reduced(i,i);
            vec_2be_added_21 = Matrix_2be_added_Gibbs_reduced(ind_noi,i);
            
            temp_vec_2be_added_21 = temp_matrix_2be_added(ind_noi,i);
            tau_sq_12  = TAU_sq_reduced(ind_noi,i);
            nu_12 = Nu_reduced(ind_noi,i);
            gamma_param_tilda = gamrnd(n/2 + 1, 2/(s_22_tilda+1/lambda));
            
            tilda_W_11 = Omega_reduced_tilda(ind_noi, ind_noi);
            inv_Omega_11 = inv(tilda_W_11);
            
            inv_C = diag(1./(tau_sq_12*lambda*lambda)) + (s_22_tilda + 1/lambda)*inv_Omega_11;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            mu_i = -inv_C\(s_21_tilda + vec_2be_added_21./(tau_sq_12*lambda*lambda) ...
                 + temp_vec_2be_added_21./(tau_sq_12*lambda*lambda));
            
            inv_C_chol = chol(inv_C);
            beta = mu_i+ inv_C_chol\randn(p_reduced-1,1);
            % Sampling from the Normal density of Equation (16) in the paper

            omega_12 = beta; omega_22 = gamma_param_tilda + beta'*inv_Omega_11*beta;
            %%% sample tau_sq and nu
            rate = omega_12.^2/(2*lambda^2)+1./nu_12;
            tau_sq_12 = 1./gamrnd(1,1./rate);    % random inv gamma with shape=1, rate=rate
            nu_12 = 1./gamrnd(1,1./(1+1./tau_sq_12));    % random inv gamma with shape=1, rate=1+1/lambda_sq_12
      
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Omega_reduced_tilda(i,ind_noi) = omega_12; Omega_reduced_tilda(ind_noi,i) = omega_12;
            Omega_reduced_tilda(i,i) = omega_22;
            
            TAU_sq_reduced(i,ind_noi) = tau_sq_12; TAU_sq_reduced(ind_noi,i) = tau_sq_12;
            Nu_reduced(i,ind_noi) = nu_12; Nu_reduced(ind_noi,i) = nu_12;
            
        end
        
        Omega_reduced = Omega_reduced_tilda + ...
            (1/omega_pp)*(fixed_last_col*fixed_last_col');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        full_Omega = [Omega_reduced, fixed_last_col; fixed_last_col',omega_pp];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        s_22 = S_reduced(1,1);
        gamma_param =  gamrnd(n/2 + 1, 2/(s_22+1/lambda));
        Omega_reduced = gamma_param + fixed_last_col'*inv(omega_pp)*fixed_last_col;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        full_Omega = [Omega_reduced, fixed_last_col; fixed_last_col',omega_pp];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    if iter > burnin
        %omega_save(1:p_reduced,1:p_reduced,iter-burnin) = Omega_reduced;
        %omega_save(p, 1:p_reduced, iter-burnin) = fixed_last_col';
        %omega_save(1:p_reduced, p , iter-burnin) = fixed_last_col;
        %omega_save(p,p,iter-burnin) = omega_pp;
        
        omega_save(:,:,iter-burnin) = full_Omega;
        posterior_u_save(1,iter-burnin) = posterior_u_;

        %tau_sq_save(1:p_reduced,1:p_reduced,iter-burnin) = TAU_sq_reduced;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
post_mean_omega_22_2ndGibbs = mean(omega_save(p, p, :));
vec_log_gamma_density = ones(1,nmc);
vec_log_gamma_density = -Inf.*vec_log_gamma_density;

%%% We are starting with a vector of -Infinity because if the
%%% indicator condition is not met, then the likelihood iz zero,
%%% and the log-likelihood is -Infinity

ind_noi = [1:p-1]';

for sample_index = 1:nmc
    Omega_11 = omega_save(ind_noi,ind_noi, sample_index);
    inv_Omega_11 = inv(Omega_11);

    temp_gamma_val = post_mean_omega_22_2ndGibbs  - ...
        fixed_last_col'*inv_Omega_11*fixed_last_col;

    if(temp_gamma_val > 0)
        vec_log_gamma_density(1, sample_index) = ...
            log(gampdf(temp_gamma_val,...
            n/2 + 1 , ...
            2/(posterior_u_save(1,sample_index) .*S_orig(p,p)+1/lambda)));
    else
        % do nothing
    end

end

MC_average_Equation_11 = log(mean(exp(vec_log_gamma_density)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

