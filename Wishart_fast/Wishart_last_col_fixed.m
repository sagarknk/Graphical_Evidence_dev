function [post_mean_omega_22_2ndGibbs, MC_average_Equation_11] = ...
    Wishart_last_col_fixed(S,n,burnin,nmc,dof,fixed_last_col)

%%% S: Sample covariance matrix
%%% n: sample size
%%% burnin: burn-in for MCMC
%%% nmc: number of samples to be saved after burn-in
%%% dof: degree of freedom of wishart (alpha in the main code)
%%% fixed_last_col: \omega_12^*

[p] = size(S,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if p-1~=1
    inv_omega_11_save = zeros(p-1,p-1,nmc);
else
    inv_omega_11_save = zeros(1,nmc);
end
omega_save_pp = zeros(1,nmc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% S_reduced is sample covariance matrix for first p-1 columns, as the
%%% lastone remains fixed

S_reduced = S(1:(p-1), 1:(p-1));
p_reduced = p-1;

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

%Omega_reduced = eye(p_reduced);
omega_pp = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%
Sigma = eye(p);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iter = 1:(burnin + nmc)

    %%% First we update omega_pp which is nothing but sampling \omega_22
    %%% with \omega_12^* held fixed

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% ADDDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%
    Sigma_11 = Sigma([1:p_reduced]',[1:p_reduced]');
    sigma_12 = Sigma([1:p_reduced]',p);
    sigma_22 = Sigma(p,p);
    inv_Omega_11 = Sigma_11 - sigma_12*sigma_12'/sigma_22;
    s_22 = S(p,p);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%% sample omega_22 %%%%%%%%%%%%%
    gamma_param = gamrnd((dof + n -p +1)/2,2/(s_22+1)); %% Variable name change
    % Sampling from the Gamma density of Equation (10) in the paper

    omega_pp =  gamma_param + ...
        fixed_last_col'*inv_Omega_11*fixed_last_col ;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    temp = inv_Omega_11*fixed_last_col;
    Sigma_11 = inv_Omega_11 + temp*temp'/gamma_param;
    sigma_12 = -temp/gamma_param;
    sigma_22 = 1/gamma_param;
    Sigma([1:p_reduced]',[1:p_reduced]') = Sigma_11;
    Sigma(p,p) = sigma_22;
    Sigma(p,[1:p_reduced]') = sigma_12;
    Sigma([1:p_reduced]',p)= sigma_12;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    if p_reduced ~=1

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Sigma_reduced = Sigma_11;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%% sample Omega_11_tilde %%%%%%%%%%%%%
        for i=1:p_reduced

            ind_noi = ind_noi_all(:,i);
            s_21_tilda = S_reduced(ind_noi,i); s_22_tilda = S_reduced(i,i);
            gamma_param_tilda = gamrnd((dof + n -p +1)/2, 2/(s_22_tilda+1));

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            Sigma_11_reduced = Sigma_reduced(ind_noi,ind_noi); sigma_12_reduced = Sigma_reduced(ind_noi,i);
            sigma_22_reduced = Sigma_reduced(i,i);

            inv_Omega_11 = Sigma_11_reduced - sigma_12_reduced*sigma_12_reduced'/sigma_22_reduced;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            inv_C = (s_22_tilda+1)*inv_Omega_11;
            mu_i = -inv_C\s_21_tilda;
            inv_C_chol = chol(inv_C);
            beta = mu_i+ inv_C_chol\randn(p_reduced-1,1);
            % Sampling from the Normal density of Equation (10) in the paper

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            temp = inv_Omega_11*beta;
            Sigma_11_reduced = inv_Omega_11 + temp*temp'/gamma_param_tilda;
            sigma_12_reduced = -temp/gamma_param_tilda;
            sigma_22_reduced = 1/gamma_param_tilda;
            Sigma_reduced(ind_noi,ind_noi) = Sigma_11_reduced;
            Sigma_reduced(i,i) = sigma_22_reduced;
            Sigma_reduced(i,ind_noi) = sigma_12_reduced;
            Sigma_reduced(ind_noi,i) = sigma_12_reduced;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Sigma([1:p_reduced]',[1:p_reduced]') = Sigma_reduced;
        sigma_12 = -Sigma_reduced*fixed_last_col*(1/omega_pp);
        sigma_22 = (1/omega_pp) + (1/omega_pp)*fixed_last_col'*Sigma_reduced*fixed_last_col*(1/omega_pp);
        Sigma(p,p) = sigma_22;
        Sigma(p,[1:p_reduced]') = sigma_12;
        Sigma([1:p_reduced]',p)= sigma_12;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    else
        s_22 = S_reduced(1,1);
        gamma_param =  gamrnd((dof + n -p +1)/2, 2/(s_22+1));
        Omega_reduced = gamma_param + fixed_last_col'*(1/omega_pp)*fixed_last_col;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Sigma = inv([Omega_reduced, fixed_last_col;fixed_last_col, omega_pp]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if iter > burnin && p_reduced~=1
        inv_omega_11_save(:,:,iter-burnin) = Sigma_reduced - sigma_12*sigma_12'/sigma_22;
        omega_save_pp(1,iter-burnin) = omega_pp;

    elseif iter > burnin && p_reduced==1
        inv_omega_11_save(1,iter-burnin) = 1/Omega_reduced;
        omega_save_pp(1,iter-burnin) = omega_pp;

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
post_mean_omega_22_2ndGibbs = mean(omega_save_pp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vec_log_gamma_density = ones(1,nmc);
vec_log_gamma_density = -Inf.*vec_log_gamma_density;

%%% We are starting with a vector of -Infinity because if the
%%% indicator condition is not met, then the likelihood is zero,
%%% and the log-likelihood is -Infinity


for sample_index = 1:nmc

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if p_reduced~=1
        inv_Omega_11 = inv_omega_11_save(:,:, sample_index);
    else
        inv_Omega_11 = inv_omega_11_save(1, sample_index);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    temp_gamma_val = post_mean_omega_22_2ndGibbs  - ...
        fixed_last_col'*inv_Omega_11*fixed_last_col;

    if(temp_gamma_val > 0)
        vec_log_gamma_density(1, sample_index) = ...
            log(gampdf(temp_gamma_val,...
            (dof +  n - p+1)/2 , ...
            2/(S(p,p)+1)));
    else
        % do nothing
    end

end

MC_average_Equation_11 = log(mean(exp(vec_log_gamma_density)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

