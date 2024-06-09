function [start_point_first_gibbs, post_mean_omega_22_2ndGibbs, MC_average_Equation_11]  = ...
    G_wishart_last_col_fixed(S,n,burnin,nmc,delta,fixed_last_col,...
    scale_matrix_this_order_reduced,  G_mat_adj_this_order_reduced, Matrix_2be_added_Gibbs,...
    post_mean_omega)

%%% S: Sample covariance matrix
%%% n: sample size
%%% burnin: burn-in for MCMC
%%% nmc: number of samples to be saved after burn-in
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
omega_reduced_save = zeros(p-1,p-1,nmc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% S_reduced is sample covariance matrix for first p-1 columns, as the
%%% lastone remains fixed

S_reduced = S(1:(p-1), 1:(p-1));
p_reduced = p-1;
Matrix_2be_added_Gibbs_reduced = Matrix_2be_added_Gibbs(1:p_reduced, 1:p_reduced);
scale_matrix_required = scale_matrix_this_order_reduced(1:p_reduced, 1:p_reduced);
G_mat_adj_required = G_mat_adj_this_order_reduced(1:p_reduced, 1:p_reduced);

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%
Sigma = inv(post_mean_omega);
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

    s_22 = S(p,p);
    V_mat_22 = scale_matrix_this_order_reduced(p,p);

    %%%%% sample omega_22 %%%%%%%%%%%%%
    gamma_param = gamrnd(delta + n/2 + 1,2/(s_22+V_mat_22)); %% Variable name change
    % Sampling from the Gamma density of Equation (S.5) in the paper

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

        temp_matrix_2be_added = (1/omega_pp)*(fixed_last_col*fixed_last_col');
        Sigma_reduced = Sigma_11;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Omega_reduced_tilda = Omega_reduced - temp_matrix_2be_added;

        %%%%% sample Omega_11_tilde %%%%%%%%%%%%%
        for i=1:p_reduced

            V_mat_22 = scale_matrix_required(i,i);
            ind_noi = ind_noi_all(:,i);
            V_mat_12 = scale_matrix_required(ind_noi,i);

            s_21_tilda = S_reduced(ind_noi,i); s_22_tilda = S_reduced(i,i);
            vec_2be_added_21 = -1 * Matrix_2be_added_Gibbs_reduced(ind_noi,i);
            temp_vec_2be_added_21 = -1 * temp_matrix_2be_added(ind_noi,i);

            %%% Note the -1 above. This is done to make sure to respect the
            %%% crucial indicator function.

            gamma_param_tilda = gamrnd(delta + n/2 + 1, 2/(s_22_tilda+V_mat_22)); %% Variable name change

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            Sigma_11_reduced = Sigma_reduced(ind_noi,ind_noi); sigma_12_reduced = Sigma_reduced(ind_noi,i);
            sigma_22_reduced = Sigma_reduced(i,i);

            inv_Omega_11 = Sigma_11_reduced - sigma_12_reduced*sigma_12_reduced'/sigma_22_reduced;

            inv_C = (s_22_tilda+V_mat_22)*inv_Omega_11;
            %%%%%%%%%%%%%%%% finding which elements in beta to sample %%%%%%%%%
            G_mat_current_col = G_mat_adj_required(ind_noi,i);
            logi_which_ones = (G_mat_current_col ==1);
            logi_which_zeros = (G_mat_current_col ==0);

            if sum(logi_which_ones)>=1
                inv_C_required = inv_C(logi_which_ones, logi_which_ones);
                inv_C_chol_required = chol(inv_C_required);

                V_mat_12_required = V_mat_12(logi_which_ones,1);
                s_21_tilda_required = s_21_tilda(logi_which_ones,1);

                if sum(logi_which_zeros)>=1
                    vec_2be_added_21_required = vec_2be_added_21(logi_which_zeros,1);
                    inv_C_not_required = inv_C(logi_which_zeros, logi_which_ones);

                    vec_2be_added_21_required_modified = ...
                        vec_2be_added_21_required' * inv_C_not_required;

                    vec_2be_added_21_required_modified = ...
                        vec_2be_added_21_required_modified' ;

                    temp_vec_2be_added_21_required = temp_vec_2be_added_21(logi_which_zeros,1);

                    temp_vec_2be_added_21_required_modified = ...
                        temp_vec_2be_added_21_required' * inv_C_not_required;

                    temp_vec_2be_added_21_required_modified = ...
                        temp_vec_2be_added_21_required_modified' ;


                    mu_i_reduced = -inv_C_required\...
                        (V_mat_12_required + s_21_tilda_required + ...
                        vec_2be_added_21_required_modified + ...
                        temp_vec_2be_added_21_required_modified);

                    beta_reduced = ...
                        mu_i_reduced +  inv_C_chol_required\randn(sum(logi_which_ones),1);
                    % Sampling from the Normal density of Equation (S.5) in the paper

                else
                    mu_i_reduced = -inv_C_required\(V_mat_12_required + s_21_tilda_required);
                    beta_reduced = mu_i_reduced +  inv_C_chol_required\randn(sum(logi_which_ones),1);
                    % Sampling from the Normal density of Equation (S.5) in the paper
                end


                beta = zeros(p_reduced-1,1);

                if sum(logi_which_ones)== p_reduced-1
                    beta(logi_which_ones,1) = beta_reduced;
                else
                    beta(logi_which_ones,1) = beta_reduced;
                    beta(logi_which_zeros,1) = vec_2be_added_21_required + ...
                        temp_vec_2be_added_21_required;
                end

            else
                beta =vec_2be_added_21 + temp_vec_2be_added_21;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            omega_12 = beta; omega_22 = gamma_param_tilda + beta'*inv_Omega_11*beta;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Omega_reduced_tilda(i,ind_noi) = omega_12; Omega_reduced_tilda(ind_noi,i) = omega_12;
            Omega_reduced_tilda(i,i) = omega_22;

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

        Omega_reduced = Omega_reduced_tilda + ...
            (1/omega_pp)*(fixed_last_col*fixed_last_col');

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
        V_mat_22 = scale_matrix_required(1,1);
        gamma_param =  gamrnd(delta + n/2 + 1, 2/(s_22+V_mat_22)); %% Variable name change
        Omega_reduced = gamma_param + fixed_last_col'*inv(omega_pp)*fixed_last_col;
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
        omega_reduced_save(:,:,iter-burnin) = Omega_reduced;

    elseif iter > burnin && p_reduced==1
        inv_omega_11_save(1,iter-burnin) = 1/Omega_reduced;
        omega_save_pp(1,iter-burnin) = omega_pp;
        omega_reduced_save(:,:,iter-burnin) = Omega_reduced;

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
post_mean_omega_22_2ndGibbs = mean(omega_save_pp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vec_log_gamma_density = ones(1,nmc);
vec_log_gamma_density = -Inf.*vec_log_gamma_density;

%%% We are starting with a vector of -Infinity because if the
%%% indicator condition is not met, then the likelihood is zero,
%%% and the log-likelihood is -Infinity

%ind_noi = [1:p-1]';

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
            delta + n/2 + 1 , ...
            2/(S(p,p)+...
            scale_matrix_this_order_reduced(p, p))));
    else
        % do nothing
    end

end

MC_average_Equation_11 = log(mean(exp(vec_log_gamma_density)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% Updating the start_point_first_gibbs %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ADDED NEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start_point_first_gibbs = mean(omega_reduced_save,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

