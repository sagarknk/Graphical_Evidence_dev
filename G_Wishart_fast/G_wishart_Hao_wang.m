function [post_mean_omega, MC_average_Equation_9]  = ...
    G_wishart_Hao_wang(S,n,burnin, nmc,delta,...
    scale_matrix_this_order_reduced,  G_mat_adj_this_order_reduced, Matrix_2be_added_Gibbs, start_point_first_gibbs)

%%% S: Sample covariance matrix
%%% n: sample size
%%% burnin: burn-in for MCMC
%%% nmc: number of samples to be saved after burn-in

[p] = size(S,1);

omega_save = zeros(p,p,nmc);

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

%%% The three initializations below are for computing the Normal density
%%% in the evaluation of the term IV_{p-j+1}

vec_log_normal_density = zeros(1,nmc);

G_mat_last_col = G_mat_adj_this_order_reduced(1:p-1,p);
logi_which_ones = (G_mat_last_col ==1);

if sum(logi_which_ones)>=1
    inv_C_required_store = zeros(sum(logi_which_ones),sum(logi_which_ones),nmc);
    mean_vec_store = zeros(sum(logi_which_ones),nmc);
else
    % do nothing 
end


% set initial value
Omega = start_point_first_gibbs;

for iter = 1:(burnin+nmc)
    
    %%% Gibb's sampler for Omega with Hao-Wang's decomposition
    
    for i = 1:p
        V_mat_22 = scale_matrix_this_order_reduced(i,i);
        ind_noi = ind_noi_all(:,i);
        V_mat_12 = scale_matrix_this_order_reduced(ind_noi,i);
        
        s_21 = S(ind_noi,i); s_22 = S(i,i);
        vec_2be_added_21 = -1 * Matrix_2be_added_Gibbs(ind_noi,i);
        %%% Note the -1 above. This is done to make sure to respect the
        %%% crucial indicator function. 
        
        %%% sample gamma and beta
        
        gamma = gamrnd(delta + n/2 +1 , 2/(s_22 + V_mat_22));   
        %%% gamma with shape=delta + n/2 +1, rate=(s_22+V_mat_22)/2 or scale = 2/(s_22 + V_mat_22)
        % Sampling from the Gamma density of Equation (S.2) in the paper

        inv_Omega_11 = inv(Omega(ind_noi, ind_noi));
        inv_C = (s_22 + V_mat_22)*inv_Omega_11; 
        
        %%%%%%%%%%%%%%%% finding which elements in beta to sample %%%%%%%%%
        G_mat_current_col = G_mat_adj_this_order_reduced(ind_noi,i);
        logi_which_ones = (G_mat_current_col ==1);
        logi_which_zeros = (G_mat_current_col ==0);
        
        if sum(logi_which_ones)>=1
            inv_C_required = inv_C(logi_which_ones, logi_which_ones);
            inv_C_chol_required = chol(inv_C_required);
            
            V_mat_12_required = V_mat_12(logi_which_ones,1);
            s_21_required = s_21(logi_which_ones,1);
            
            if sum(logi_which_zeros)>=1
                vec_2be_added_21_required = vec_2be_added_21(logi_which_zeros,1);
                inv_C_not_required = inv_C(logi_which_zeros, logi_which_ones);
                
                vec_2be_added_21_required_modified = ...
                    vec_2be_added_21_required' * inv_C_not_required;
                
                vec_2be_added_21_required_modified = ...
                    vec_2be_added_21_required_modified' ;

                mu_i_reduced = -inv_C_required\...
                    (V_mat_12_required + s_21_required + vec_2be_added_21_required_modified);

                if  iter > burnin && i == p
                    inv_C_required_store(:,:,iter - burnin) = inv_C_required;
                    mean_vec_store(:,iter-burnin) = mu_i_reduced;
                end

                beta_reduced = ...
                    mu_i_reduced +  inv_C_chol_required\randn(sum(logi_which_ones),1);
                % Sampling from the Normal density of Equation (S.2) in the paper

            else
                mu_i_reduced = -inv_C_required\(V_mat_12_required + s_21_required);

                if  iter > burnin && i == p
                    inv_C_required_store(:,:,iter - burnin) = inv_C_required;
                    mean_vec_store(:,iter-burnin) = mu_i_reduced;
                end

                beta_reduced = mu_i_reduced +  inv_C_chol_required\randn(sum(logi_which_ones),1);
                % Sampling from the Normal density of Equation (S.2) in the paper
            end
            
            
            beta = zeros(p-1,1);
            
            if sum(logi_which_ones)== p-1
                beta(logi_which_ones,1) = beta_reduced;
            else
                beta(logi_which_ones,1) = beta_reduced;
                beta(logi_which_zeros,1) = vec_2be_added_21_required;
            end
            
        else
            beta = vec_2be_added_21;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        omega_12 = beta; omega_22 = gamma + beta'*inv_Omega_11*beta;
        
        %%% update Omega
        Omega(i,ind_noi) = omega_12; Omega(ind_noi,i) = omega_12;
        Omega(i,i) = omega_22;
    end
    
    if iter > burnin
        omega_save(:,:,iter-burnin) = Omega;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
post_mean_omega = mean(omega_save,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G_mat_last_col = G_mat_adj_this_order_reduced(1:p-1,p);
logi_which_ones = (G_mat_last_col ==1);

for sample_index  = 1:nmc
    
    if sum(logi_which_ones)>=1

        inv_C_required = inv_C_required_store(:,:,sample_index);
        mean_vec = mean_vec_store(:,sample_index);

        %%% mean vector of the normal densities
        %%% That it, computing each term in the sum in Equation (S.3) of the
        %%% paper

        vec_log_normal_density(1, sample_index)= ...
            log(mvnpdf(post_mean_omega(p,logi_which_ones),...
            mean_vec', inv(inv_C_required)));
    else
        vec_log_normal_density(1, sample_index)= 0;
    end
end

MC_average_Equation_9 = log(mean(exp(vec_log_normal_density)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

