rng(123456789)
xx_all = csvread("DLBC_AKT_MTOR_pathways.csv",1,0);
xx = xx_all(1:20,:);

[n,q] = size(xx);
S = xx'*xx;

%%% total_num_lambda permutations of vector 1:q
lambda_sequence = 0.1:0.01:5;
total_num_lambda = length(lambda_sequence);

Matrix_of_log_ratio_likelihoods = zeros(total_num_lambda, q);
log_data_likelihood = zeros(total_num_lambda, q);
log_posterior_density = zeros(total_num_lambda, q);

log_normal_posterior_density = zeros(total_num_lambda, q);
log_gamma_posterior_density = zeros(total_num_lambda, q);

direct_eval_log_prior_density = zeros(1,total_num_lambda);

Harmonic_mean_est_vec = zeros(1,total_num_lambda);

bad_lambda = zeros(1,total_num_lambda);
burnin = 1e3;
nmc = 5e3;


parfor num_lambda = 1:total_num_lambda   
    %tic;
    lambda = lambda_sequence(1, num_lambda);
    try
        random_order = 1:q;
        
        fprintf("Marginal likelihood is being computed for %dth lambda\n", num_lambda);
        
        log_ratio_of_liklelihoods = zeros(1,q);
        %%%% temporary storage for I_{p-j+1} -IV_{p-j+1} for
        %%%% every j from 1 to p.
        
        Cell_storage = cell(q,1);
        %%% storage of posterior means i.e., \tilde{\theta}_{p-j+1}^* for every
        %%% j from 1 to p.
        
        Matrix_to_be_added = zeros(q,q);
        %%%% The above matrix keeps track of the linear shifts that we are
        %%%% supposed to do every step. 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for num_BGL = 1:q
           
            if num_BGL<= q-1
                %fprintf("Working with %d th telescopic sum\n", num_BGL);
                reduced_data_xx = xx(:,random_order(1,1:(q-num_BGL+1)));
                
                [~,q_reduced] = size(reduced_data_xx);
                S = reduced_data_xx'*reduced_data_xx;
                
                Matrix_2be_added_Gibbs = ...
                    Matrix_to_be_added(1:q_reduced, 1:q_reduced);
                
                %%% Run the unrestricted sampler to get samples, which will
                %%% be used to approximate the Normal density in the
                %%% evaluation of the term IV_j
                
                [omega_save, tau_save] = ...
                    BGL_Hao_wang(S,n,burnin,nmc,lambda, Matrix_2be_added_Gibbs);
                
                post_mean_omega = mean(omega_save,3);
                fixed_last_col = post_mean_omega(1:(q_reduced-1),q_reduced);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%% computing harmonic estimate when num_BGL = 1 or when the full
                %%%%% data matrix is considered %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                %if num_BGL == 1
                %    temp_normal_log_likli_at_posterior_samples = zeros(1,nmc);
                %    for post_sample = 1:nmc
                %        temp_normal_log_likli_at_posterior_samples(post_sample) = ...
                %            sum(log((mvnpdf(reduced_data_xx,0, inv(omega_save(:,:,post_sample))))));
                %    end
                %    
                %    temp_numerical_1 = -temp_normal_log_likli_at_posterior_samples - ...
                %        median(-temp_normal_log_likli_at_posterior_samples);
                %    
                %    mean_temp_numerical_1 = mean(exp(temp_numerical_1));
                %    
                %    Harmonic_mean_est = -median(-temp_normal_log_likli_at_posterior_samples) ...
                %        -log(mean_temp_numerical_1);
                %    
                %    Harmonic_mean_est_vec(num_lambda) = Harmonic_mean_est;
                %else
                %    % do nothing
                %end
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                column_number = q - num_BGL + 1;
                %%% Also to be noted that column_number = q_reduced
                
                
                %%%%%%%%%%% computing <4> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%% Posterior likelihood at posterior mean %%%%%%%%%%%%%%%%%%%%%%%%
                
                %%% ind_noi_all has indices of non-diagonal entries for each column
                %%% for first column it has numbers 2 to q-1. For second column it has
                %%% numbers 1, {3,...,q} and so on.
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                ind_noi_all = zeros(q_reduced-1,q_reduced);
                for i = 1:q_reduced
                    if i==1
                        ind_noi = [2:q_reduced]';
                    elseif i==q_reduced
                        ind_noi = [1:q_reduced-1]';
                    else
                        ind_noi = [1:i-1,i+1:q_reduced]';
                    end
                    
                    ind_noi_all(:,i) = ind_noi;
                end
                
                ind_noi = ind_noi_all(:,column_number);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                vec_log_normal_density = zeros(1,nmc);
                
                for sample_index  = 1:nmc
                    Omega_11 = omega_save(ind_noi,ind_noi, sample_index);
                    tau_12 = tau_save(ind_noi, column_number, sample_index);
                    inv_Omega_11 = inv(Omega_11);
                    D_tau = diag(tau_12);
                    inv_C = inv(D_tau) + ...
                        (S(column_number, column_number)+lambda)*inv_Omega_11;
                    
                    vec_2be_added_21 = Matrix_2be_added_Gibbs(ind_noi,column_number);
                    s_12_mod = S(ind_noi, column_number) + ...
                        vec_2be_added_21./tau_12;
                    
                    %%% mean vector of the normal (density of beta)
                    
                    mean_vec = -1*inv_C\s_12_mod;
                    
                    vec_log_normal_density(1, sample_index)= ...
                        log(mvnpdf(post_mean_omega(column_number,ind_noi),...
                        mean_vec', inv(inv_C)));
                end
                
                %%% Run the restricted sampler to get samples, which will
                %%% be used to approximate the truncated Gamma density in the
                %%% evaluation of the term IV_j
                
                post_mean_tau_save = mean(tau_save,3);
                omega_save_2ndGibbs = ...
                    BGL_last_col_fixed(S,n,burnin, nmc,lambda, fixed_last_col, ...
                    Matrix_2be_added_Gibbs, post_mean_omega, post_mean_tau_save);
                
                post_mean_omega_22_2ndGibbs = mean(omega_save_2ndGibbs(q_reduced, q_reduced, :));
                
                vec_log_gamma_density = ones(1,nmc);
                vec_log_gamma_density = -Inf.*vec_log_gamma_density;
                %%% We are starting with a vector of -Infinity because if the
                %%% indicator condition is not met, then the likelihood is zero,
                %%% and the log-likelihood is -Infinity
                
                for sample_index = 1:nmc
                    Omega_11 = omega_save_2ndGibbs(ind_noi,ind_noi, sample_index);
                    inv_Omega_11 = inv(Omega_11);
                    
                    temp_gamma_val = post_mean_omega_22_2ndGibbs  - ...
                        fixed_last_col'*inv_Omega_11*fixed_last_col;
                    
                    if(temp_gamma_val > 0)
                        vec_log_gamma_density(1, sample_index) = ...
                            log(gampdf(temp_gamma_val,...
                            n/2 + 1 , ...
                            2/(S(column_number,column_number)+lambda)));
                    else
                        % do nothing
                    end
                    
                end
                
                log_normal_posterior_density(num_lambda, num_BGL) = ...
                    log(mean(exp(vec_log_normal_density)));
                
                log_gamma_posterior_density(num_lambda, num_BGL) = ...
                    log(mean(exp(vec_log_gamma_density)));
                
                log_posterior_density(num_lambda, num_BGL)= ...
                    log(mean(exp(vec_log_normal_density)))+...
                    log(mean(exp(vec_log_gamma_density)));
                
                %%%%%%%%%% Computing <1> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%% Data likelihood (conditonal of multivatiate normal) %%%%%%%%%%%%
                %%%% computing mean, sd and then the data likelihood
                
                st_dev = sqrt(1/post_mean_omega_22_2ndGibbs);
                mean_vec = -1*reduced_data_xx(:,setdiff(1:q_reduced, column_number))*...
                    post_mean_omega(column_number,setdiff(1:q_reduced,column_number))'...
                    *st_dev*st_dev;
                
                log_data_likelihood(num_lambda, num_BGL) = ...
                    sum(log(normpdf(reduced_data_xx(:,column_number), mean_vec, st_dev)));
                
                %%%%%%%%%% Computing <3> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%% As conditional prior desnities are not available, we
                %%% will evaluate the sum of all terms III at the end
                %%% (after the else case [when dealing with just y_1 or \omega_{11}], 
                %%% which is coming up next)
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                log_ratio_of_liklelihoods(1,num_BGL) = ...
                    log_data_likelihood(num_lambda, num_BGL) ...
                    - log_posterior_density(num_lambda, num_BGL);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                Matrix_to_be_added(1:(q-num_BGL), 1:(q-num_BGL)) = ...
                    Matrix_to_be_added(1:(q-num_BGL), 1:(q-num_BGL)) + ...
                    (1/post_mean_omega_22_2ndGibbs)*(fixed_last_col*fixed_last_col');
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                Cell_storage{num_BGL} = [fixed_last_col', post_mean_omega_22_2ndGibbs];
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            else
                %%% To evaluate log f(y_1) or the last row in the telescopic sum. Here
                %%% we compute I_1 and IV_1, II_1 is because no further columns of the
                %%% data are left and III_1 is taken care of the joint prior evaluation
                
                xx_reduced = xx(:,random_order(1,1));
                S = xx_reduced'*xx_reduced;
                q_reduced = 1;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                omega_save_direct = zeros(q_reduced,q_reduced,nmc);
                for i = 1:nmc
                    omega_save_direct(:,:,i) = gamrnd(n/2 + 1, 2/(lambda + S));
                end
                
                post_mean_omega_direct = mean(omega_save_direct,3);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%% Changing Cell_storage for easy handling
                Cell_storage{q} = post_mean_omega_direct;
                Cell_storage_copy = cell(q,1);
                
                for column_index = 1:q
                    Cell_storage_copy{column_index} = Cell_storage{q-column_index+1};
                end
                
                Cell_storage = Cell_storage_copy;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%% post_mean_omega resconstruction %%%%%%%%%%%%%%%%%%%%%
                
                Reconstructed_matrix = zeros(q,q);
                for column_index = 1:q
                    Reconstructed_matrix(column_index, 1:column_index) = Cell_storage{column_index};
                    Reconstructed_matrix(1:column_index, column_index) = Cell_storage{column_index}';
                end
                
                for column_index = 1:(q-1)
                    temp_length = length(Cell_storage{column_index+1});
                    temp_col_vec = Cell_storage{column_index+1}(1,1:(temp_length-1));
                    temp_col_vec = temp_col_vec';
                    temp_diag_element = Cell_storage{column_index+1}(1,temp_length);
                    
                    Reconstructed_matrix(1:column_index, 1:column_index) = ...
                        Reconstructed_matrix(1:column_index, 1:column_index) + ...
                        (1/temp_diag_element)*(temp_col_vec*temp_col_vec');
                end
                
                %%%% quick check
                %%%% sum(eig(Reconstructed_matrix)>0) %%% should be equal to q
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                LOG_data_density = sum(log(normpdf(xx_reduced,0, inv(sqrt(post_mean_omega_direct)))));
                
                LOG_posterior_density = log(gampdf(post_mean_omega_direct, n/2+1, 2/(lambda+S)));
                
                log_data_likelihood(num_lambda,num_BGL) = LOG_data_density;
                log_posterior_density(num_lambda, num_BGL) = LOG_posterior_density;
               
                
                log_ratio_of_liklelihoods(1,q) = LOG_data_density - LOG_posterior_density;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%% Sum of all terms III is the expression below
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if q==2
                    
                    direct_eval_log_prior_density(1,num_lambda) = -log(0.67) + 0.5*(q*(q-1))*log(lambda/2)...
                    -(lambda)*sum(abs(Reconstructed_matrix(tril(true(size(Reconstructed_matrix)),-1)))) ...
                    + q*log(lambda/2) - (lambda/2)*sum(diag(Reconstructed_matrix));
                    
                else
                    
                    direct_eval_log_prior_density(1,num_lambda) = 0.5*(q*(q-1))*log(lambda/2)...
                    -(lambda)*sum(abs(Reconstructed_matrix(tril(true(size(Reconstructed_matrix)),-1)))) ...
                    + q*log(lambda/2) - (lambda/2)*sum(diag(Reconstructed_matrix));
                
                end
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                Matrix_of_log_ratio_likelihoods(num_lambda,:) = log_ratio_of_liklelihoods;
            end            
        end
    catch
        fprintf("Move on to next lambda : %d\n", num_lambda+1);
        bad_lambda(1, num_lambda) = 1;
    end
    %toc;
    
end


temp_harmonic_vec = Harmonic_mean_est_vec(Harmonic_mean_est_vec~=0);

temp_our_est = sum(Matrix_of_log_ratio_likelihoods,2) ;
temp_our_est = temp_our_est(temp_our_est~=0);

logi_bad_lambda=(bad_lambda~=1);
All_mle_at_lambda = temp_our_est + direct_eval_log_prior_density(direct_eval_log_prior_density~=0)';
lambda_seq_reduced = lambda_sequence(1, logi_bad_lambda);

plot(lambda_seq_reduced, All_mle_at_lambda)
xlabel("lambda")
ylabel("log-marginal")
title("log-marginal vs. lambda, BGL, n=20 (train), p=12 (ATK and mTOR pathway proteins)")

lambda_max_idx = find(All_mle_at_lambda == max(All_mle_at_lambda));
fprintf("%d is the length of lambda_max_idx\n", length(lambda_max_idx));
lambda_max = lambda_seq_reduced(1,lambda_max_idx);
lambda_max %#ok<NOPTS>
fprintf("%f is the MMLE at lambda = %f\n", max(All_mle_at_lambda), lambda_max);

OSP_Real_data_BGL_graphical_evidence(lambda_max)
