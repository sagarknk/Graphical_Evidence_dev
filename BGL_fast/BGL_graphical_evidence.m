%%%% Setting seed and dimensions of the precision matrix ('p' in the paper
%%%% which is 'q' here

rng(123456789)
q = 10; %% This is 'p' in the paper
n = 20;
lambda = 2;

%%%% For setting the True_Omega, we run the prior sample 6e3 times with a
%%%% burnin of 1e3 samples and then select the last sample the True_Omega

burnin = 1e3;
nmc = 5e3;

[True_Omega, ~,~] = prior_sampling(q,burnin,nmc, lambda, 0);
csvwrite(['./True_Omega_mat/True_Omega_mat_q_',num2str(q),'_n_',num2str(n),'_lambda_',num2str(lambda),'.csv'], True_Omega);
%%% 0 which is passed to the prior_sampling function, controls seed
%%% The function prior_sampling need not be translated to C/C++

%%% Generating a nxq data matrix from N(0, \Sigma = inv(True_Omega))
xx= mvnrnd(zeros(1,q),inv(True_Omega),n);

%%% Computing the sample covariance matrix
S = xx'*xx;

%%% Storing the generated matrices
csvwrite(['./X_mat/X_mat_q_',num2str(q),'_n_',num2str(n),'_lambda_',num2str(lambda),'.csv'], xx);
csvwrite(['./S_mat/S_mat_q_',num2str(q),'_n_',num2str(n),'_lambda_',num2str(lambda),'.csv'], S);

%%%% if q=2 then we can compute log_marginal upto an additive constant
%%%% C_{BGL} in a closed form which is Proposition 2 in paper

if q==2
    constant_part_log_marginal = 3*log(lambda) + log(gamma(n/2 + 1))+...
        log(gamma((n+3)/2))-(n - 1/2)*log(pi)...
        -((n+3)/2)*log((lambda + S(1,1))*(lambda + S(2,2))-(lambda-abs(S(1,2)))^2);


    scale_MC_expectation = 2/((lambda + S(1,1))*(lambda + S(2,2))...
        -(lambda-abs(S(1,2)))^2);

    shape_MC_expectation = (n+3)/2;

    rand_gamma_MC_expectation = gamrnd(shape_MC_expectation, scale_MC_expectation,...
        1e4,1);

    inverse_gauss_CDF_P1 = normcdf(lambda.*sqrt(rand_gamma_MC_expectation).*...
        (abs(S(1,2))/lambda - 1));

    inverse_gauss_CDF_P2 = exp(2.*lambda.*abs(S(1,2)).*rand_gamma_MC_expectation).*...
        normcdf(-lambda.*sqrt(rand_gamma_MC_expectation).*...
        (abs(S(1,2))/lambda + 1));

    integral_part_log_marginal = log(mean(inverse_gauss_CDF_P1 + inverse_gauss_CDF_P2));

    log_marginal = -log(0.67) + constant_part_log_marginal + integral_part_log_marginal;
else
    %%% do nothing
end

%%% Initializing matrices to store

%%%% storage for I_{p-j+1} -IV_{p-j+1} for every j from 1 to p
log_ratio_of_liklelihoods = zeros(1,q);

log_data_likelihood = zeros(1, q);
log_posterior_density = zeros(1, q);

log_normal_posterior_density = zeros(1, q);
log_gamma_posterior_density = zeros(1, q);
direct_eval_log_prior_density = zeros(1,1);

burnin = 1e3; %%% burn-in for MCMC iterations
nmc = 5e3; %%% number of MC iterations after burn-in

%%% Noting the start time 
tStart = tic;

%%% We do a try-catch because, computing log-marginal -> log f(y_{1:p})
%%% kind of depends on the order of columns in xx (when 'p' is large). Hence having a try-catch
%%% becomes necessary 

try

    Cell_storage = cell(q,1);
    %%% storage of posterior means i.e., \tilde{\theta}_{p-j+1}^* for every
    %%% j from 1 to p.

    Matrix_to_be_added = zeros(q,q);
    %%%% The above matrix keeps track of the linear shifts 
    %%%% Needs to be updated every step

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for num_BGL = 1:q %%% consider num_BGL equivalent to index j in the paper

        if num_BGL<= q-1

            fprintf("Working with %d th row of the telescoping sum\n", num_BGL);

            %%% for every iteration we need smaller and smaller blocks of
            %%% the data matrix xx. When num_BGL =1, we need the entire
            %%% matrix xx. When num_BGL = 2, we need the (p-1) x (p-1)
            %%% block of the matrix xx. And so on ... 

            reduced_data_xx = xx(:,1:(q-num_BGL+1));

            [~,q_reduced] = size(reduced_data_xx);
            column_number = q - num_BGL + 1;
            %%% Also to be noted that column_number = q_reduced

            S_reduced = reduced_data_xx'*reduced_data_xx;

            Matrix_2be_added_Gibbs = ...
                Matrix_to_be_added(1:q_reduced, 1:q_reduced);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Run the unrestricted sampler to get samples, which will
            %%% be used to approximate the Normal density in the
            %%% evaluation of the term IV_{p-j+1} - Equation (9)
            
            %%% The function below (BGL_Hao_wang) needs to be translated to C/C++
            %%% The first 2 return values are matrices and the last return
            %%% value is a scalar

            [post_mean_omega,post_mean_tau_save, MC_average_Equation_9] = ...
                BGL_Hao_wang(S_reduced,n,burnin,nmc,lambda, Matrix_2be_added_Gibbs);
            
            %%% The function above (BGL_Hao_wang) calls the function
            %%% 'gigrnd' to sample from generalized inverse Gaussian. As
            %%% BGL_Hao_wang should be translated to C/C++, gigrnd should also be translated 
            
            fixed_last_col = post_mean_omega(1:(q_reduced-1),q_reduced);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Run the restricted sampler to get samples, which will
            %%% be used to approximate the truncated Gamma density in the
            %%% evaluation of the term IV_{p-j+1}

            %%% The function below (BGL_last_col_fixed) needs to be translated to C/C++
            %%% Both the return values are scalars

            [post_mean_omega_22_2ndGibbs, MC_average_Equation_11] = ...
                BGL_last_col_fixed(S_reduced,n,burnin, nmc,lambda, fixed_last_col, ...
                Matrix_2be_added_Gibbs, post_mean_omega, post_mean_tau_save);

            %%%%%%%%%%% computing IV_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%% Posterior likelihood at posterior mean %%%%%%%%%%%%%%%%%%%%%%%%

            log_normal_posterior_density(1, num_BGL) =  MC_average_Equation_9;

            log_gamma_posterior_density(1, num_BGL) =  MC_average_Equation_11;

            log_posterior_density(1, num_BGL)= MC_average_Equation_9 + MC_average_Equation_11;

            %%%%%%%%%% Computing I_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%% Data likelihood (conditonal of multivatiate normal) %%%%%%%%%%%%
            %%%% computing mean, sd and then the data likelihood

            st_dev = sqrt(1/post_mean_omega_22_2ndGibbs);
            mean_vec = -1*reduced_data_xx(:,setdiff(1:q_reduced, column_number))*...
                post_mean_omega(column_number,setdiff(1:q_reduced,column_number))'...
                *st_dev*st_dev;

            log_data_likelihood(1, num_BGL) = ...
                sum(log(normpdf(reduced_data_xx(:,column_number), mean_vec, st_dev)));

            %%%%%%%%%% Computing III_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% As conditional prior desnities are not available, we
            %%% will evaluate the sum of all terms III at the end
            %%% (after the else case [when dealing with just y_1 or \omega_{11}],
            %%% which is coming up next)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            log_ratio_of_liklelihoods(1,num_BGL) = ...
                log_data_likelihood(1, num_BGL) ...
                - log_posterior_density(1, num_BGL);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Matrix_to_be_added(1:(q-num_BGL), 1:(q-num_BGL)) = ...
                Matrix_to_be_added(1:(q-num_BGL), 1:(q-num_BGL)) + ...
                (1/post_mean_omega_22_2ndGibbs)*(fixed_last_col*fixed_last_col');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Cell_storage{num_BGL} = [fixed_last_col', post_mean_omega_22_2ndGibbs];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else
            %%% To evaluate log f(y_1) or the last row in the telescopic sum. Here
            %%% we compute I_1 and IV_1 only. II_1=0 because no further columns of the
            %%% data are left and III_1 is taken care of the joint prior evaluation

            xx_reduced = xx(:,1);
            S_reduced = xx_reduced'*xx_reduced;
            q_reduced = 1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            omega_save_direct = zeros(q_reduced,q_reduced,nmc);
            for i = 1:nmc
                omega_save_direct(:,:,i) = gamrnd(n/2 + 1, 2/(lambda + S_reduced));
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

            LOG_posterior_density = log(gampdf(post_mean_omega_direct, n/2+1, 2/(lambda+S_reduced)));

            log_data_likelihood(1,q) = LOG_data_density; %%% Equal to I_1
            log_posterior_density(1,q) = LOG_posterior_density; %%%% Equal to IV_1
            log_ratio_of_liklelihoods(1,q) = LOG_data_density - LOG_posterior_density;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% Sum of all terms III is the expression below
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if q==2

                direct_eval_log_prior_density(1,1) = -log(0.67) + 0.5*(q*(q-1))*log(lambda/2)...
                    -(lambda)*sum(abs(Reconstructed_matrix(tril(true(size(Reconstructed_matrix)),-1)))) ...
                    + q*log(lambda/2) - (lambda/2)*sum(diag(Reconstructed_matrix));

            else

                direct_eval_log_prior_density(1,1) = 0.5*(q*(q-1))*log(lambda/2)...
                    -(lambda)*sum(abs(Reconstructed_matrix(tril(true(size(Reconstructed_matrix)),-1)))) ...
                    + q*log(lambda/2) - (lambda/2)*sum(diag(Reconstructed_matrix));

            end


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end

catch
    fprintf("Log marginal cannot be computed for this data matrix \n");
end

our_time = toc(tStart);



if q ==2
    log_marginal %#ok<NOPTS>
    %%% prints the closed form of log-marginal (Proposition 2 in the paper)
else
    Final_estimate_log_marginal = sum(log_ratio_of_liklelihoods,2) +  direct_eval_log_prior_density %#ok<NOPTS>
    %%% Above is the log marginal log f(y_{1:p}) - Equation (4) 

    our_time %#ok<NOPTS>
end


