%%%% Setting seed and dimensions of the precision matrix ('p' in the paper
%%%% which is 'q' here

rng(123456789)
q = 5; %% This is 'p' in the paper
n = 10;
lambda = 1;
t_dof = 1e4;

%%% In the codes, we use the hirearchy \omega_{ij} ~ N(0,\tau_{ij}^2/\lambda^2)
%%% followed by the Inverse-gamma scale mixture of half-Cauchy densities
%%% (proposed by Makalic and Schmidt, 2016) for the half-Cauchy prior on \tau_{ij}.
%%% Hence the value of \lambda mentioned in the paper, is equivalent to 1/lambda in the code
%%% Example, if in Table 3, lambda = 6, then set lambda = 1/6 above

%%%% For setting the True_Omega, we run the prior sample 6e3 times with a
%%%% burnin of 1e3 samples and then select the last sample i.e., nmcth
%%%% sample as the True_Omega

burnin = 1e3;
nmc = 5e3;

[True_Omega, ~,~] = prior_sampling(q,burnin,nmc, lambda, 0);
csvwrite(['./True_Omega_mat/True_Omega_mat_q_',num2str(q),'_n_',num2str(n),'_lambda_',num2str(1/lambda),'.csv'], True_Omega);

%%% 0 which is passed to the prior_sampling function, controls seed
%%% The function prior_sampling need not be translated to C/C++

%%% Generating a nxq data matrix from N(0, \Sigma = inv(True_Omega))
xx= mvnrnd(zeros(1,q),inv(True_Omega),n);
S = xx'*xx;

%%% Storing the generated matrices
csvwrite(['./X_mat/X_mat_q_',num2str(q),'_n_',num2str(n),'_lambda_',num2str(1/lambda),'.csv'], xx);
csvwrite(['./S_mat/S_mat_q_',num2str(q),'_n_',num2str(n),'_lambda_',num2str(1/lambda),'.csv'], S);

%%%% if q=2 then we can compute log_marginal upto an additive constant
%%%% C_{GHS} in a closed form which is Proposition 3 in paper

if q==2
    
    S_orig = S;
    prior_u_ = gamrnd(0.5*t_dof*ones([1,1e4]), 2/t_dof);

    outer_vals_for_expectation = zeros([1,1e4]);

    for outer_gam_idx = 1:1e4
        
        S = prior_u_(1,outer_gam_idx).*S_orig;

        K = 1/lambda;
        s_11 = S(1,1);s_12 = S(1,2);s_22 = S(2,2);
        omega_22_rand = gamrnd(n/2+1, 2/(K + s_22),[1,1e6]);

        func_for_expectation = @(m, omega_22) exp(0.5.*m.*s_12.*s_12).*(m.^(-0.5))...
            .*((m + (omega_22 - m.*(K + s_11))/(K^2*omega_22)).^(-1));

        func_vals = zeros(1,length(omega_22_rand));

        for mc_iter = 1:length(omega_22_rand)

            func_vals(1,mc_iter) = integral(@(m) func_for_expectation(m,omega_22_rand(1,mc_iter)),...
                0, omega_22_rand(1,mc_iter)/(K + s_11));


        end

        log_const = log(K/pi) + 2*log(gamma(n/2+1)) - (n/2+1)*log(K + s_11) - ...
            (n/2+1)*log(K + s_22) - n*log(pi);
        
        log_mc_expecation = log(mean(func_vals));
        
        outer_vals_for_expectation(1,outer_gam_idx) = ...
            prior_u_(1,outer_gam_idx)^n .* exp(log_const + log_mc_expecation);

    end

   
    temp_m = exprnd(2,[1,1e5]);
    temp_tau = tan(rand([1,1e5])*pi/2);

    C_val = mean(sqrt(temp_m./(temp_m + temp_tau.^2)));

    %%% C_val = 0.64; %%% Indepdent of lambda

    log_marginal_true =  log(mean(outer_vals_for_expectation)) - log(C_val);

    S = S_orig;
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
    for num_GHS = 1:q %%% consider num_GHS equivalent to index j in the paper

        if num_GHS<= q-1

            fprintf("Working with %d th telescopic sum\n", num_GHS);

            %%% for every iteration we need smaller and smaller blocks of
            %%% the data matrix xx. When num_GHS =1, we need the entire
            %%% matrix xx. When num_GHS = 2, we need the (p-1) x (p-1)
            %%% block of the matrix xx. And so on ... 

            reduced_data_xx = xx(:,1:(q-num_GHS+1));

            [~,q_reduced] = size(reduced_data_xx);
            column_number = q - num_GHS + 1;
            %%% Also to be noted that column_number = q_reduced

            S_reduced = reduced_data_xx'*reduced_data_xx;

            Matrix_2be_added_Gibbs = ...
                Matrix_to_be_added(1:q_reduced, 1:q_reduced);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Run the unrestricted sampler to get samples, which will
            %%% be used to approximate the Normal density in the
            %%% evaluation of the term IV_{p-j+1} - Equation (9)
            
            %%% The function below (GHS_Hao_wang) needs to be translated to C/C++
            %%% The first 2 return values are matrices and the last return
            %%% value is a scalar
            
            [post_mean_omega,post_mean_tau_save, MC_average_Equation_9] = ...
                GHS_Hao_wang(S_reduced,n,burnin,nmc,lambda, t_dof,Matrix_2be_added_Gibbs);

            fixed_last_col = post_mean_omega(1:(q_reduced-1),q_reduced);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Run the restricted sampler to get samples, which will
            %%% be used to approximate the truncated Gamma density in the
            %%% evaluation of the term IV_{p-j+1}

            %%% The function below (GHS_last_col_fixed) needs to be translated to C/C++
            %%% Both the return values are scalars

            [post_mean_omega_22_2ndGibbs, MC_average_Equation_11] = ...
                GHS_last_col_fixed(S_reduced,n,burnin, nmc,lambda,t_dof ,fixed_last_col, ...
                Matrix_2be_added_Gibbs, post_mean_omega, post_mean_tau_save);

            %%%%%%%%%%% computing IV_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%% Posterior likelihood at posterior mean %%%%%%%%%%%%%%%%%%%%%%%%

            log_normal_posterior_density(1, num_GHS) =  MC_average_Equation_9;

            log_gamma_posterior_density(1, num_GHS) =  MC_average_Equation_11;

            log_posterior_density(1, num_GHS)= MC_average_Equation_9 + MC_average_Equation_11;

            %%%%%%%%%% Computing I_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%% Data likelihood (conditonal of multivatiate normal) %%%%%%%%%%%%
            %%%% computing mean, sd and then the data likelihood

            if 0.5*(n + t_dof) < 172
                st_dev = sqrt(1/post_mean_omega_22_2ndGibbs);
                mean_vec = -1*reduced_data_xx(:,setdiff(1:q_reduced, column_number))*...
                    post_mean_omega(column_number,setdiff(1:q_reduced,column_number))'...
                    *st_dev*st_dev;


                temp_SS = sum(sum((reduced_data_xx(:,column_number) - mean_vec).*(reduced_data_xx(:,column_number) - mean_vec)));
                temp_SS = (post_mean_omega_22_2ndGibbs/t_dof)*temp_SS;

                log_data_likelihood(1, num_GHS) = log(gamma(0.5*(n+t_dof))) - log(gamma(0.5*t_dof)) -0.5*n*log(t_dof) ...
                    +0.5*n*log(post_mean_omega_22_2ndGibbs) - 0.5*n*log(pi) ...
                    -0.5*(t_dof + n)*log(1+temp_SS);

            else

                prior_u_telescoping = gamrnd(0.5*t_dof*ones([1,1e4]), 2/t_dof);
                temp_norm_pdf = zeros([1,1e4]);

                for telescoping_gam_idx = 1:1e4
                    st_dev = sqrt(1/(prior_u_telescoping(1,telescoping_gam_idx) .* post_mean_omega_22_2ndGibbs));
                    mean_vec = -1*reduced_data_xx(:,setdiff(1:q_reduced, column_number))*...
                        post_mean_omega(column_number,setdiff(1:q_reduced,column_number))'...
                        *st_dev*st_dev;

                    temp_norm_pdf(1,telescoping_gam_idx) = prod(normpdf(reduced_data_xx(:,column_number), mean_vec, st_dev));
                end


                log_data_likelihood(1, num_GHS) = log(mean(temp_norm_pdf));

            end

            %%%%%%%%%% Computing III_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% As conditional prior desnities are not available, we
            %%% will evaluate the sum of all terms III at the end
            %%% (after the else case [when dealing with just y_1 or \omega_{11}],
            %%% which is coming up next)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            log_ratio_of_liklelihoods(1,num_GHS) = ...
                log_data_likelihood(1, num_GHS) ...
                - log_posterior_density(1, num_GHS);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Matrix_to_be_added(1:(q-num_GHS), 1:(q-num_GHS)) = ...
                Matrix_to_be_added(1:(q-num_GHS), 1:(q-num_GHS)) + ...
                (1/post_mean_omega_22_2ndGibbs)*(fixed_last_col*fixed_last_col');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Cell_storage{num_GHS} = [fixed_last_col', post_mean_omega_22_2ndGibbs];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else
            %%% To evaluate log f(y_1) or the last row in the telescopic sum. Here
            %%% we compute I_1 and IV_1, II_1 is because no further columns of the
            %%% data are left and III_1 is taken care of the joint prior evaluation

            xx_reduced = xx(:,1);
            S_reduced = xx_reduced'*xx_reduced;
            q_reduced = 1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            omega_save_direct = zeros(q_reduced,q_reduced,nmc);
            posterior_u_save = zeros(1,nmc);
            temp_omega_sample =  gamrnd(n/2 + 1, 2/(1/lambda +  S_reduced));

            for i = 1:nmc

                shape_posterior_u_ = (n + t_dof)/2;
                scale_posterior_u_ = 2/(t_dof + trace(S_reduced * temp_omega_sample));
                posterior_u_ = gamrnd(shape_posterior_u_, scale_posterior_u_);
                posterior_u_save(1,i) = posterior_u_;

                temp_omega_sample = gamrnd(n/2 + 1, 2/(1/lambda + posterior_u_ .* S_reduced));
                omega_save_direct(:,:,i) = temp_omega_sample;
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
            if 0.5*(n+t_dof) < 172

                temp_SS = sum(sum(xx_reduced.*xx_reduced));
                temp_SS = (post_mean_omega_direct/t_dof)*temp_SS;

                LOG_data_density = log(gamma(0.5*(n+t_dof))) - log(gamma(0.5*t_dof)) -0.5*n*log(t_dof) ...
                    +0.5*n*log(post_mean_omega_direct) - 0.5*n*log(pi) ...
                    -0.5*(t_dof + n)*log(1+temp_SS);

            else
                prior_u_telescoping = gamrnd(0.5*t_dof*ones([1,1e4]), 2/t_dof);
                temp_norm_pdf = zeros([1,1e4]);

                for telescoping_gam_idx = 1:1e4
                    st_dev = sqrt(1/(prior_u_telescoping(1,telescoping_gam_idx) .* post_mean_omega_direct));
                    mean_vec = 0;

                    temp_norm_pdf(1,telescoping_gam_idx) = prod(normpdf(xx_reduced, mean_vec, st_dev));
                end


                LOG_data_density = log(mean(temp_norm_pdf));

            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            temp_log_posterior_density = zeros(1,nmc);

            for i=1:nmc
                temp_log_posterior_density(1,i) = gampdf(post_mean_omega_direct, ...
                    n/2+1, 2/(1/lambda+ posterior_u_save(1,i).*S_reduced));
            end

            LOG_posterior_density = log(mean(temp_log_posterior_density));
            
            
            %LOG_data_density = sum(log(normpdf(xx_reduced,0, inv(sqrt(post_mean_omega_direct)))));
            %LOG_posterior_density = log(gampdf(post_mean_omega_direct, n/2+1, 2/(1/lambda+S_reduced)));

            log_data_likelihood(1,q) = LOG_data_density; %%% Equal to I_1
            log_posterior_density(1,q) = LOG_posterior_density; %%%% Equal to IV_1
            log_ratio_of_liklelihoods(1,q) = LOG_data_density - LOG_posterior_density;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% Sum of all terms III is computed below
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            temp_abs_lower_tri_est_mat = abs(Reconstructed_matrix(tril(true(size(Reconstructed_matrix)),-1)));

            log_dawson_vals = zeros(1,0.5*q*(q-1));

            for lower_tri_iter = 1:(0.5*q*(q-1))

                %%% Horseshoe density can also be expressed as a
                %%% Laplace mixture: https://ieeexplore.ieee.org/document/9979805
                %%% So, we this representation of horseshoe density for
                %%% faster and stable computation, rather than drawing samples
                %%% from half  Cauchy distribution

                temp_rate = temp_abs_lower_tri_est_mat(lower_tri_iter,1)/lambda;
                temp_mean = 1/temp_rate;

                temp_rand_sample = exprnd(temp_mean, [1,1e4]); %%% u
                temp_rand_sample = temp_rand_sample/sqrt(2); %%% u/sqrt(2)

                %%% As calculating values of Dawson intergal function
                %%% is an time extensive operation, we use the following
                %%% approxmiation which has a MSE of 10^-8 when compared with
                %%% the values given by dawson() function of matlab

                %%% Approximation used can be accesed from:
                %%% https://www.sciencedirect.com/science/article/abs/pii/S009630039600330X

                temp_rand_sample_pow_2 = temp_rand_sample.^2;
                temp_rand_sample_pow_4 = temp_rand_sample.^4;
                temp_rand_sample_pow_6 = temp_rand_sample.^6;
                temp_rand_sample_pow_8 = temp_rand_sample.^8;

                temp_dawson_vals_num = 1+ (33/232)*temp_rand_sample_pow_2+...
                    (19/632)*temp_rand_sample_pow_4+...
                    (23/1471)*temp_rand_sample_pow_6;

                temp_dawson_vals_denom = 1+(517/646)*temp_rand_sample_pow_2+...
                    (58/173)*temp_rand_sample_pow_4+...
                    (11/262)*temp_rand_sample_pow_6+...
                    (46/1471)*temp_rand_sample_pow_8;

                temp_dawson_vals = temp_rand_sample.*(temp_dawson_vals_num./...
                    temp_dawson_vals_denom);

                log_dawson_vals(1,lower_tri_iter) = log(mean(temp_dawson_vals));
            end

            if q==2
                direct_eval_log_prior_density(1,1) = -log(C_val) + 0.5*(q*(q-1))*(log(2)-1.5*log(pi))...
                    + sum(log_dawson_vals) -sum(log(temp_abs_lower_tri_est_mat))...
                    + q*log(1/(2*lambda)) - (1/(2*lambda))*sum(diag(Reconstructed_matrix));

            else
                direct_eval_log_prior_density(1,1) = 0.5*(q*(q-1))*(log(2)-1.5*log(pi))...
                    + sum(log_dawson_vals) -sum(log(temp_abs_lower_tri_est_mat))...
                    + q*log(1/(2*lambda)) - (1/(2*lambda))*sum(diag(Reconstructed_matrix));

            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end

catch
    fprintf("Log marginal cannot be computed for this data matrix \n");
end

our_time= toc(tStart);



if q ==2
    log_marginal_true %#ok<NOPTS>
    %%% prints the closed form of log-marginal (Proposition 2 in the paper)
    Final_estimate_log_marginal = sum(log_ratio_of_liklelihoods,2) +  direct_eval_log_prior_density %#ok<NOPTS>
    %%% Above is the log marginal log f(y_{1:p}) - Equation (4) 

    our_time %#ok<NOPTS>
else
    Final_estimate_log_marginal = sum(log_ratio_of_liklelihoods,2) +  direct_eval_log_prior_density %#ok<NOPTS>
    %%% Above is the log marginal log f(y_{1:p}) - Equation (4) 

    our_time %#ok<NOPTS>
end
