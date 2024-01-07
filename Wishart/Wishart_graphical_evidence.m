%%%% Setting seed and dimensions of the precision matrix ('p' in the paper
%%%% which is 'q' here

rng(123456789)
q = 5; %% This is 'p' in the paper
n = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% degree of freedom for wishart
alpha = 7;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
scale_matrix = eye(q);
for scale_col = 1:(q-1)
    scale_matrix(scale_col, scale_col+1) = 0.25;
    scale_matrix(scale_col+1, scale_col) = 0.25;
end

%sum(eig(scale_matrix)>0)
scale_matrix = (1/(alpha))*scale_matrix;
csvwrite(['./Scale_mat/Scale_mat_q_',num2str(q),'_n_',num2str(n),'_alpha_',num2str(alpha),'.csv'], scale_matrix);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

True_Omega = wishrnd(scale_matrix, alpha);
csvwrite(['./True_Omega_mat/True_Omega_mat_q_',num2str(q),'_n_',num2str(n),'_alpha_',num2str(alpha),'.csv'], True_Omega);

%%%% Original data from a multivariate normal
xx_orig = mvnrnd(zeros(1,q),inv(True_Omega),n);
%%%%% Scaled data
xx = xx_orig*sqrtm(scale_matrix);
S = xx'*xx;

csvwrite(['./X_mat/X_mat_q_',num2str(q),'_n_',num2str(n),'_alpha_',num2str(alpha),'.csv'], xx);
csvwrite(['./S_mat/S_mat_q_',num2str(q),'_n_',num2str(n),'_alpha_',num2str(alpha),'.csv'], S);

%%%% log of marginal density
%%%% Nothing but  \int \pi(\Y|\Omega)\pi(\Omega) d\Omega
%%%% with scaled data and Wishart(I, alpha) prior

log_marginal = -(n*q/2)*log(pi) + logmvgamma((alpha+n)/2, q) - ...
    ((alpha+n)/2)*log(det(eye(q) + S)) - logmvgamma(alpha/2,q);

%%%% orginal_marginal
%%%% with unscaled data and Wishart(V, alpha) prior

log_orig_marginal =  -(n*q/2)*log(pi) + logmvgamma((alpha+n)/2, q) - ...
    ((alpha+n)/2)*log(det(inv(scale_matrix) + xx_orig'*xx_orig))...
    -(alpha/2)*log(det(scale_matrix))- logmvgamma(alpha/2,q);

%%% which is same as

log_re_transformed_marginal = log_marginal+ (n/2)*log(det(scale_matrix));

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

Cell_storage = cell(q,1);
%%% storage of posterior means i.e., \tilde{\theta}_{p-j+1}^* for every
%%% j from 1 to p.


%%% Noting the start time
tStart = tic;

for num_Wishart = 1:q

    if num_Wishart<=(q-1)

        fprintf("Working with %d th row of the telescoping sum\n", num_Wishart);

        %%% for every iteration we need smaller and smaller blocks of
        %%% the data matrix xx. When num_Wishart =1, we need the entire
        %%% matrix xx. When num_Wishart = 2, we need the (p-1) x (p-1)
        %%% block of the matrix xx. And so on ...

        reduced_data_xx = xx(:,1:(q-num_Wishart+1));

        [~,q_reduced] = size(reduced_data_xx);
        column_number = q - num_Wishart + 1;
        %%% Also to be noted that column_number = q_reduced

        S_reduced = reduced_data_xx'*reduced_data_xx;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Run the unrestricted sampler to get samples, which will
        %%% be used to approximate the Normal density in the
        %%% evaluation of the term IV_{p-j+1} - Equation (9)

        %%% The function below (Wishart_Hao_wang) needs to be translated to C/C++
        %%% The first return value is a matrix and the second return
        %%% value is a scalar

        [post_mean_omega, MC_average_Equation_9] = ...
            Wishart_Hao_wang(S_reduced,n,burnin,nmc,alpha + (1-num_Wishart));

        fixed_last_col = post_mean_omega(1:(q_reduced-1),q_reduced);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Run the restricted sampler to get samples, which will
        %%% be used to approximate the truncated Gamma density in the
        %%% evaluation of the term IV_{p-j+1}

        %%% The function below (Wishart_last_col_fixed) needs to be translated to C/C++
        %%% Both the return values are scalars

        [post_mean_omega_22_2ndGibbs, MC_average_Equation_11]  = ...
            Wishart_last_col_fixed(S_reduced,n,burnin,nmc,...
            alpha + (1-num_Wishart),fixed_last_col);


        %%%%%%%%%%% computing IV_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%% Posterior likelihood at posterior mean %%%%%%%%%%%%%%%%%%%%%%%%

        log_normal_posterior_density(1, num_Wishart) =  MC_average_Equation_9;

        log_gamma_posterior_density(1, num_Wishart) =  MC_average_Equation_11;

        log_posterior_density(1, num_Wishart)= MC_average_Equation_9 + MC_average_Equation_11;


        %%%%%%%%%% Computing I_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%% Data likelihood (conditonal of multivatiate normal) %%%%%%%%%%%%
        %%%% computing mean, sd and then the data likelihood

        st_dev = sqrt(1/post_mean_omega_22_2ndGibbs);
        mean_vec = -1*reduced_data_xx(:,setdiff(1:q_reduced, column_number))*...
            post_mean_omega(column_number,setdiff(1:q_reduced,column_number))'...
            *st_dev*st_dev;

        log_data_likelihood(1, num_Wishart) = ...
            sum(log(normpdf(reduced_data_xx(:,column_number), mean_vec, st_dev)));


        %%%%%%%%%% Computing III_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% As conditional p    rior desnities are not available, we
        %%% will evaluate the sum of all terms III at the end
        %%% (after the else case [when dealing with just y_1 or \omega_{11}],
        %%% which is coming up next)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        log_ratio_of_liklelihoods(1,num_Wishart) = ...
            log_data_likelihood(1, num_Wishart) - log_posterior_density(1, num_Wishart);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Cell_storage{num_Wishart} = [fixed_last_col', post_mean_omega_22_2ndGibbs];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        %%% evaluate log f(y_1) or the last row in the telescoping sum.

        xx_reduced = xx(:,1);
        S_reduced = xx_reduced'*xx_reduced;
        q_reduced = 1;
        omega_save_direct = zeros(q_reduced,q_reduced,nmc);
        for i = 1:nmc
            %omega_save_direct(:,:,i) = wishrnd(inv(eye(q_reduced) + S), alpha+(1-q)+n);
            %%% The above is same as below
            omega_save_direct(:,:,i) = gamrnd(0.5*(alpha+(1-q)+n),2/(1 + S_reduced));
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
        %%%% That is reversing the Schur's complements %%%%%%%%%%%

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

        %%%% Quick check
        %%%% sum(eig(Reconstructed_matrix)>0) %%% should be equal to q

        LOG_data_density = sum(log(normpdf(xx_reduced,0, inv(sqrt(post_mean_omega_direct)))));
        %%% Equal to I_1

        LOG_posterior_density = ((alpha + (1-q)+n-q_reduced-1)/2)*log(det(post_mean_omega_direct)) ...
            -0.5*trace((eye(q_reduced)+S_reduced)*post_mean_omega_direct)...
            +1*((alpha + (1-q)+n)/2)*log(det(eye(q_reduced)+S_reduced)) ...
            -((alpha + (1-q)+n)*q_reduced/2)*log(2)...
            -logmvgamma((alpha + (1-q)+n)/2,q_reduced);
        %%% Equal to IV_1

        log_data_likelihood(1,q) = LOG_data_density;
        log_posterior_density(1, q) = LOG_posterior_density;
        log_ratio_of_liklelihoods(1,q) = LOG_data_density - LOG_posterior_density;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Sum of all terms III is the expression below
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        direct_eval_log_prior_density= 0.5*(alpha-q-1)*log(det(Reconstructed_matrix))...
            -0.5*trace(Reconstructed_matrix) - 0.5*alpha*q*log(2) -logmvgamma(alpha/2,q);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end


end
our_time=  toc(tStart);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




log_orig_marginal %#ok<NOPTS>
%%%prints the true log_marginal

sum(log_ratio_of_liklelihoods,2)+ direct_eval_log_prior_density + (n/2)*log(det(scale_matrix)) %#ok<NOPTS>
%%% Above is the log marginal log f(y_{1:p}) - Equation (4)

our_time %#ok<NOPTS>
