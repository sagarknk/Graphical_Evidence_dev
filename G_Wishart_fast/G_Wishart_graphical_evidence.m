%%%% Setting seed and dimensions of the precision matrix ('p' in the paper
%%%% which is 'q' here

rng(123456789)
q = 15; %% This is 'p' in the paper
n = 2*q;
delta = 5; %%% in paper delta is alpha
is_banded = false;

if is_banded
    banded_param = 1;

    G_mat_adj = eye(q);

    for g_mat_row = 1:(q-banded_param)
        for g_mat_col = (g_mat_row + 1):(g_mat_row +banded_param)
            G_mat_adj(g_mat_row , g_mat_col) = 1;
            G_mat_adj(g_mat_col , g_mat_row) = 1;
        end
    end

    if banded_param >=2
        for g_mat_row = (q-banded_param+1):(q-1)
            for g_mat_col = (g_mat_row + 1):q
                G_mat_adj(g_mat_row , g_mat_col) = 1;
                G_mat_adj(g_mat_col , g_mat_row) = 1;
            end
        end
    end

    csvwrite(['./G_mat/G_mat_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'_banded_param_',num2str(banded_param),'.csv'], G_mat_adj);
    scale_matrix = (q)*eye(q);
    %scale_matrix = (2*delta + max(temp_sum_G_adj))*eye(q);
    csvwrite(['./Scale_matrix/Scale_mat_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'_banded_param_',num2str(banded_param),'.csv'], scale_matrix);

    %sum(eig(scale_matrix)>0)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [True_Omega,~] = prior_sampling(q, 100, 100, G_mat_adj, scale_matrix, delta,0);
    csvwrite(['./True_precision_mat/True_Omega_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'_banded_param_',num2str(banded_param),'.csv'], True_Omega);

    xx = mvnrnd(zeros(1,q),inv(True_Omega),n);
    csvwrite(['./X_mat/X_mat_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'_banded_param_',num2str(banded_param),'.csv'], xx);

    S = xx'*xx;
    csvwrite(['./S_mat/S_mat_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'_banded_param_',num2str(banded_param),'.csv'], S);

else
    G_mat_adj = eye(q);

    for g_mat_row = 1:(q-1)
        for g_mat_col = (g_mat_row + 1):q
            G_mat_adj(g_mat_row , g_mat_col) = 2*binornd(1,0.5);
        end
    end

    G_mat_adj = 0.5*(G_mat_adj + G_mat_adj');
    %temp_sum_G_adj = sum(G_mat_adj,2);
    %temp_sum_G_adj = temp_sum_G_adj - 1;

    csvwrite(['./G_mat/G_mat_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'.csv'], G_mat_adj);

    %%%%%%%%%%%%% Scale matrix V for G-Wishart

    scale_matrix = (q)*eye(q);
    %scale_matrix = (2*delta + max(temp_sum_G_adj))*eye(q);

    csvwrite(['./Scale_matrix/Scale_mat_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'.csv'], scale_matrix);

    %sum(eig(scale_matrix)>0)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [True_Omega,~] = prior_sampling(q, 100, 100, G_mat_adj, scale_matrix, delta,0);
    %%% 0 which is passed to the prior_sampling function, controls seed
    %%% The function prior_sampling need not be translated to C/C++

    csvwrite(['./True_precision_mat/True_Omega_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'.csv'], True_Omega);

    %%% Generating a nxq data matrix from N(0, \Sigma = inv(True_Omega)) and storing it
    xx = mvnrnd(zeros(1,q),inv(True_Omega),n);
    csvwrite(['./X_mat/X_mat_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'.csv'], xx);

    %%% Computing the sample covariance matrix and storing it
    S = xx'*xx;
    csvwrite(['./S_mat/S_mat_q_',num2str(q),'_n_',num2str(n),'_delta_',num2str(delta),'.csv'], S);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The true log-marginal can be computed in the case of banded structures

if is_banded

    num_cliques = q - banded_param;
    num_seperators = num_cliques -1;

    cliques_list = cell(1,num_cliques);
    seperators_list = cell(1, num_seperators);

    for cliques_index =1:num_cliques
        cliques_list{1,cliques_index} = cliques_index:(cliques_index+banded_param);
    end

    for seperators_index = 2:num_cliques
        seperators_list{1,seperators_index-1} = (seperators_index):(seperators_index+banded_param -1);
    end

    %%%%%%%%%%%%%%%%%%% Prior nomalizing constant %%%%%%%%%%%%%%%%%%%%%%%%%%%
    delta_modified_clique = (2*delta + banded_param +2)*0.5;
    delta_modified_seperator = (2*delta + banded_param +1)*0.5;

    log_num_prior_norm_const = 0;

    for cliques_index = 1:num_cliques

        scale_mat_clique = scale_matrix(cliques_list{1,cliques_index}, cliques_list{1,cliques_index});

        log_num_prior_norm_const = log_num_prior_norm_const + delta_modified_clique*(banded_param+1)*log(2) + ...
            logmvgamma(delta_modified_clique, banded_param+1) - delta_modified_clique*log(det(scale_mat_clique));

    end

    log_denom_prior_norm_const = 0;

    for seperators_index = 2:num_cliques

        scale_mat_seperator = scale_matrix(seperators_list{1,seperators_index-1}, seperators_list{1,seperators_index-1});

        log_denom_prior_norm_const = log_denom_prior_norm_const + delta_modified_seperator*(banded_param)*log(2) + ...
            logmvgamma(delta_modified_seperator, banded_param) - delta_modified_seperator*log(det(scale_mat_seperator));

    end

    log_prior_norm_constant = log_num_prior_norm_const - log_denom_prior_norm_const;

    %%%%%%%%%%%%%%%%%%% Posterior nomalizing constant %%%%%%%%%%%%%%%%%%%%%%%%%%%
    delta_modified_clique = (2*delta + banded_param +2 + n)*0.5;
    delta_modified_seperator = (2*delta + banded_param +1 + n)*0.5;

    log_num_posterior_norm_const = 0;

    for cliques_index = 1:num_cliques

        scale_mat_clique = S(cliques_list{1,cliques_index}, cliques_list{1,cliques_index}) + ...
            scale_matrix(cliques_list{1,cliques_index}, cliques_list{1,cliques_index});

        log_num_posterior_norm_const = log_num_posterior_norm_const + delta_modified_clique*(banded_param+1)*log(2) + ...
            logmvgamma(delta_modified_clique, banded_param+1) - delta_modified_clique*log(det(scale_mat_clique));

    end

    log_denom_posterior_norm_const = 0;

    for seperators_index = 2:num_cliques

        scale_mat_seperator = S(seperators_list{1,seperators_index-1}, seperators_list{1,seperators_index-1})+...
            scale_matrix(seperators_list{1,seperators_index-1}, seperators_list{1,seperators_index-1});

        log_denom_posterior_norm_const = log_denom_posterior_norm_const + delta_modified_seperator*(banded_param)*log(2) + ...
            logmvgamma(delta_modified_seperator, banded_param) - delta_modified_seperator*log(det(scale_mat_seperator));

    end

    log_posterior_norm_constant = log_num_posterior_norm_const - log_denom_posterior_norm_const;

    %%%%%%%%%%%%%%%%%%% True log marginal %%%%%%%%%%%%%%%%%%%%%%%%%%%

    log_marginal_true = -(n*q*0.5)*log(2*pi)+log_posterior_norm_constant - log_prior_norm_constant;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    % do nothing
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
burnin = 2e3; %%% burn-in for MCMC iterations
nmc = 1e4; %%% number of MC iterations after burn-in

%%% Initializing matrices to store

%%%% storage for I_{p-j+1} + III_{p-j+1} - IV_{p-j+1} for every j from 1 to p
log_ratio_of_liklelihoods = zeros(1,q);

log_data_likelihood = zeros(1, q);
log_posterior_density = zeros(1, q);
log_normal_posterior_density = zeros(1,q);
log_gamma_posterior_density = zeros(1,q);

log_prior_density = zeros(1, q); %%% Terms III_{p-j+1}
log_prior_density_scalar_gamma = zeros(1, q); %%% Gamma density in Equation (20)
log_prior_density_vector_normal = zeros(1, q); %%% Normal density in Equation (20)

our_time = zeros(1, 1);

%%% Noting the start time
tStart = tic;

Matrix_to_be_added = zeros(q,q);
%%%% The above matrix keeps track of the linear shifts
%%%% Needs to be updated every step


G_mat_adj_this_order = G_mat_adj;
scale_matrix_this_order = scale_matrix;

%%% The above two matrices adjust the adjacency matrix and the scale matrix
%%% as per the permutations to the columns of xx. As we ar enot considering
%%% any permutations, they will be identical to G_mat_adj and scale_matrix
%%% respectively

%%%%%%%%%%%% Starting point %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[start_point_first_gibbs, ~] = prior_sampling(q, 5, 5, G_mat_adj_this_order, scale_matrix_this_order, delta,0);
%%% 0 which is passed to the prior_sampling function, controls seed
%%% The function prior_sampling need not be translated to C/C++
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for num_G_Wishart = 1:q %%% consider num_G_Wishart equivalent to index j in the paper

    if num_G_Wishart <= q-1
        fprintf("%d th num_G_Wishart\n", num_G_Wishart);

        %%% for every iteration we need smaller and smaller blocks of
        %%% the data matrix xx. When num_G_Wishart =1, we need the entire
        %%% matrix xx. When num_G_Wishart = 2, we need the (p-1) x (p-1)
        %%% block of the matrix xx. And so on ...
        reduced_data_xx = xx(:,1:(q-num_G_Wishart+1));

        [~,q_reduced] = size(reduced_data_xx);
        S_reduced = reduced_data_xx'*reduced_data_xx;
        column_number = q - num_G_Wishart + 1;
        %%% Also to be noted that column_number = q_reduced

        Matrix_2be_added_Gibbs = ...
            Matrix_to_be_added(1:q_reduced, 1:q_reduced);

        G_mat_adj_this_order_reduced = G_mat_adj_this_order(1:q_reduced, 1:q_reduced);
        scale_matrix_this_order_reduced = scale_matrix_this_order(1:q_reduced, 1:q_reduced);

        %%% Run the unrestricted sampler to get samples, which will
        %%% be used to approximate the Normal density in the
        %%% evaluation of the term IV_{p-j+1} - Equation (21)

        %%% The function below (G_wishart_Hao_wang) needs to be translated to C/C++
        %%% The first return value is a matrix and the second return
        %%% value is a scalar

        [post_mean_omega, MC_average_Equation_9]  = ...
            G_wishart_Hao_wang(S_reduced,n,burnin,nmc,delta, ...
            scale_matrix_this_order_reduced,  G_mat_adj_this_order_reduced, Matrix_2be_added_Gibbs,...
            start_point_first_gibbs);


        fixed_last_col = post_mean_omega(1:(q_reduced-1),q_reduced);

        %%% Run the restricted sampler to get samples, which will
        %%% be used to approximate the truncated Gamma density in the
        %%% evaluation of the term IV_{p-j+1}

        %%% The function below (G_wishart_last_col_fixed) needs to be translated to C/C++
        %%% The first return value is a matrix and thenext two are scalars

        [start_point_first_gibbs, post_mean_omega_22_2ndGibbs, MC_average_Equation_11] = ...
            G_wishart_last_col_fixed(S_reduced,n,burnin, nmc,...
            delta,fixed_last_col,...
            scale_matrix_this_order_reduced,  G_mat_adj_this_order_reduced, Matrix_2be_added_Gibbs,...
            post_mean_omega);

        %%%%%%%%%%% computing IV_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%% Posterior likelihood at posterior mean %%%%%%%%%%%%%%%%%%%%%%%%

        log_normal_posterior_density(1, num_G_Wishart) =  MC_average_Equation_9;

        log_gamma_posterior_density(1, num_G_Wishart) =  MC_average_Equation_11;

        log_posterior_density(1, num_G_Wishart)= MC_average_Equation_9 + MC_average_Equation_11;

        %%%%%%%%%% Computing I_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%% Data likelihood (conditonal of multivatiate normal) %%%%%%%%%%%%
        %%%% computing mean, sd and then the data likelihood

        st_dev = sqrt(1/post_mean_omega_22_2ndGibbs);

        mean_vec = -1*reduced_data_xx(:,setdiff(1:q_reduced, column_number))*...
            post_mean_omega(column_number,setdiff(1:q_reduced,column_number))'...
            *st_dev*st_dev;

        log_data_likelihood(1, num_G_Wishart) = ...
            sum(log(normpdf(reduced_data_xx(:,column_number), mean_vec, st_dev)));

        %%%%%%%%%% Computing III_{p-j+1} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%% Prior density at posterior mean %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        V_mat_22 = scale_matrix_this_order_reduced(column_number,column_number);
        V_mat_12 = scale_matrix_this_order_reduced(1:q_reduced-1,column_number);
        vec_2be_added_21 =  -1 * Matrix_2be_added_Gibbs(1:q_reduced-1,column_number);
        s_21 = S(1:q_reduced-1,column_number);

        %%% Note the -1 above. This is done to make sure to respect the
        %%% crucial indicator function.
        %%%%%%%%%%%%%%%% finding which elements in beta to evaluate the %%%
        %%%%%%%%% likelihood %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        G_mat_current_col = G_mat_adj_this_order_reduced(1:q_reduced-1,column_number);
        logi_which_ones = (G_mat_current_col ==1);
        logi_which_zeros = (G_mat_current_col ==0);

        if sum(logi_which_ones)>=1
            V_mat_12_required = V_mat_12(logi_which_ones,1);
            s_21_required = s_21(logi_which_ones,1);

            if sum(logi_which_zeros)>=1
                vec_2be_added_21_required = vec_2be_added_21(logi_which_zeros,1);
            end

        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        inv_C = (post_mean_omega_22_2ndGibbs^(-1))*scale_matrix_this_order_reduced(1:q_reduced-1,1:q_reduced-1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if sum(logi_which_ones)>=1
            inv_C_required = inv_C(logi_which_ones, logi_which_ones);

            if sum(logi_which_zeros)>=1

                inv_C_not_required = inv_C(logi_which_zeros, logi_which_ones);

                vec_2be_added_21_required_modified = ...
                    vec_2be_added_21_required' * inv_C_not_required;

                vec_2be_added_21_required_modified = ...
                    vec_2be_added_21_required_modified' ;

                mu_i_reduced = -inv_C_required\...
                    (V_mat_12_required + vec_2be_added_21_required_modified);
            else
                mu_i_reduced = -inv_C_required\(V_mat_12_required + s_21_required);
            end

            mean_vec = mu_i_reduced;

            log_prior_density_vector_normal(1, num_G_Wishart)= ...
                log(mvnpdf(post_mean_omega(column_number,logi_which_ones),...
                mean_vec', inv(inv_C_required)));
            % Normal density in Equation (S.6)

        else
            log_prior_density_vector_normal(1, num_G_Wishart)=0;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Computing the GIG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (sum(logi_which_zeros)==0)
            log_prior_density_scalar_gamma(1, num_G_Wishart) = ...
                log(gampdf(post_mean_omega_22_2ndGibbs,...
                delta  + sum(logi_which_ones)/2 + 1 ,2/V_mat_22));
            % Gamma density in Equation (20)

        else
            GIG_a = V_mat_22;

            diag_V_mat = diag(scale_matrix_this_order_reduced);
            diag_V_mat_required = diag_V_mat(logi_which_zeros,1);
            vec_2be_added_21 =  -1 * Matrix_2be_added_Gibbs(1:q_reduced-1,column_number);
            vec_2be_added_21_required = vec_2be_added_21(logi_which_zeros,1);

            GIG_b = sum(diag_V_mat_required.*vec_2be_added_21_required.* vec_2be_added_21_required);

            if GIG_b == 0
                log_prior_density_scalar_gamma(1, num_G_Wishart) = ...
                    log(gampdf(post_mean_omega_22_2ndGibbs,...
                    delta  + sum(logi_which_ones)/2 + 1 ,2/V_mat_22));
                % Gamma density in Equation (20)
            else
                GIG_p =  delta  + sum(logi_which_ones)/2 + 1;
                log_prior_density_scalar_gamma(1, num_G_Wishart) = ...
                    (GIG_p/2)*log(GIG_a/GIG_b) - log(2)...
                    -log(besselk(GIG_p, sqrt(GIG_a*GIG_b))) ...
                    +(GIG_p - 1)*log(post_mean_omega_22_2ndGibbs) ...
                    -0.5*(GIG_a*post_mean_omega_22_2ndGibbs + (GIG_b/post_mean_omega_22_2ndGibbs));
                % GIG density in Equation (S.6)

            end
        end

        log_prior_density(1, num_G_Wishart) = ...
            log_prior_density_vector_normal(1, num_G_Wishart) + ...
            log_prior_density_scalar_gamma(1, num_G_Wishart);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        log_ratio_of_liklelihoods(1,num_G_Wishart) = ...
            log_data_likelihood(1, num_G_Wishart) + ...
            log_prior_density(1, num_G_Wishart) ...
            - log_posterior_density(1, num_G_Wishart);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Matrix_to_be_added(1:(q-num_G_Wishart), 1:(q-num_G_Wishart)) = ...
            Matrix_to_be_added(1:(q-num_G_Wishart), 1:(q-num_G_Wishart)) + ...
            (1/post_mean_omega_22_2ndGibbs)*(fixed_last_col*fixed_last_col');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else

        %%% To evaluate log f(y_1) or the marginal likelihood of the
        %%% remaining first column, it's like
        %%% imposing a univariate G-wishart prior on the precision of a
        %%% univariate normal random variable

        xx_reduced = xx(:,1);
        S_reduced = xx_reduced'*xx_reduced;
        q_reduced = 1;

        V_mat_22 = scale_matrix_this_order_reduced(q_reduced,q_reduced);
        LOG_marginal_first_col_only = -(n*q_reduced/2)*log(pi) + ...
            logmvgamma(delta+n/2+1, q_reduced) - ...
            (delta + n/2 + 1)*log(det(V_mat_22 + S_reduced)) - logmvgamma(delta +1,q_reduced) + ...
            (delta + 1)*log(V_mat_22);

        log_ratio_of_liklelihoods(1,q) = LOG_marginal_first_col_only;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end

end
our_time =   toc(tStart);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if is_banded
    log_marginal_true %#ok<UNRCH>
else
    % do nothing
end

sum(log_ratio_of_liklelihoods)
%%% Above is the log marginal log f(y_{1:p}) - Equation (4)

our_time %#ok<NOPTS>

