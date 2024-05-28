
lambda_vec = 5:5:200; %% for q = 50
%lambda_vec = 40:10:300; %% for q = 75
q = 40; 
sample_size = [80,100,125,150,175,200,225]; %% for q=40
%sample_size = [100,125,150,175,200,225,250]; %% for q=50
%sample_size = [125,150,175,200,225,250]; %% for q=75

burnin = 1e3;
nmc = 5e3;

parfor lambda_idx = 1:length(lambda_vec)
    
    lambda = lambda_vec(1,lambda_idx);
    fprintf("%d lambda is being used\n", lambda);       
    [True_Omega, ~,~] = prior_sampling(q,burnin,nmc, lambda, 0); 
    %%% 0 which is passed to the prior_sampling function, controls seed
    csvwrite(['./True_Omega_mat/Omega_q_',num2str(q),'_lambda_',num2str(lambda),'.csv'], True_Omega);

    for n = sample_size
        xx= mvnrnd(zeros(1,q),inv(True_Omega),n);
        S = xx'*xx;
        csvwrite(['./X_mat/X_mat_q_',num2str(q),'_n_',num2str(n),'_lambda_',num2str(lambda),'.csv'], xx);
        csvwrite(['./S_mat/S_mat_q_',num2str(q),'_n_',num2str(n),'_lambda_',num2str(lambda),'.csv'], S);
    end

end

total_num_rand_orders = 25;
Matrix_of_random_orders = zeros(total_num_rand_orders, q);
for num_rand_orders = 1:total_num_rand_orders
    Matrix_of_random_orders(num_rand_orders,:) = randperm(q);
end


csvwrite(['./Random_orders/Order_mat_q_',num2str(q),'.csv'], Matrix_of_random_orders)

