num_samples = 1e5;
mu = 3;
lambda = 2;

rejection_samples = zeros(1, num_samples);
tic;
for sample_idx = 1:num_samples

    rand_nu = randn(1);
    rand_y = rand_nu.*rand_nu;
    rand_x = mu + (mu.*mu.*rand_y)./(2*lambda)...
        -(mu./(2*lambda)).*sqrt(4*lambda.*mu.*rand_y ...
        + (mu.*rand_y).*(mu.*rand_y));
    rand_z = rand(1);

    temp_logical = (rand_z <= (mu)./(mu + rand_x));

    %%% u_12 = rand_x.*(temp_logical) + (1-temp_logical).*(mu.*mu./rand_x);
    %%% the above is buggy because there will be cases when rand_x is
    %%% close to zero and temp_logical is 1. Then it results in 0 +
    %%% 0*Inf which is NaN. Hence changing to for loop.

    u_12 = rand_x;
    u_12_else = (mu.*mu./rand_x);

    if temp_logical== 0
        u_12 = u_12_else;
    end

    rejection_samples(1, sample_idx) = u_12;

end
toc;

tic;
gig_samples = gigrnd(-1/2,lambda/mu^2, lambda,num_samples);
toc;

[mean(rejection_samples), mean(gig_samples)]
[mean(1./rejection_samples), mean(1./gig_samples)]
[median(rejection_samples), median(gig_samples)]
%[mode(rejection_samples), mode(gig_samples)]
%mu*((1 + 9*mu*mu*0.25*lambda^(-2))^(0.5) - 1.5*mu/lambda)

[mean(rejection_samples.^2) - (mean(rejection_samples))^2, mean(gig_samples.^2) - (mean(gig_samples))^2]
[mean(rejection_samples.^(-2)) - (mean(1./rejection_samples))^2, mean(gig_samples.^(-2)) - (mean(1./gig_samples))^2]

tiledlayout(1,2)
nexttile
histogram(rejection_samples)
nexttile
histogram(gig_samples)

