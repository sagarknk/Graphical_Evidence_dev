# ### n=20, p = 10
# lambda = c(0.01,0.5,1,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.5,5,5.5,6,8,10)
# marginal_likelihood = c(-525.08,-338.65,-320.9,-314.48,-313.01,-312.05,-311.57,-311.38,-311.43,-311.60,-311.93,
#                         -312.39,-312.91,
#                         -313.49,-314.82,-316.27,-317.77,-319.41,-325.88,-332.3)
# 
# sd_marginal = c(0.69,0.29,0.24,0.13,0.1,0.1,0.09,0.11,0.09,0.09,0.07,0.06,0.06,0.08,0.08,0.08,0.05,0.08,0.08,0.06)
# library(ggplot2)
# 
# log_marginal_data = as.data.frame(cbind(lambda, marginal_likelihood, sd_marginal))
# 
# temp_label_1 = "lambda[0]"
# temp_label_2 = "lambda[max]"
# ggplot(log_marginal_data, aes(x=lambda, y = marginal_likelihood))+geom_line(lwd = 0.3)+geom_point(size=0.75)+
#   xlab(expression(lambda))+ylab("log-marginal")+theme_classic()+
#   geom_vline(xintercept = 2, size = 0.2, color ="green")+
#   geom_vline(xintercept = 2.5, size = 0.2, color ="red")+
#   annotate("text", x = 1.75, y =-450, label = temp_label_1,parse = TRUE)+
#   annotate("text", x = 2.75, y =-450, label = temp_label_2, parse = TRUE)+
#   ggtitle(expression("log-marginal vs." ~ lambda ~", under BGL when p=10, n = 20."))+ 
#   theme(plot.title = element_text(hjust = 0.5))
# 
# 
# lambda_max = lambda[which(marginal_likelihood == max(marginal_likelihood))]
# lambda_max_idx = which(marginal_likelihood == max(marginal_likelihood))
# 
# marginal_likelihood_diff = marginal_likelihood[lambda_max_idx] - marginal_likelihood
# bf_favourable = marginal_likelihood_diff>1.10
# lambda_not_favourable = lambda[!bf_favourable]
# lambda_not_favourable
# 
# ########### n = 100, p =10
# 
# ### n=20, p = 10
# lambda = c(seq(1,3,0.1),3.5)
# marginal_likelihood = c(-1.3947,-1.392,-1.3906,-1.3892,-1.3888,-1.3871,-1.3863,-1.3859,
#                         -1.3855,-1.3851,-1.385,-1.3848,-1.3848,-1.3849,-1.3851,-1.3853,-1.3855,-1.3859,-1.3863,
#                         -1.3866,-1.3873,-1.3898)
# 
# #sd_marginal = c(0.69,0.29,0.24,0.13,0.1,0.1,0.09,0.11,0.09,0.09,0.07,0.06,0.06,0.08,0.08,0.08,0.05,0.08,0.08,0.06)
# #library(ggplot2)
# 
# log_marginal_data = as.data.frame(cbind(lambda, marginal_likelihood))
# 
# temp_label_1 = "lambda[0]"
# temp_label_2 = "lambda[max]"
# ggplot(log_marginal_data, aes(x=lambda, y = marginal_likelihood))+geom_line(lwd = 0.3)+geom_point(size=0.75)+
#   xlab(expression(lambda))+ylab(expression("log-marginal x"~10^3 ))+theme_classic()+
#   geom_vline(xintercept = 2, size = 0.2, color ="green")+
#   geom_vline(xintercept = 2.1, size = 0.2, color ="red")+
#   annotate("text", x = 1.9, y =-1.39, label = temp_label_1,parse = TRUE)+
#   annotate("text", x = 2.2, y =-1.39, label = temp_label_2, parse = TRUE)+
#   ggtitle(expression("log-marginal x"~10^3 ~"vs."~ lambda ~", under BGL when p=10, n = 100."))+ 
#   theme(plot.title = element_text(hjust = 0.5))
# 
# 
# lambda_max = lambda[which(marginal_likelihood == max(marginal_likelihood))]
# lambda_max_idx = which(marginal_likelihood == max(marginal_likelihood))
# if(length(lambda_max_idx)>1){
#   lambda_max_idx = lambda_max_idx[1]
# }
# 
# marginal_likelihood_diff = marginal_likelihood[lambda_max_idx] - marginal_likelihood
# bf_favourable = marginal_likelihood_diff*1e3>1.10
# lambda_not_favourable = lambda[!bf_favourable]
# lambda_not_favourable

########### n = 150, p =10
library(ggplot2)
lambda = c(0,0.5,1,1.5,1.6,1.7,1.8,1.9,1.905,1.91,1.915,1.92,1.925,1.95,1.975,2,
           2.1,2.2,2.3,2.4,2.5,3,4,5)
marginal_likelihood = c(-2.1625565,-2.052769,-2.031577,-2.024968,-2.024439,-2.024038,-2.023839,-2.023835,
                        -2.023818,-2.023716,-2.023904,-2.023730,-2.023879,-2.023966,-2.023995,
                        -2.023901,-2.024141,-2.024238,-2.024660,-2.024954,-2.025526,
                        -2.028696,-2.037615,-2.048059)

rm_mle = mean(range(marginal_likelihood))
#sd_marginal = c(0.69,0.29,0.24,0.13,0.1,0.1,0.09,0.11,0.09,0.09,0.07,0.06,0.06,0.08,0.08,0.08,0.05,0.08,0.08,0.06)
#library(ggplot2)

log_marginal_data = as.data.frame(cbind(lambda, marginal_likelihood))

temp_label_1 = "lambda[0]"
temp_label_2 = "lambda[max]"
y_axis_label = "~x10^3"
ggplot(log_marginal_data, aes(x=lambda, y = marginal_likelihood))+geom_line(lwd = 0.3)+geom_point(size=0.75)+
  xlab(expression(lambda))+ylab(expression("log-marginal"))+theme_classic()+
  geom_vline(xintercept = 2, size = 0.2, color ="darkgreen")+
  geom_vline(xintercept = 1.91, size = 0.2, color ="red")+
  annotate("text", x = 0.75, y =rm_mle, label = temp_label_2,parse = TRUE, size =8)+
  annotate("text", x = 1.51, y =rm_mle, label = "=1.91", size = 7.5)+
  annotate("text", x = 0.01, y =max(marginal_likelihood), label = y_axis_label, parse=TRUE, size = 6)+
  annotate("text", x = 2.25, y =rm_mle, label = temp_label_1, parse = TRUE, size =8)+#ggtitle(expression("log-marginal x"~10^3 ~"vs."~ lambda ~", under BGL when p=10, n = 150."))+ 
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(size=16),
        axis.text.y = element_text(size=16),
        axis.title.x = element_text(size=17),
        axis.title.y = element_text(size=17))


lambda_max = lambda[which(marginal_likelihood == max(marginal_likelihood))]
lambda_max_idx = which(marginal_likelihood == max(marginal_likelihood))
if(length(lambda_max_idx)>1){
  lambda_max_idx = lambda_max_idx[1]
}

marginal_likelihood_diff = marginal_likelihood[lambda_max_idx] - marginal_likelihood
bf_favourable = marginal_likelihood_diff*1e3>1.10
lambda_not_favourable = lambda[!bf_favourable]
lambda_not_favourable

BF_test_likeli = c(-2.1625565,-2.052769,-2.031577,-2.024968,-2.025526,-2.028696,-2.037615,-2.048059)
BF_test_likeli_diff = marginal_likelihood[lambda_max_idx] - BF_test_likeli
length(BF_test_likeli)
BF_test_likeli_diff*1e3
sum(BF_test_likeli_diff*1e3>1.10)
lambda_vec_H0_compar = c(0.05,0.5,1,1.5,2.5,3,4,5)
lambda_vec_H0_compar
