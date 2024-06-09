library(ggplot2)
########### n = 150, p =10

lambda = c(0.05,0.5,1,1.5,1.6,1.7,1.8,1.9,2,2.05,2.1,2.125,2.1375,2.15,2.2,2.3,2.4,2.5,3,4,5)
marginal_likelihood = c(-2.113368,-2.023465,-2.005955,-2.000028,-1.999431,-1.998625,-1.998618,-1.998365,-1.998182,
                        -1.998132,-1.998153,-1.998108,-1.998062,-1.998127,-1.998127,
                        -1.998186,-1.998257,-1.998446,-1.999851,-2.001690,-2.01089)

rm_marginal_likeli = mean(range(marginal_likelihood))
#sd_marginal = c(0.69,0.29,0.24,0.13,0.1,0.1,0.09,0.11,0.09,0.09,0.07,0.06,0.06,0.08,0.08,0.08,0.05,0.08,0.08,0.06)
#library(ggplot2)

log_marginal_data = as.data.frame(cbind(lambda, marginal_likelihood))

temp_label_1 = "lambda[0]"
temp_label_2 = "lambda[max]"
y_axis_label = "~x10^3"
ggplot(log_marginal_data, aes(x=lambda, y = marginal_likelihood))+geom_line(lwd = 0.3)+geom_point(size=0.75)+
  xlab(expression(lambda))+ylab(expression("log-marginal" ))+theme_classic()+
  geom_vline(xintercept = 2, size = 0.2, color ="darkgreen")+
  geom_vline(xintercept = 2.1375, size = 0.2, color ="red")+
  annotate("text", x = 2.5, y = rm_marginal_likeli, label = temp_label_2,parse = TRUE,size =8)+
  annotate("text", x = 3.5, y = rm_marginal_likeli, label = "=2.1375", size = 8)+
  annotate("text", x = 0.01, y =max(marginal_likelihood)+0.01, label = y_axis_label, parse=TRUE, size = 6)+
  annotate("text", x = 1.8, y = rm_marginal_likeli, label = temp_label_1, parse = TRUE, size =8)+#ggtitle(expression("log-marginal x"~10^3 ~"vs."~ lambda ~", under BGL when p=10, n = 150."))+ 
  theme(plot.title = element_text(hjust = 0.5))+
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

BF_test_likeli = c(-2.113368,-2.023465,-2.005955,-2.000028,-1.998446,-1.999851,-2.001690,-2.01089)
BF_test_likeli_diff = marginal_likelihood[lambda_max_idx] - BF_test_likeli
length(BF_test_likeli)
BF_test_likeli_diff*1e3
sum(BF_test_likeli_diff*1e3>1.10)
lambda_vec_H0_compar = c(0.05,0.5,1,1.5,2.5,3,4,5)
lambda_vec_H0_compar
