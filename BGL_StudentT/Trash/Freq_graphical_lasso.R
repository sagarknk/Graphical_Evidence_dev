######
## Estimate precision matrix by frequentist GLS
######

if (!require(glasso)) install.packages('glasso')
if (!require(cvTools)) install.packages('cvTools')

library(glasso)
library(cvTools)


#################################################################################
#xx_all =read.csv("Merged_11_protein_data.csv",header = FALSE, sep = ",")

xx_all = read.csv("LUSC_data.csv", header =FALSE, sep=",")
xx_all  = as.matrix(xx_all)
###########################################
#Cross validation for Graphical LASSO
log_likelihood <- function(precision, emp_cov) {
  p      <- nrow(precision)
  logdet <- determinant(precision, logarithm=T)$modulus
  loglik <- 0.5 * (logdet - sum(emp_cov * precision) - p*log(2*pi))
  return(as.numeric(loglik))
}

glasso_cv <- function(ts, k=5, rholist, verbose=T, penalize.diag) {
  library(glasso)
  library(cvTools)
  n     <- nrow(ts)
  folds <- cvFolds(n, k, type="consecutive")
  
  loglikes <- sapply(1:k, function(ki) {
#     if (verbose) cat("Fold ", ki, "\n")
    S_train <- cov(ts[folds$which!=ki,])
    S_test  <- cov(ts[folds$which==ki,])
    if (penalize.diag==T) {
      GLP     <- glassopath(S_train, rholist=rho_seq, trace=0, penalize.diagonal=TRUE, maxit=20) }
    else if (penalize.diag==F) {
      GLP     <- glassopath(S_train, rholist=rho_seq, trace=0, penalize.diagonal=FALSE, maxit=20) }
    loglike <- apply(GLP$wi, 3, function(P_train) log_likelihood(P_train, S_test))
    loglike
  })
  
  ind     <- which.max(rowMeans(loglikes))
  rhomax  <- rholist[ind]
  S       <- cov(ts)
  a       <- glasso(S, rhomax)
  a$rhomax <- rhomax
  
  return(a)
}

m=1 ## number of datasets
rho_seq = exp(seq(from=-15,to=0,by=0.01))
#rho_seq = seq(1e-5,1e-3,1e-6)
rhomax_GLT = rep(NA,m); 
GLT_time = rep(NA,m); 

#xx = xx_all[1:150,]
#xx_test = xx_all[151:300,]

xx = xx_all[1:125,]
xx_test = xx_all[126:250,]


t1 = Sys.time()
gl_cv_T = glasso_cv(ts=xx, k=5, verbose=T, rholist=rho_seq, penalize.diag=T)
GLT_time = Sys.time()-t1

GL_est_T = gl_cv_T$wi
GL_cov_T = gl_cv_T$w
rhomax_GLT = gl_cv_T$rhomax

print(rhomax_GLT)

############ Out of sample prediction #######################

Coeff_matrix = GL_est_T
diag(Coeff_matrix) =0

for(diag_idx in 1:11){
  Coeff_matrix[, diag_idx] = Coeff_matrix[,diag_idx]/GL_est_T[diag_idx, diag_idx]
  Coeff_matrix[, diag_idx] = -1*Coeff_matrix[, diag_idx]
}

xx_test = as.matrix(xx_test)
Est_matrix = xx_test %*% Coeff_matrix
Est_norm = sqrt(sum((Est_matrix - xx_test)^2))

print(Est_norm)

n = nrow(xx)
q = ncol(xx)
S_train = t(xx) %*% xx
S_omega = S_train %*% GL_est_T
S_omega = as.matrix(S_omega)
likelihood_at_lambda_max = -0.5*n*q*log(2*pi) + 0.5*n*log(det(GL_est_T)) -0.5*sum(diag((S_omega)))
# library(plot.matrix)
# par(mar=c(5.1, 4.1, 4.1, 4.1))
# plot(t(xx_test))
# plot(t(Est_matrix))
