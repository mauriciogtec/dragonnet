
cov <- list()
for(i in 1:P){
  cov[[i]] <- rnorm(n,0,1)
}
Xpred <- do.call(cbind, cov)
Xcut <- lapply(1:dim(Xpred)[2], function(t) sort(unique(Xpred[,t]))) # e.g. unique values of predictors

reg1 <- ifelse(Xpred[,1]< 0, 1, -1 )
reg2 <- ifelse(Xpred[,2]< 0, -1, 1 )

Y_trt <- rbinom(n, 1, pnorm(0.5+reg1+reg2-0.5*abs(Xpred[,3]-1)+1.5*Xpred[,4]*Xpred[,5]+1.5*Xpred[,6]-1*Xpred[,7])) # exposure model
Y_1 <- rnorm(n, 1*reg1+1.5*reg2-1+2*abs(Xpred[,3]+1) + 2*Xpred[,4]+ exp(0.5*Xpred[,5])-0.5*1*abs(Xpred[,5]), 0.3)

Y_0 <- rnorm(n, 1*reg1+1.5*reg2-0+2*abs(Xpred[,3]+1) + 2*Xpred[,4]+ exp(0.5*Xpred[,5])-0.5*0*abs(Xpred[,5]), 0.3)

Y_out <- Y_trt*Y_1 + (1-Y_trt)*Y_0

