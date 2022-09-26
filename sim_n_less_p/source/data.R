
cov <- list()
for(i in 1:P){
  cov[[i]] <- rnorm(n,0,1)
}
Xpred <- do.call(cbind, cov)

reg1 <- ifelse(Xpred[,1]< 0, 1, -1 )
reg2 <- ifelse(Xpred[,2]< 0, -1, 1 )

Y_trt <- rbinom(n, 1, pnorm(0.5+reg1+reg2-0.5*abs(Xpred[,3]-1)+1.5*Xpred[,4]*Xpred[,5])) # exposure model
Y_1 <- rnorm(n, 0.5*reg1+0.5*reg2-1+0.5*abs(Xpred[,3]+1) + 0.3*Xpred[,4]+ exp(0.5*Xpred[,5])-0.5*1*abs(Xpred[,6]) - 1*1*abs(Xpred[,7]+1) + 1*Xpred[,8]+ 1*Xpred[,9] + 1*Xpred[,10] + 0.5*Xpred[,11]+ 0.5*Xpred[,12] + 0.5*Xpred[,13] - 0.5*Xpred[,14] - 0.5*Xpred[,15] - 0.5*Xpred[,16] - exp(0.2*Xpred[,17]), 0.3)

Y_0 <- rnorm(n, 0.5*reg1+0.5*reg2-0+0.5*abs(Xpred[,3]+1) + 0.3*Xpred[,4]+ exp(0.5*Xpred[,5])-0.5*0*abs(Xpred[,6]) - 1*0*abs(Xpred[,7]+1) + 1*Xpred[,8]+ 1*Xpred[,9] + 1*Xpred[,10] + 0.5*Xpred[,11]+ 0.5*Xpred[,12] + 0.5*Xpred[,13] - 0.5*Xpred[,14] - 0.5*Xpred[,15] - 0.5*Xpred[,16] - exp(0.2*Xpred[,17]), 0.3)

Y_out <- Y_trt*Y_1 + (1-Y_trt)*Y_0

