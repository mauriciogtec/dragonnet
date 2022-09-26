## SIM 2

rm(list=ls())

num_trial = 200
n=300 # num. of observations
P=100 # num. of potential confounders

# place holder for each effect estimate
Effect = vector(mode = "numeric", length = num_trial)
cover = rep(0, num_trial)

for(test_case in 1:200) {
    
    # draw each replicate
    # Y_out: outcome
    # Y_trt: treatment/exposure
    # Xpred: potential confounders
    source("source/data.R")
    
    fit = # dragonnet model
    Effect[test_case] = # average treatment effect from "fit"
    ci <- # 95\% C.I. of estimated Effect from "fit"
    if(ci[1] < -1.3989 & ci[2] > -1.3989){cover[test_case] = 1}
}

# Bias
mean(Effect- (-1.3989))
# MSE
mean((Effect- (-1.3989))^2)
# Coverage
mean(cover)


