## SIM n < P

rm(list=ls())

num_trial = 200
n=100 # num. of observations
P=150 # num. of potential confounders

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
  if(ci[1] < -2.5656 & ci[2] > -2.5656){cover[test_case] = 1}
}

# Bias
mean(Effect- (-2.5656))
# MSE
mean((Effect- (-2.5656))^2)
# Coverage
mean(cover)
