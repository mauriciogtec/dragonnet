library(reticulate)
np = import("numpy")

for (year in c(2013, 2014)) {
  load(sprintf("dat/data%s.RData", year))
  np$savez(
    file=sprintf("dat/data%s.npz", year),
    X=np$array(X, dtype=np$float32),
    A=np$array(A, dtype=np$int),
    Y=np$array(Y, dtype=np$float32),
    varnames=np$array(dimnames(X)[[2]])
  )
}
