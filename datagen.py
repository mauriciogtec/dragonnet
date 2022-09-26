import numpy as np
import random
import torch
import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.preprocessing import StandardScaler


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except:
        pass


def sim_data(scenario, n = 300, P = 100):
    X = [np.random.randn(n) for _ in range(P)]
    reg1 = (X[0] < 0) * 1.0 + (X[0] >= 0) * (-1.0) 
    reg2 = (X[1] < 0) * (-1.0) + (X[1] >= 0) * 1.0
    mu_trt = 0.5 + reg1 + reg2 - 0.5*np.abs(X[2]-1) + 1.5*X[3] * X[4]
    if scenario == 0: # IHDP
        j = np.random.randint(1, 51)
        file_path = f"dat/ihdp/csv/ihdp_npci_{j}.csv"
        data = np.loadtxt(file_path, delimiter=',')
        A, Y_out, Y_cf = data[:, 0].astype(int), data[:, 1], data[:, 2]
        Y_0 = Y_cf * A + Y_out * (1 - A)
        Y_1 = Y_cf * (1 - A) + Y_out * A
        mu_0, mu_1, Xpred = data[:, 3], data[:, 4], data[:, 5:]
    elif scenario == 1:    
        mu_0 = reg1 + 1.5*reg2 + 2*np.abs(X[2]+1) + 2*X[3] + np.exp(0.5*X[4])
        mu_1 = reg1 + 1.5*reg2 -1 +  2*np.abs(X[2]+1) + 2*X[3] + np.exp(0.5*X[4]) - 0.5*np.abs(X[5]) - np.abs(X[6]+1)
    elif scenario == 2:
        mu_0 = reg1 + 1.5*reg2 + 2*np.abs(X[2]+1) + 2*X[3] + np.exp(0.5*X[4])
        mu_1 = reg1 + 1.5*reg2 -1 +  2*np.abs(X[2]+1) + 2*X[3] + np.exp(0.5*X[4]) - 0.5*np.abs(X[4])
    elif scenario == 3:
        mu_trt += 1.5*X[5] - X[6]
        mu_0 = reg1 + 1.5*reg2 + 2*np.abs(X[2]+1) + 2*X[3] + np.exp(0.5*X[4])
        mu_1 = reg1 + 1.5*reg2 -1 +  2*np.abs(X[2]+1) + 2*X[3] + np.exp(0.5*X[4]) - 0.5*np.abs(X[4])
    elif scenario in (4, 5):
        mu_0 = (
            reg1 + 1.5*reg2 + 2*np.abs(X[2]+1) + 2*X[3] + np.exp(0.5*X[4])
            + X[7] + X[8] + X[9] + 0.5*X[10] + 0.5*X[11] + 0.5*X[12] - 0.5*X[13]
            - 0.5*X[14] - 0.5*X[15] - np.exp(-0.2*X[16])
        )
        mu_1 = (
            reg1 + 1.5*reg2 -1 + 2*np.abs(X[2]+1) + 2*X[3] + np.exp(0.5*X[4])
            -0.5*np.abs(X[5]) - np.abs(X[6]+1)
            + X[7] + X[8] + X[9] + 0.5*X[10] + 0.5*X[11] + 0.5*X[12] - 0.5*X[13]
            - 0.5*X[14] - 0.5*X[15] - np.exp(-0.2*X[16])
        )
    elif scenario == 6:
        mu_0 = (
            0.5*reg1 + 0.5*reg2 + 0.5*np.abs(X[2]+1) + 0.3*X[3] + np.exp(0.5*X[4])
            + X[7] + X[8] + X[9] + 0.5*X[10] + 0.5*X[11] + 0.5*X[12] - 0.5*X[13]
            - 0.5*X[14] - 0.5*X[15] - np.exp(-0.2*X[16])
        )
        mu_1 = (
            0.5*reg1 + 0.5*reg2 + -1 + 0.5*np.abs(X[2]+1) + 0.3*X[3] + np.exp(0.5*X[4])
            -0.5*np.exp(X[5]) - np.abs(X[6]+1)
            + X[7] + X[8] + X[9] + 0.5*X[10] + 0.5*X[11] + 0.5*X[12] - 0.5*X[13]
            - 0.5*X[14] - 0.5*X[15] - np.exp(-0.2*X[16])
        )
    else:
        raise NotImplementedError(scenario)
  
    if scenario != 0:
        Y_0 = np.random.normal(mu_0, 0.3, n) 
        Y_1 = np.random.normal(mu_1, 0.3, n)
        Xpred = np.stack(X, -1)
        A = np.random.binomial(1, norm.cdf(mu_trt), n)
        Y_out = A * Y_1 + (1 - A) * Y_0

    scaler = StandardScaler().fit(Y_out[:,None])
    Y_0, Y_1, mu_0, mu_1, Y_out = [scaler.transform(u[:,None])[...,0] for u in (Y_0, Y_1, mu_0, mu_1, Y_out)]
    
    return Xpred, A, Y_0, Y_1, Y_out, scaler
