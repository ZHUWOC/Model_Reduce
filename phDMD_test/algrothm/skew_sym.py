# -*- coding: utf-8 -*-
import logging

import numpy as np
import numpy.linalg as npl

def sym(A):
    return 1 / 2 * (A + A.T)

def skew(A):
    return 1 / 2 * (A - A.T)

def skew_project(A):
    # 首先对其进行特征分解，只保留其中正的特征值
    eigvalues, eigvector = np.linalg.eigh(sym(A))
    eigvalues[eigvalues <= 0] = 0
    PA = eigvector @ np.diag(eigvalues) @ eigvector.T
    return PA

def skew_symmetric(Tao,ZZ1):
    # 求解对应的skew_symmetric问题,目标函数是
    n , m = Tao.shape
    r = npl.matrix_rank(Tao)
    V,S,W = npl.svd(Tao)
    W = W.T
    S1 = S[:r]
    SS_squared = S1 ** 2
    PHI = 1 / (SS_squared[:, np.newaxis] + SS_squared)
    S1_inv = np.diag(1/S1)
    S1 = np.diag(S1)
    zz = V.T @ ZZ1 @ W
    Z1 = zz[:r,:r]
    Z3 = zz[r:,:r]
    J = np.zeros((n,n))
    J[:r,:r] = PHI * (2 * skew(Z1 @ S1))  
    J[r:,:r] = Z3 @ S1_inv
    J[:r,r:] = -(Z3 @ S1_inv).T
    JJ = V @ J @ V.T
    err = npl.norm(JJ @ Tao- ZZ1,'fro')
    # print(err)
    return JJ