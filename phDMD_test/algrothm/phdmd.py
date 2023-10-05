# -*- coding: utf-8 -*-
import logging

import numpy as np
import numpy.linalg as npl

from algrothm.skew_sym import sym,skew,skew_project,skew_symmetric

def phdmd(X, Y, U, delta_t, E, delta=1e-10):
    # 根据数据生成模型对应的快照
    W = (X[:, 1:] - X[:, :-1]) / delta_t
    V = 1 / 2 * (X[:, 1:] + X[:, :-1])
    U = 1 / 2 * (U[:, 1:] + U[:, :-1])
    Y = 1 / 2 * (Y[:, 1:] + Y[:, :-1])

    T = np.concatenate((V, U))
    Z = np.concatenate((E @ W, -Y))
    J, R = weight_phdmd(T, Z)
    J, R, e = phdmd_FGM(T, Z, J, R,  delta)
    return J, R


def weight_phdmd(Tao, Z, tol=1e-12):
    # 文中给的方法是通过求解加权后的优化问题给出一个初始值估计
    r = np.linalg.matrix_rank(Tao)
    V1,Omega,WT = np.linalg.svd(Tao)
    W1 = WT.T
    # 取skin svd,其中r是Tao的秩，也就是计算其截断SVD
    Omega1 = Omega[:r]
    Omega1_inv = np.diag(1/Omega1)  # 计算文献里SVD中对角矩阵及其逆矩阵的截断
    Omega1 = np.diag(Omega1)
    Z_T = V1.T @ Z @ W1
    Z1 = Omega1 @ Z_T[:r,:r]
    JJ = skew(Z1)
    JP = skew_project(-Z1)
    J11 = V1[:,:r] @ Omega1_inv @ JJ @ Omega1_inv @ V1[:,:r].T
    R11 = V1[:,:r] @ Omega1_inv @ JP @ Omega1_inv @ V1[:,:r].T
    ERR = np.linalg.norm(Tao.T@(Z -(J11-R11)@Tao),"fro")
    return J11 , R11


def phdmd_FGM(Tao , Z ,J , R,err_end=1e-10):
    TT = Tao @ Tao.T
    eigvalues,eigvector = np.linalg.eigh(TT)
    L = max(eigvalues)
    q = min(eigvalues)/L
    max_iter = 5000
    alpha0,e0 = 0.1,0
    Q = R
    J0, R0 = J, R
    for k in range(max_iter):
        Z1 = Z + R0 @ Tao
        JK = skew_symmetric(Tao,Z1)
        Z2 = JK @ Tao - Z
        DELTA = Q @ TT - Z2 @ Tao.T
        RK = skew_project(Q-DELTA/L)
        alpha1 = 0.5*(q- alpha0**2 + np.sqrt((q-alpha0**2)**2+4*alpha0**2))
        beta = alpha0*(1-alpha0)/(alpha0**2+alpha1)
        Q = RK + beta * (RK - R0)
        err = npl.norm(RK-R0,'fro')/npl.norm(RK,"fro")+npl.norm(J0-JK,"fro")/npl.norm(JK,'fro')
        e1 = npl.norm(Z-(JK-RK)@Tao)
        if k%1000 == 0: 
            print("e",e1,"err",err)
        if err<err_end or abs(e1- e0)<err_end:   
            break
        alpha0, e0, J0, R0 = alpha1, e1, JK, RK
    err = npl.norm(Z-(J0-R0)@Tao)
    return J0,R0,err


if __name__ =="__main__":
    pass