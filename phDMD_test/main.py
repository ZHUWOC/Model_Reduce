# -*- coding: utf-8 -*-
import logging
import os
import config
import numpy as np

from algrothm.phdmd import phdmd
from system.phlti import PHLTI


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if config.save_results:
        import matplotlib
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

        if not os.path.exists(config.figures_path):
            os.makedirs(config.figures_path)

    # Initialize the original ph system
    E, J, R, G, P, D, Q = config.ph_matrices()

    ph = PHLTI(E, J, R, G, P, D, Q)
    lti = ph.to_lti(matrices=False)

    # 生成训练数据
    U_train, X_train, Y_train = ph.sim(config.u, config.T, config.x0)

    # trajectories_plot(config.T, U_train, label='u', train=True)
    # trajectories_plot(config.T, Y_train, train=True)

    # 用phDMD方法对问题进行实现
    J_dmd, R_dmd = phdmd(X_train, Y_train, U_train, config.delta,
                         E=ph.E)
    ph_dmd = PHLTI(ph.E, J_dmd, R_dmd)
    
    # 生成测试数据并进行测试
    U_train_dmd, X_train_dmd, Y_train_dmd = ph_dmd.sim(config.u, config.T, config.x0)
    U_test, X_test, Y_test = ph.sim(config.u_test, config.T_test, config.x0)
    U_dmd, X_dmd, Y_dmd = ph_dmd.sim(config.u_test, config.T_test, config.x0)

    lti_dmd = ph_dmd.to_lti(matrices=False)
    lti_error = lti - lti_dmd


if __name__ == "__main__":
    main()
