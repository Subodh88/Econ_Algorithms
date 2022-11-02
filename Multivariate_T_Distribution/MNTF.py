import numpy as np
import pandas as pd
from scipy.stats import t
from math import *
from numpy.linalg import inv, det, cholesky, multi_dot, cond
from scipy.linalg import cho_solve, cho_factor, solve, pinv
import sys
from statsmodels.sandbox.distributions.multivariate import mvstdtprob, mvstdnormcdf
import multiprocessing
from multiprocessing import Pool, freeze_support
from functools import partial
from scipy.optimize import fmin_bfgs
import statsmodels.tools.numdiff as ndd
import time
import sys

np.set_printoptions(precision=17)


def samplet(mu, Sigma, df, M, seedcurr):
    np.random.seed(seedcurr)
    d = len(Sigma)
    g = np.tile(np.random.gamma(df / 2., 2. / df, M), (d, 1)).T
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, M)
    return (mu + Z / np.sqrt(g)).T


def tpdf1(x, mu, Sigma, df):
    d = 1
    Num = gamma(1. * (d + df) / 2)
    Denom = (gamma(1. * df / 2) * pow(df * pi, 1. * d / 2) * pow(Sigma, 1. / 2) * pow(
        1 + (1. / df) * ((x - mu) * (1 / Sigma) * (x - mu)), (d + df) / 2))
    d = Num / Denom
    return d


def tpdf(x, mu, Sigma, df):
    x = x.flatten()
    mu = mu.flatten()
    d = Sigma.shape[0]
    Num = gamma(1. * (d + df) / 2)
    Denom = (gamma(1. * df / 2) * pow(df * pi, 1. * d / 2) * pow(np.linalg.det(Sigma), 1. / 2) * pow(
        1 + (1. / df) * np.dot(np.dot((x - mu), np.linalg.inv(Sigma)), (x - mu)), 1. * (d + df) / 2))
    d = 1. * Num / Denom
    return d


def tcdfinv(x, df):
    return (t.ppf(x, df, loc=0, scale=1))


def tcdf(x, df):
    return (t.cdf(x, df, loc=0, scale=1))


def HaltonSequence(n, dim):
    prim = np.array(
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
         109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
         233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
         283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
         431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541])
    prim = prim[:, np.newaxis]
    hs = np.zeros((n, dim))
    for idim in range(0, dim, 1):
        b = prim[idim, 0]
        hs[:, idim] = halton(n, b)

    return (hs[10:n, :])


def halton(n, s):
    k = floor(log(n + 1) / log(s))
    phi = np.zeros((1, 1))
    i = 1
    count = 0
    while i <= k:
        count = count + 1
        x = phi
        j = 1
        while j < s:
            y = phi + (j / s ** i)
            x = np.vstack((x, y))
            j = j + 1

        phi = x
        i = i + 1

    x = phi
    j = 1
    while (j < s) and (len(x) < (n + 1)):
        y = phi + (j / s ** i)
        x = np.vstack((x, y))
        j = j + 1

    out = x[1:(n + 1), 0]
    return (out)


def cdfmvtGHK(a, r, df, s):
    global _halt_maxdraws, _halt_numdraws, allHaltDraws, nrep

    nintegdim = a.shape[1]
    if sys.getsizeof(s) < 50:
        np.random.seed(s)
    else:
        np.random.set_state(s)

    rnum = np.random.random((1, 1))
    ss = np.random.get_state()
    s = ss

    startRow = ceil(rnum * (_halt_maxdraws - _halt_numdraws - 1))
    uniRands = allHaltDraws[startRow:startRow + _halt_numdraws, 0:nintegdim - 1]
    chol_r = cholesky(r)
    Zmat = np.zeros((nrep, nintegdim))
    Umat = np.zeros((nrep, (nintegdim - 1)))

    temp = tcdf(a[0, 0] / chol_r[0, 0], df) * np.ones((nrep, 1))
    Zmat[:, 0] = temp[:, 0]
    del temp

    ZW = np.multiply(Zmat[:, 0], uniRands[:, 0])
    temp = tcdfinv(ZW, df)
    Umat[:, 0] = temp
    del temp
    Ymat = np.zeros((nrep, nintegdim))
    Ymat[:, 0] = Umat[:, 0]

    for iintegdim in range(1, nintegdim, 1):
        i_index = iintegdim + 1
        bi = a[0, iintegdim] * np.ones((nrep, 1))
        ghkElem1 = 0
        ghkElem2 = 0
        for jintegdim in range(0, iintegdim, 1):
            temp1 = chol_r[iintegdim, jintegdim] * Ymat[:, jintegdim]
            temp1 = temp1[:, np.newaxis]
            ghkElem1 = ghkElem1 + temp1
            temp2 = Ymat[:, jintegdim] ** 2
            temp2 = temp2[:, np.newaxis]
            ghkElem2 = ghkElem2 + temp2
            del temp1, temp2

        temp1 = (bi - ghkElem1) * (1 / chol_r[iintegdim, iintegdim])
        temp2 = (df + ghkElem2) ** 0.5
        bi_bar = sqrt(df + i_index - 1) * np.divide(temp1, temp2)

        del temp1, temp2

        ei = tcdf(bi_bar.flatten(), df + i_index - 1)
        Zmat[:, iintegdim] = ei

        if (iintegdim < nintegdim - 1):
            ZW = np.multiply(Zmat[:, iintegdim], uniRands[:, iintegdim])
            temp = tcdfinv(ZW, df + i_index - 1)
            Umat[:, iintegdim] = temp
            temp1 = (1 / sqrt(df + i_index - 1)) * np.multiply(Umat[:, iintegdim][:, np.newaxis],
                                                               (df + ghkElem2) ** 0.5)
            Ymat[:, iintegdim] = temp1.flatten()
            del temp, temp1

    probab = Zmat[:, 0]
    probab = probab[:, np.newaxis]
    for iintegdim in range(1, nintegdim, 1):
        temp = Zmat[:, iintegdim]
        temp = temp[:, np.newaxis]
        probab = np.multiply(probab, temp)
        del temp
    probab = np.mean(probab, axis=0)[0]
    return (probab, s)


def bvt(a1, a2, r, df):
    dh = a1
    dk = a2
    nu = df

    tpi = 2 * pi
    ors = 1 - r * r
    hrk = dh - r * dk
    krh = dk - r * dh
    if (abs(hrk) + ors > 0):
        xnhk = hrk ** 2 / (hrk ** 2 + ors * (nu + dk ** 2))
        xnkh = krh ** 2 / (krh ** 2 + ors * (nu + dh ** 2))
    else:
        xnhk = 0
        xnkh = 0
    hs = np.sign(dh - r * dk)
    ks = np.sign(dk - r * dh)
    if (np.mod(nu, 2) == 0):
        bvt = np.arctan2(sqrt(ors), -r) / tpi
        gmph = dh / sqrt(16 * (nu + dh ** 2))
        gmpk = dk / sqrt(16 * (nu + dk ** 2))
        btnckh = 2 * np.arctan2(sqrt(xnkh), sqrt(1 - xnkh)) / pi
        btpdkh = 2 * sqrt(xnkh * (1 - xnkh)) / pi
        btnchk = 2 * np.arctan2(sqrt(xnhk), sqrt(1 - xnhk)) / pi
        btpdhk = 2 * sqrt(xnhk * (1 - xnhk)) / pi
        if (nu >= 2):
            for j_temp in range(0, int(nu / 2), 1):
                j = j_temp + 1
                bvt = bvt + gmph * (1 + ks * btnckh)
                bvt = bvt + gmpk * (1 + hs * btnchk)
                btnckh = btnckh + btpdkh
                btpdkh = 2 * j * btpdkh * (1 - xnkh) / (2 * j + 1)
                btnchk = btnchk + btpdhk
                btpdhk = 2 * j * btpdhk * (1 - xnhk) / (2 * j + 1)
                gmph = gmph * (j - 1 / 2) / (j * (1 + dh ** 2 / nu))
                gmpk = gmpk * (j - 1 / 2) / (j * (1 + dk ** 2 / nu))

    else:
        check = dh ** 2 + dk ** 2 - 2 * r * dh * dk + nu * ors
        if (check >= 0):
            qhrk = sqrt(dh ** 2 + dk ** 2 - 2 * r * dh * dk + nu * ors)
        else:
            qhrk = 0
        hkrn = dh * dk + r * nu
        hkn = dh * dk - nu
        hpk = dh + dk
        bvt = np.arctan2(-sqrt(nu) * (hkn * qhrk + hpk * hkrn), hkn * hkrn - nu * hpk * qhrk) / tpi
        if (bvt < -10 * np.finfo(float).eps):
            bvt = bvt + 1

        if (nu >= 3):
            gmph = dh / (tpi * sqrt(nu) * (1 + dh ** 2 / nu))
            gmpk = dk / (tpi * sqrt(nu) * (1 + dk ** 2 / nu))
            btnckh = sqrt(xnkh)
            btpdkh = btnckh
            btnchk = sqrt(xnhk)
            btpdhk = btnchk
            jrange = ((nu) - 1) / 2
            for j_temp in range(0, int(jrange), 1):
                j = j_temp + 1
                bvt = bvt + gmph * (1 + ks * btnckh)
                bvt = bvt + gmpk * (1 + hs * btnchk)
                btpdkh = (2 * j - 1) * btpdkh * (1 - xnkh) / (2 * j)
                btnckh = btnckh + btpdkh
                btpdhk = (2 * j - 1) * btpdhk * (1 - xnhk) / (2 * j)
                btnchk = btnchk + btpdhk
                gmph = gmph * j / ((j + 1 / 2) * (1 + dh ** 2 / nu))
                gmpk = gmpk * j / ((j + 1 / 2) * (1 + dk ** 2 / nu))

    p = bvt
    return (p)

def corrvc(S):
    temp = np.diag(S)
    temp = temp[:, np.newaxis]
    D = temp ** 0.5
    Dcol = np.kron(np.ones((1, S.shape[0])), D)
    Drow = np.kron(np.ones((S.shape[0], 1)), D.T)
    DF = np.divide(S, Dcol)
    DF = np.divide(DF, Drow)
    R = diagrv(DF)
    return (R)


def chol(r):
    a = cholesky(r)
    return (a)


def diagrv(a):
    for i in range(0, a.shape[0], 1):
        a[i, i] = 1.0
    return (a)


def pd_inv1(a):
    n = a.shape[0]
    I = np.identity(n)
    check1 = np.isfinite(cond(a))
    det1 = det(a)
    check2 = det1 > 0.01
    check = check1 & check2
    if (check):
        return cho_solve(cho_factor(a, lower=True), I)
    else:
        return (pinv(a))


def pd_inv(a):
    n = a.shape[0]
    I = np.identity(n)
    check1 = np.isfinite(cond(a))
    det1 = det(a)
    check2 = det1 > 0.01
    check3 = is_pos_def(a)
    check = check1 & check2 & check3
    if (check):
        return solve(a, I, sym_pos=True, overwrite_b=True)
    else:
        return (pinv(a))


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def xpnd(r):
    d = int((-1 + sqrt(1 + 8 * len(r))) / 2)
    xp = np.zeros((d, d), dtype=float)
    count = 0
    xp[0, 0] = r[0, 0]
    for i in range(1, d, 1):
        for j in range(0, i + 1, 1):
            count = count + 1
            xp[i, j] = r[count, 0]
            xp[j, i] = r[count, 0]
    return (xp)


def vech(r):
    drow = r.shape[0]
    d = int(drow * (drow + 1) * 0.5)
    xp = np.zeros((d, 1))
    xp[0, 0] = r[0, 0]
    count = 0
    for i in range(1, drow, 1):
        for j in range(0, i + 1, 1):
            count = count + 1
            xp[count, 0] = r[i, j]
    return (xp)


def upmat(r):
    drow = r.shape[0]
    xp = np.zeros((drow, drow))
    for i in range(0, drow, 1):
        for j in range(i, drow, 1):
            xp[i, j] = r[i, j]
    return (xp)


def lowmat(r):
    drow = r.shape[0]
    xp = np.zeros((drow, drow))
    xp[0, 0] = r[0, 0]
    for i in range(1, drow, 1):
        for j in range(0, i + 1, 1):
            xp[i, j] = r[i, j]
    return (xp)


def dloop(parm, seeddata):
    smallb = parm[0:nvarma]
    smallc = parm[nvarma:nvarma + nvarmc]
    temp = parm[nvarma + nvarmc:nvarma + nvarmc + nCholErr]
    Psi = xpnd(temp)
    del temp
    df = parm[nvarma + nvarmc + nCholErr]

    err_psi = samplet(np.zeros((nvar_cont + nc - 1)), Psi, df, nind, seeddata)
    err_psi = err_psi.T
    if (nvar_cont > 1):
        err_psi = np.hstack((err_psi[:, 0:nvar_cont], np.zeros((nind, 1)), err_psi[:, nvar_cont:err_psi.shape[1]]))
    else:
        err_psi = np.hstack((err_psi[:, 0][:, np.newaxis], np.zeros((nind, 1)), err_psi[:, nvar_cont:err_psi.shape[1]]))

    c1 = (np.kron(np.ones((nvar_cont, 1)), smallc)) * (Main_data.loc[:, ivgenvc].as_matrix().T)
    c = np.empty(shape=(nobs, nvar_cont), dtype=float)
    j = 0
    for i in range(nvar_cont):
        j = i + 1
        c[:, i] = np.sum(c1[(j - 1) * nvarmc:(j * nvarmc), :], axis=0)
    del c1

    if (nvar_cont > 1):
        Propensity_temp = c + err_psi[:, 0:nvar_cont]
    else:
        Propensity_temp = c + err_psi[:, 0][:, np.newaxis]

    Main_data[continuous_var] = Propensity_temp

    v1 = (np.kron(np.ones((nc, 1)), smallb)) * (Main_data.loc[:, ivgenva].as_matrix().T)
    v = np.empty(shape=(nobs, nc), dtype=float)
    j = 0
    for i in range(nc):
        j = i + 1
        v[:, i] = np.sum(v1[(j - 1) * nvarma:(j * nvarma), :], axis=0)
    del v1

    Utility_temp = v + err_psi[:, nvar_cont:err_psi.shape[1]]

    del err_psi, v, c
    ff = np.argmax(Utility_temp.T, axis=0)
    ff = ff[:, np.newaxis]
    ff = ff + np.ones((nobs, 1))
    temp = np.arange(1, nc + 1, 1)
    temp = temp[:, np.newaxis].T
    mask = np.kron(np.ones((nobs, 1)), temp)
    mask1 = (mask == ff).astype(int)

    Main_data[altchm] = ff
    Main_data[req_col] = mask1
    del Utility_temp, mask
    count = -1
    for i in range(0, nind, 1):
        Main_data.loc[i, 'PID'] = i + 1

    return (0)


def lpr(parm_estimated, idx_estimated, idx_fixed, parm_fixed):
    x = reconstruct(parm_estimated, idx_estimated, idx_fixed, parm_fixed)
    data_list = []
    for iter in range(0, Num_Threads, 1):
        data_list.append(iter)

    pool = Pool(processes=Num_Threads)
    prod_x = partial(lprT, parm=x)
    result_list = pool.map(prod_x, data_list)

    pool.close()
    pool.join()

    result_list = np.array(result_list)
    a_temp = []
    for iter in range(0, Num_Threads, 1):
        temp = result_list[iter]
        for j in range(0, temp.shape[0], 1):
            a_temp.append(temp[j])

    atemp_array = np.asarray(a_temp)
    if (Parametrized == 1):
        return (-np.mean(atemp_array, axis=0))

    if (Parametrized == 0):
        return (-atemp_array)


def lprT(iter, parm):
    st_iter = int(Data_Split[iter, 0])
    end_iter = int(Data_Split[iter, 1])
    num_obs = int(end_iter - st_iter + 1)
    obs_range = np.arange(st_iter - 1, end_iter, 1)
    obs_range = obs_range[:, np.newaxis]

    smallb = parm[0:nvarma]
    smallc = parm[nvarma:nvarma + nvarmc]
    temp = parm[nvarma + nvarmc:nvarma + nvarmc + nCholErr]
    if (Parametrized == 1):
        cholPsi = (upmat(xpnd(temp))).T
        row_num = nvar_cont
        if (nvar_cont > 1):
            value = cholPsi[row_num, 0:row_num]
            value = value[:, np.newaxis]
            value = multi_dot([value.T, value])
            denom = sqrt(1 + value)
            for i in range(0, row_num, 1):
                cholPsi[row_num, i] = cholPsi[row_num, i] / denom
            cholPsi[row_num, row_num] = 1 / denom
            cholPsi = cholPsi.T
        else:
            value = cholPsi[row_num, 0]
            value = value * value
            denom = sqrt(1 + value)
            cholPsi[row_num, 0] = cholPsi[row_num, 0] / denom
            cholPsi[row_num, 1] = 1 / denom
            cholPsi = cholPsi.T
        Psi = multi_dot([cholPsi.T, cholPsi])
    else:
        Psi = xpnd(temp)
    del temp
    Psi = multi_dot([D_matrix, Psi, D_matrix.T])
    df = parm[nvarma + nvarmc + nCholErr]
    if (Parametrized == 1):
        df = exp(df)

    iden_matrix = np.eye(nc - 1)
    one_negative = -1 * np.ones((nc - 1, 1))
    seednext = MACMLS[0, iter]

    v1 = (np.kron(np.ones((nc, 1)), smallb)) * (Main_data.loc[st_iter - 1:end_iter - 1, ivgenva].as_matrix().T)
    c1 = (np.kron(np.ones((nvar_cont, 1)), smallc)) * (Main_data.loc[st_iter - 1:end_iter - 1, ivgenvc].as_matrix().T)
    Utility = np.empty(shape=(num_obs, nc), dtype=float)
    Propensity = np.empty(shape=(num_obs, nvar_cont), dtype=float)
    for i in range(0, nc, 1):
        j = i + 1
        Utility[:, i] = np.sum(v1[(j - 1) * nvarma:(j * nvarma), :], axis=0)
    j = 0
    for i in range(0, nvar_cont, 1):
        j = i + 1
        Propensity[:, i] = np.sum(c1[(j - 1) * nvarmc:(j * nvarmc), :], axis=0)
    del v1, c1

    All_Chosen = Main_data.loc[st_iter - 1:end_iter - 1, altchm].as_matrix()
    All_X = Main_data.loc[st_iter - 1:end_iter - 1, continuous_var].as_matrix()
    obs_count = -1
    pair_count = -1
    Likelihood = np.zeros((num_obs, 1))

    for i in range(0, num_obs, 1):
        obs_count = obs_count + 1
        if (nvar_cont > 1):
            temp1 = Propensity[i, :]
            temp1 = temp1[:, np.newaxis]
            temp2 = Utility[i, :]
            temp2 = temp2[:, np.newaxis]
            UY = np.vstack((temp1, temp2))
            del temp1, temp2
        else:
            temp2 = Utility[i, :]
            temp2 = temp2[:, np.newaxis]
            UY = Propensity[i, 0]
            UY = np.vstack((UY, temp2))
            del temp2

        Full_error = Psi

        M = np.zeros((nvar_cont + (nc - 1), nvar_cont + nc))
        if (nvar_cont > 1):
            M[0:nvar_cont, 0:nvar_cont] = np.eye(nvar_cont)
        else:
            M[0, 0] = 1
        Alt_chosen = All_Chosen[i, 0]

        if (Alt_chosen == 1):
            temp = np.hstack((one_negative, iden_matrix))
        elif (Alt_chosen == nc):
            temp = np.hstack((iden_matrix, one_negative))
        else:
            ch = int(Alt_chosen)
            t1 = iden_matrix[:, 0:ch - 1]
            t2 = iden_matrix[:, ch - 1:nc - 1]
            temp = np.hstack((t1, one_negative, t2))

        M[nvar_cont:M.shape[0], nvar_cont:M.shape[1]] = temp
        del temp

        B_Tild = multi_dot([M, UY])
        Error_Tild = multi_dot([M, Full_error, M.T])

        BU_Tild = B_Tild[nvar_cont:B_Tild.shape[0]]
        if (nvar_cont > 1):
            Error_YU_Tild = Error_Tild[nvar_cont:Error_Tild.shape[0], 0:nvar_cont]
        else:
            Error_YU_Tild = Error_Tild[nvar_cont:Error_Tild.shape[0], 0]
            Error_YU_Tild = Error_YU_Tild[:, np.newaxis]

        Error_U_Tild = Error_Tild[nvar_cont:Error_Tild.shape[0], nvar_cont:Error_Tild.shape[1]]

        if (nvar_cont > 1):
            Error_Y = Error_Tild[0:nvar_cont, 0:nvar_cont]
            Diff_Mean = All_X[i, :][:, np.newaxis]
            Diff_Mean_mu = B_Tild[0:nvar_cont]
        else:
            Error_Y = Error_Tild[0, 0]
            Diff_Mean = All_X[i, :]
            Diff_Mean_mu = B_Tild[0]

        if (nvar_cont > 1):
            Mean_changed = BU_Tild + multi_dot([Error_YU_Tild, inv(Error_Y), (Diff_Mean - Diff_Mean_mu)])
            Error_changed = Error_U_Tild - multi_dot([Error_YU_Tild, inv(Error_Y), Error_YU_Tild.T])
        else:
            Mean_changed = BU_Tild + (Error_YU_Tild * (1 / Error_Y) * (Diff_Mean - Diff_Mean_mu))
            Error_changed = Error_U_Tild - (Error_YU_Tild * (1 / Error_Y) * (Error_YU_Tild.T))

        if (nvar_cont > 1):
            p1 = tpdf(Diff_Mean, Diff_Mean_mu, Error_Y, df)
        else:
            p1 = tpdf1(Diff_Mean, Diff_Mean_mu, Error_Y, df)

        mean_gu = -Mean_changed
        var_gu = Error_changed

        if (nvar_cont > 1):
            num_mult = multi_dot([(Diff_Mean - Diff_Mean_mu).T, inv(Error_Y), (Diff_Mean - Diff_Mean_mu)])
            num_multf = (df + num_mult) / (df + nvar_cont)
        else:
            num_mult = ((Diff_Mean - Diff_Mean_mu).T) * (1 / Error_Y) * (Diff_Mean - Diff_Mean_mu)
            num_multf = (df + num_mult) / (df + nvar_cont)

        var_gu = num_multf * var_gu
        om = np.diag(var_gu)
        om = om[:, np.newaxis]
        om = om ** 0.5
        mean_gu_final = np.divide(mean_gu.T, om.T)
        var_gu_final = corrvc(var_gu)
        df_final = df + nvar_cont

        if (nc > 3):
            seed20 = seednext
            p2, sss = cdfmvtGHK(mean_gu_final, var_gu_final, df_final, seed20)
            seednext = sss
        elif (nc == 3):
            if (np.mod(df_final, 2) == 0):
                p2 = bvt(mean_gu_final[0, 0], mean_gu_final[0, 1], var_gu_final[0, 1], df_final)
            else:
                p21 = bvt(mean_gu_final[0, 0], mean_gu_final[0, 1], var_gu_final[0, 1], ceil(df_final))
                p20 = bvt(mean_gu_final[0, 0], mean_gu_final[0, 1], var_gu_final[0, 1], floor(df_final))
                p2 = p21 - ((ceil(df_final) - df_final) * (p21 - p20))
        else:
            p2 = tcdf(mean_gu_final, df_final)

        if (p1 < upper_limit):
            p1 = upper_limit

        if (p2 < upper_limit):
            p2 = upper_limit

        LL = np.log(p1) + np.log(p2)
        Likelihood[obs_count, 0] = LL
    return (Likelihood)


def lgd_NM(parm_estimated, idx_estimated, idx_fixed, parm_fixed):
    eps = np.sqrt(np.finfo(np.float).eps)
    grad = ndd.approx_fprime(parm_estimated, lpr, eps, args=(idx_estimated, idx_fixed, parm_fixed), centered=False)
    return (grad)


def reconstruct(parm_estimated, idx_estimated, idx_fixed, parm_fixed):
    total_parm = parm_estimated.shape[0] + parm_fixed.shape[0]
    beta_reconstructed = np.empty(shape=(total_parm, 1), dtype=float)

    beta_reconstructed[idx_estimated] = parm_estimated
    beta_reconstructed[idx_fixed] = parm_fixed
    return (beta_reconstructed)
'''
-------------------------Simulation Setting-----------------------------------------------------------------------------
'''

global nchocc, upper_limit, nc, nind, nobs, seed, seed1, Non_IID, Num_Threads, Parametrized, D_matrix, Data_Split
output_path = "/home/prateek/T_DIst/"
nc = 5              # Number of alternatives
nvar_cont = 1       # Number of continuous variables
nind = 3000         # Number of observations
nobs = nind
Num_Threads = 15    # Number of threads

st_sample = 1   # Starting sample number
Total_sam = 50  # Total number of samples. These many samples will be generated and estimated. The results for all the samples are written in the Output_file_name 

Orignal_parm = 0     # Make this 1, if want to pass true parameter values as starting value
upper_limit = 1e-05  # Any value of CDF below this limit is considered as zero

Output_file_name = 'Results.xlsx'   # Name of the output file where all the results are written
'''  Output file Format:
sheet_name='Param_Estimates'     : Contains all the parameter estimates
sheet_name='Standard_Error'      : Contains all the parameter's standard errros
sheet_name='Estimation_time'     : Sample estimation time in mins
sheet_name='Likelihood_estimates : Log-likelihood value at convergence
'''

ncol = nc
_halt_numdraws = 200   # Number of halton draws 
nrep = _halt_numdraws
nrephalt = nrep
allHaltDraws = HaltonSequence(nobs * (nrep + 10), ncol)
_halt_maxdraws = allHaltDraws.shape[0]

global MACMLS
MACMLS = np.array([[300000, 300001, 300002, 300003, 300004, 300005, 300006, 300007, 300008, 300009, 300010, 300011,
                    300012, 300013, 300014, 300015, 300016, 300017, 300018, 300019, 300020]])

  

dataframe_col = []
global All_Parameters, All_Jacobian, All_Times, All_LL

sample_sucess = 0
for run_no in range(st_sample, st_sample + Total_sam, 1):
    seed = 623854 + run_no - 1  # Seed for the error term
    seed1 = 358696  # The seed for the data variable generation
    seed4 = 400 + run_no + 2

    vnames = ['income', 'child', 'education', ]

    ntot = len(vnames)
    np.random.seed(seed1)
    Taste_Het = np.random.uniform(0, 1, (nobs, ntot))
    X_Var = (Taste_Het > 0.5).astype(int)

    invar2 = ['Chosen', 'Den_2000', 'Den_0_99', 'Den_100_499', 'Den_500_1499', 'Den_1500_1999', 'Commute_dist', 'PID',
              'uno', 'sero']
    outvar = vnames + invar2
    outdata = np.hstack((X_Var, np.zeros((nobs, len(invar2) - 2)), np.ones((nobs, 1)), np.zeros((nobs, 1))))
    global Main_data
    Main_data = pd.DataFrame(outdata, index=range(0, nobs, 1), columns=outvar)
    del Taste_Het, X_Var, outdata

    # *****************************************************************************
    #                 True Value of Parameters
    # *****************************************************************************
    dgp_bcont = np.vstack((1.00, 0.50, 0.75, -0.50))  # Exogeneous variables coefficients for continuous variable
    dgp_bd = np.vstack((-1.50, 1.00, 0.90, 1.00,
                        -1.30, 0.90, 0.80, 0.90,
                        -1.20, 0.80, 0.70, 0.80,
                        -1.00, 0.70, 0.60, 0.70))  # Exogeneous variables coefficients for DC variable

    dgp_Psi = np.array([[1.50,
                         0.30, 1.00,
                         0.40, 0.50, 1.10,
                         0.60, 0.50, 0.50, 1.20,
                         0.50, 0.50, 0.50, 0.50, 1.30]]).T  # Covariance matrix for the differenced error term vector

    dgp_Psi = xpnd(dgp_Psi)

    psi_active = np.array([[1,
                            1, 0,
                            1, 0, 1,
                            1, 0, 0, 1,
                            1, 0, 0, 0, 1]]).T

    global nCholErr
    row_psi = dgp_Psi.shape[0]
    nCholErr = int((row_psi + 1) * (row_psi) * 0.5)
    DOF = 12

    # ****************************************************************************************************************************************************************************************
    #                 Packing of all parameters in a single vector ( Do not change anything below this line)
    # *****************************************************************************************************************************************************************************************
    dgp_X = dgp_bd
    dgp_X = np.vstack((dgp_X, dgp_bcont))
    dgp_X = np.vstack((dgp_X, vech(dgp_Psi), DOF))

    cont_01 = ['uno', 'income', 'child', 'education']

    # Utility specification for the alternatives
    Den_2000 = ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
                'sero', 'sero', 'sero', ]
    Den_0099 = ['uno', 'income', 'child', 'Commute_dist', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero',
                'sero', 'sero', 'sero', 'sero', 'sero', ]
    Den_0500 = ['sero', 'sero', 'sero', 'sero', 'uno', 'income', 'child', 'Commute_dist', 'sero', 'sero', 'sero',
                'sero', 'sero', 'sero', 'sero', 'sero', ]
    Den_1499 = ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno', 'income', 'child',
                'Commute_dist', 'sero', 'sero', 'sero', 'sero', ]
    Den_1999 = ['sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'sero', 'uno',
                'income', 'child', 'Commute_dist']

    global ivgenva, nvarma, ivgenvc, nvarmc
    ivgenva = Den_2000 + Den_0099 + Den_0500 + Den_1499 + Den_1999
    nvarma = len(Den_2000)

    ivgenvc = cont_01
    nvarmc = len(cont_01)

    global req_col, altchm, continuous_var
    req_col = ['Den_2000', 'Den_0_99', 'Den_100_499', 'Den_500_1499', 'Den_1500_1999']
    altchm = ['Chosen']
    continuous_var = ['Commute_dist']
    Chosen_Alt = dloop(dgp_X, seed)
    Share_data = Main_data.loc[:, req_col].as_matrix()
    Share = np.sum(Share_data, axis=0)
    Share = Share / nobs
    ID_data = Main_data.loc[:, vnames].as_matrix()
    ID_Share = np.sum(ID_data, axis=0)
    ID_Share = ID_Share / nobs

    print("Sample Number: ", run_no)
    print('-------------------------------------------')
    # print("Alternative Share :")
    # for i in range(0,nc,1):
    #   print('{0:20s}'.format(req_col[i]).ljust(19) + '{0:1.2f}'.format(Share[i]).ljust(10))
    # # print("ID Share :")
    # # for i in range(0,len(vnames),1):
    # #   print('{0:10s}'.format(vnames[i]).ljust(19) + '{0:1.2f}'.format(ID_Share[i]).ljust(10))
    # print('-------------------------------------------')

    if (Orignal_parm == 1):
        dgp_bd1 = dgp_bd
        dgp_bcont1 = dgp_bcont
        dgp_Psi1 = dgp_Psi
        dgp_dof1 = DOF
    else:
        dgp_bd1 = 0.1 * np.ones((dgp_bd.shape[0], 1))
        dgp_bcont1 = 0.1 * np.ones((dgp_bcont.shape[0], 1))
        dgp_Psi1 = np.eye(dgp_Psi.shape[0])
        dgp_dof1 = 1

    dgp_X1 = dgp_bd1
    dgp_X1 = np.vstack((dgp_X1, dgp_bcont1))
    dgp_X1 = np.vstack((dgp_X1, vech(dgp_Psi1), dgp_dof1))

    bb = dgp_X1[0:nvarma + nvarmc]
    bb = np.vstack((bb, vech(chol(xpnd(dgp_X1[nvarma + nvarmc:nvarma + nvarmc + nCholErr]))),
                    log(dgp_X1[nvarma + nvarmc + nCholErr])))

    max_active = np.ones((nvarma + nvarmc, 1))
    max_active = np.vstack((max_active, psi_active, 1))

    idx_estimated = np.where(max_active == 1)  # Index of estimated variables in beta vector
    idx_fixed = np.where(max_active == 0)  # Index of fixed variables in beta vector

    idx_estimatedt = idx_estimated[0]
    idx_estimatedt = idx_estimatedt[:, np.newaxis]

    # Split the starting value vector into parameters that are estimated and fixed
    init_estimated = bb[idx_estimated]
    init_fixed = bb[idx_fixed]

    # Defining variable labels

    Param_nam = []
    for i in range(0, nvarma, 1):
        if (i < 9):
            Param_nam.append('Beta0' + str(i + 1))
        else:
            Param_nam.append('Beta' + str(i + 1))

    if (nvar_cont > 0):
        for i in range(0, nvarmc, 1):
            if (i < 9):
                Param_nam.append('Cont0' + str(i + 1))
            else:
                Param_nam.append('Cont' + str(i + 1))

    Param_nam.append('Psi11')
    for i in range(1, dgp_Psi.shape[0], 1):
        for j in range(0, i + 1, 1):
            Param_nam.append('Psi' + str(i + 1) + str(j + 1))

    Param_nam.append('DOF')

    Active_name = []
    for i in range(0, int(np.sum(max_active, axis=0)[0]), 1):
        Active_name.append(Param_nam[idx_estimatedt[i, 0]])

    D_matrix = np.zeros((nvar_cont + nc, nvar_cont + nc - 1))
    if (nvar_cont > 1):
        D_matrix[0:nvar_cont, 0:nvar_cont] = np.eye(nvar_cont)
    else:
        D_matrix[0, 0] = 1
    D_matrix[nvar_cont + 1:D_matrix.shape[0], nvar_cont:D_matrix.shape[1]] = np.eye(nc - 1)

    Data_Split = np.zeros((Num_Threads, 2))
    for i in range(1, Num_Threads + 1, 1):
        Data_Split[i - 1, 0] = int(ceil((i - 1) * (nind / Num_Threads)) + 1)
        Data_Split[i - 1, 1] = int(ceil((i) * (nind / Num_Threads)))

    max_gradTol = 1e-04
    max_iter = 120
    Nfeval = 1
    Parametrized = 1


    def callbackF(Xi):
        global Nfeval

        print('================================================================================')
        print()
        print("Iter_no: {0:1d}".format(Nfeval))
        print('--------------------------------------------------------------------')
        print('{0:10s}'.format('Parm_name').ljust(20) + '{0:10s}'.format('Value').ljust(10))
        for i in range(0, int(np.sum(max_active, axis=0)[0]), 1):
            if (Xi[i] < 0):
                print('{0:10s}'.format(Active_name[i]).ljust(19) + '{0:3.4f}'.format(Xi[i]).ljust(10))
            if (Xi[i] >= 0):
                print('{0:10s}'.format(Active_name[i]).ljust(20) + '{0:3.4f}'.format(Xi[i]).ljust(10))
        print()

        Nfeval += 1


    print("Optimization has Started.............")
    start_time = time.time()
    [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg, allvecs] = fmin_bfgs(lpr, init_estimated, fprime=lgd_NM,
                                                                                   args=(idx_estimated, idx_fixed,
                                                                                         init_fixed), gtol=max_gradTol,
                                                                                   epsilon=1.4901161193847656e-08,
                                                                                   maxiter=max_iter, full_output=1,
                                                                                   disp=1, retall=1, callback=callbackF)
    # [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg, allvecs] = fmin_bfgs(lpr, init_estimated, args=(idx_estimated, idx_fixed, init_fixed), gtol=max_gradTol, epsilon=1.4901161193847656e-08, maxiter=max_iter, full_output=1, disp=1, retall=1, callback=callbackF)

    xComplete = bb
    xComplete[idx_estimated] = xopt
    xComplete[idx_fixed] = init_fixed

    temp = xComplete[nvarma + nvarmc:nvarma + nvarmc + nCholErr]
    cholPsi = (upmat(xpnd(temp))).T
    row_num = nvar_cont
    if (nvar_cont > 1):
        value = cholPsi[row_num, 0:row_num]
        value = value[:, np.newaxis]
        value = multi_dot([value.T, value])
        denom = sqrt(1 + value)
        for i in range(0, row_num, 1):
            cholPsi[row_num, i] = cholPsi[row_num, i] / denom
        cholPsi[row_num, row_num] = 1 / denom
        cholPsi = cholPsi.T
    else:
        value = cholPsi[row_num, 0]
        value = value * value
        denom = sqrt(1 + value)
        cholPsi[row_num, 0] = cholPsi[row_num, 0] / denom
        cholPsi[row_num, 1] = 1 / denom
        cholPsi = cholPsi.T
    Psi = multi_dot([cholPsi.T, cholPsi])
    check1 = np.isfinite(cond(Psi))
    det1 = det(Psi)
    check2 = det1 > 0.01
    check3 = is_pos_def(Psi)
    check = check1 & check2 & check3

    Psi = vech(Psi)
    xComplete[nvarma + nvarmc:nvarma + nvarmc + nCholErr, 0] = Psi[:, 0]
    xComplete[nvarma + nvarmc + nCholErr, 0] = exp(xComplete[nvarma + nvarmc + nCholErr, 0])
    if (check):
        sample_sucess = sample_sucess + 1
        Parametrized = 0

        parm_estimated = xComplete[idx_estimated]
        parm_fixed = xComplete[idx_fixed]

        MNP_lpr = lpr(parm_estimated, idx_estimated, idx_fixed, parm_fixed)
        MNP_lgd = lgd_NM(parm_estimated, idx_estimated, idx_fixed, parm_fixed)

        IM_all = np.zeros((max_active.shape[0], 1))

        Jacobian = multi_dot([MNP_lgd.T, MNP_lgd])
        IM = np.sqrt(np.diag(np.linalg.inv(Jacobian)))
        IM_all[idx_estimated] = IM
        xopt = parm_estimated
        Total_LL = -np.sum(MNP_lpr, axis=0)[0]
        end_time = time.time()
        total_time = (end_time - start_time) / 60

        print('============Final Results=============================')
        print("Estimation time: {0:6.2f}".format(total_time))
        print()
        print("Log-Likelihood Value: {0:6.3f}".format(Total_LL))
        print('--------------------------------------------------------------------')
        print('{0:10s}'.format('Parm_name').ljust(19) + '{0:10s}'.format('Estimate').ljust(16) + '{0:10s}'.format(
            'T-stat').ljust(14))
        for i in range(0, max_active.shape[0], 1):
            if (xComplete[i, 0] < 0 and max_active[i, 0] == 1):
                if (abs(xComplete[i, 0]) >= 10):
                    print('{0:10s}'.format(Param_nam[i]).ljust(19) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(
                        14) + '{0:3.3f}'.format(xComplete[i, 0] / IM_all[i, 0]).ljust(14))
                if (abs(xComplete[i, 0]) < 10):
                    print('{0:10s}'.format(Param_nam[i]).ljust(19) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(
                        15) + '{0:3.3f}'.format(xComplete[i, 0] / IM_all[i, 0]).ljust(14))
            if (xComplete[i, 0] > 0 and max_active[i, 0] == 1):
                if (xComplete[i, 0] >= 10):
                    print('{0:10s}'.format(Param_nam[i]).ljust(20) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(
                        14) + '{0:3.3f}'.format(xComplete[i, 0] / IM_all[i, 0]).ljust(15))
                if (xComplete[i, 0] < 10):
                    print('{0:10s}'.format(Param_nam[i]).ljust(20) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(
                        15) + '{0:3.3f}'.format(xComplete[i, 0] / IM_all[i, 0]).ljust(15))
            if (xComplete[i, 0] == 0 or max_active[i, 0] == 0):
                print('{0:10s}'.format(Param_nam[i]).ljust(20) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(
                    15) + '{0:3.3f}'.format(0.000).ljust(15))

        if (run_no <= 9):
            dataframe_col.append('Sample_0' + str(run_no))
        else:
            dataframe_col.append('Sample_' + str(run_no))

        if (sample_sucess == 1):
            All_Parameters = xComplete
            All_Jacobian = IM_all
            All_Times = total_time
            All_LL = Total_LL
        else:
            All_Parameters = np.hstack((All_Parameters, xComplete))
            All_Jacobian = np.hstack((All_Jacobian, IM_all))
            All_Times = np.vstack((All_Times, total_time))
            All_LL = np.vstack((All_LL, Total_LL))

All_betas = pd.DataFrame(All_Parameters, index=range(0, All_Parameters.shape[0], 1), columns=dataframe_col)
All_error = pd.DataFrame(All_Jacobian, index=range(0, All_Jacobian.shape[0], 1), columns=dataframe_col)
All_times = pd.DataFrame(All_Times, index=range(0, All_Times.shape[0], 1), columns=['Time'])
All_LL = pd.DataFrame(All_LL, index=range(0, All_LL.shape[0], 1), columns=['Log-Likelihood'])

writer = pd.ExcelWriter(output_path+Output_file_name, engine='xlsxwriter')
All_betas.to_excel(writer,sheet_name='Param_Estimates')
All_error.to_excel(writer,sheet_name='Standard_Error')
All_times.to_excel(writer,sheet_name='Estimation_time')
All_LL.to_excel(writer,sheet_name='Likelihood_estimates')
writer.save()



