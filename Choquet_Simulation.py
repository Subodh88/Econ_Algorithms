import numpy as np
import pandas as pd
from scipy.stats import t, norm
from math import *
from numpy.linalg import inv, det, cholesky, multi_dot, cond
from scipy.linalg import cho_solve, cho_factor, solve, pinv
import sys
from statsmodels.sandbox.distributions.multivariate import mvstdtprob, mvstdnormcdf
from multiprocessing import Pool
from functools import partial
from scipy.optimize import fmin_bfgs,fmin_slsqp
import statsmodels.tools.numdiff as ndd
#import numdifftools as nd
import time
import sys
import operator
from functools import reduce
import itertools
from itertools import combinations
from scipy.special import factorial
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(precision=5)


# Procedure to calculate multivariate Normal-distribution PDF
def pdfmvn(x, mu, s):
    d = x.shape[0]
    temp1 = multi_dot([(x - mu).T, inv(s), (x - mu)])
    p1 = exp(-0.5 * temp1)
    p2 = ((2 * pi) ** (d / 2)) * sqrt(det(s))
    p = p1 / p2
    return (p)


# Procedure to calculate univariate Normal-distribution PDF
def pdfmvn1(x, mu, s):
    d = 1
    temp1 = (x - mu) * (1 / s) * (x - mu)
    p1 = exp(-0.5 * temp1)
    p2 = ((2 * pi) ** (d / 2)) * sqrt(s)
    p = p1 / p2
    return (p)


# Procedure to generate Halton Draws
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


# GHK simulator for calculation of multivariate Normal-distribution CDF
def cdfmvnGHK(a, r, s):
    global _halt_maxdraws, _halt_numdraws, allHaltDraws, nrep
    a = np.multiply(a, (a < 5.7)) + 5.7 * (a >= 5.7)
    a = np.multiply(a, (a > -5.7)) - 5.7 * (a <= -5.7)

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
    chol_r = chol_r.T
    ghkArr = np.zeros((nrep, nintegdim))
    etaArr = np.zeros((nrep, (nintegdim - 1)))
    temp = norm.cdf(a[0, 0] / chol_r[0, 0]) * np.ones((nrep, 1))
    ghkArr[:, 0] = temp[:, 0]
    del temp

    for iintegdim_main in range(1, nintegdim, 1):
        iintegdim = iintegdim_main - 1
        temp1 = uniRands[:, iintegdim]
        temp2 = ghkArr[:, iintegdim]
        temp1 = temp1[:, np.newaxis]
        temp2 = temp2[:, np.newaxis]

        temp3 = np.multiply(temp1, temp2)
        temp4 = cdfni(temp3)
        temp4 = temp4[:, np.newaxis]
        etaArr[:, iintegdim] = temp4[:, 0]
        del temp1, temp2, temp3, temp4

        ghkElem = a[0, iintegdim + 1] * np.ones((nrep, 1))
        ghkElem1 = 0
        for jintegdim in range(0, iintegdim_main, 1):
            temp = chol_r[jintegdim, iintegdim + 1] * etaArr[:, jintegdim]
            temp = temp[:, np.newaxis]
            ghkElem1 = ghkElem1 - temp
            del temp

        ghkElem1 = ghkElem1 + ghkElem
        temp1 = ghkElem1 / (chol_r[(iintegdim + 1), (iintegdim + 1)])
        temp2 = cdfn(temp1)
        temp2 = temp2[:, np.newaxis]
        ghkArr[:, iintegdim + 1] = temp2[:, 0]
        del temp1, temp2

    probab = ghkArr[:, 0]
    probab = probab[:, np.newaxis]
    for iintegdim in range(1, nintegdim, 1):
        temp = ghkArr[:, iintegdim]
        temp = temp[:, np.newaxis]
        probab = np.multiply(probab, temp)
        del temp

    probab = np.mean(probab, axis=0)[0]
    return (probab, s)


# Procedure to obtain standard normal distribution CDF
def cdfn(a):
    out = norm.cdf(a[:, 0])
    return (out)


# Procedure to obtain bi-variate normal distribution CDF
def cdfbvn(a1, a2, r):
    low_x = -np.inf * np.ones(2)
    up_x = np.array([a1, a2])
    return (mvstdnormcdf(low_x, up_x, r, abseps=1e-6))


# Procedure to obtain inverse of univariate normal distribution CDF
def cdfni(a):
    out = norm.ppf(a[:, 0])
    return (out)


# Procedure to convert co-variance matrix into correlation matrix
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


# Procedure to obtain Cholesky decomposition
def chol(r):
    a = cholesky(r)
    return (a)


# Procedure to put 1 on diagonal of a matrix
def diagrv(a):
    for i in range(0, a.shape[0], 1):
        a[i, i] = 1.0
    return (a)


# Procedure to check positive-definiteness of a matrix
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


# Procedure to check positive-definiteness of a matrix
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


# Procedure to check positive-definiteness of a matrix
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# Procedure to expand a vector into symmetric matrix
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


# Procedure to vectorize a symmetric matrix
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


# Procedure to extract upper triangular matrix
def upmat(r):
    drow = r.shape[0]
    xp = np.zeros((drow, drow))
    for i in range(0, drow, 1):
        for j in range(i, drow, 1):
            xp[i, j] = r[i, j]
    return (xp)


# Procedure to extract lower triangular matrix
def lowmat(r):
    drow = r.shape[0]
    xp = np.zeros((drow, drow))
    xp[0, 0] = r[0, 0]
    for i in range(1, drow, 1):
        for j in range(0, i + 1, 1):
            xp[i, j] = r[i, j]
    return (xp)


def combs(a):
    if len(a) == 0:
        return [[]]
    cs = []
    for c in combs(a[1:]):
        cs += [c, c + [a[0]]]
    return cs


def rSubset(arr, r):
    return list(combinations(arr, r))


def all_comb(k):
    arr = [*range(1, k + 1, 1)]
    all_list = []
    for i in range(0, len(arr), 1):
        all_list.append(str(arr[i]))

    for i in range(2, len(arr) + 1, 1):
        arr2 = rSubset(arr, i)
        for j in range(0, len(arr2), 1):
            curr_ele = arr2[j]
            comb = ''
            for k in range(0, len(curr_ele), 1):
                comb += str(curr_ele[k])
            all_list.append((comb))
    return (all_list)


def all_mu_values(mu_value):
    k = mu_value.shape[0]
    all_list = all_comb(k)
    all_mu = []
    for i in range(0, len(all_list), 1):
        arr2 = all_list[i]
        total_sum = 0
        for j in range(0, len(arr2), 1):
            curr_ele = int(arr2[j])
            total_sum += mu_value[curr_ele - 1, 0]
        all_mu.append(total_sum)
    return (all_mu)


def isBlank (myString):
    if myString and myString.strip():
        #myString is not None AND myString is not empty or blank
        return 0
    #myString is None OR myString is empty or blank
    return 1

def Shapley(mu_arr, num_attribute):
    out = np.zeros((num_attribute, 1))
    for ir in range(0, num_attribute, 1):
        i = ir + 1
        A = []

        for inum in range(1, num_attribute + 1, 1):
            A.append(inum)
        A.remove(i)
        all_comb = combs(A)
        total_sum = 0
        for jr in range(0, len(all_comb), 1):
            curr_ele = all_comb[jr]
            curr_size = len(curr_ele)
            ele_store = []
            for istore in range(0, curr_size, 1):
                ele_store.append(int(curr_ele[istore]))

            ele_store1 = deepcopy(ele_store)
            ele_store2 = deepcopy(ele_store)

            ele_store1.append(int(i))
            ele_store1f = sorted(ele_store1)
            ele_store2f = sorted(ele_store2)

            curr_index1 = ''
            curr_index2 = ''
            for jr2 in range(0, len(ele_store1f), 1):
                curr_index1 += str(ele_store1f[jr2])

            for jr2 in range(0, len(ele_store2f), 1):
                curr_index2 += str(ele_store2f[jr2])
            
            if(isBlank(curr_index2) == 1):
                total_sum += ((factorial(num_attribute - len(ele_store) - 1) * factorial(len(ele_store))) / factorial(
                num_attribute)) * (mu_arr.loc[int(curr_index1), 'mu'])
            else:
                total_sum += ((factorial(num_attribute - len(ele_store) - 1) * factorial(len(ele_store))) / factorial(
                num_attribute)) * (mu_arr.loc[int(curr_index1), 'mu'] - mu_arr.loc[int(curr_index2), 'mu'])

        out[ir, 0] = total_sum
    return (out)



def Mobius(mu_arr):
    att_comb = mu_ordering
    mu_value = mu_arr[:, 0]
    df_mu_temp = pd.DataFrame(mu_value, index=(att_comb), columns=['mu'])
    out = np.zeros((len(att_comb), 1))
    for ir in range(0, len(att_comb), 1):
        i = str(att_comb[ir])
        if (len(i) == 1):
            out[ir, 0] = mu_value[ir]
        else:
            A = [int(d) for d in i]
            all_comb = combs(A)
            total_sum = 0
            for jr in range(1, len(all_comb), 1):
                curr_ele = all_comb[jr]
                curr_size = len(curr_ele)
                ele_store = []
                for istore in range(0, curr_size, 1):
                    ele_store.append(int(curr_ele[istore]))

                ele_store = sorted(ele_store)
                curr_index1 = ''

                for jr2 in range(0, len(ele_store), 1):
                    curr_index1 += str(ele_store[jr2])

                total_sum += pow(-1, len(i) - curr_size) * (df_mu_temp.loc[int(curr_index1), 'mu'])
            out[ir, 0] = total_sum

    return (out)


def Fuzzy(mu_arr):
    att_comb = mu_ordering
    mu_value = mu_arr[:, 0]
    df_mu_temp = pd.DataFrame(mu_value, index=(att_comb), columns=['mb'])
    out = np.zeros((len(att_comb), 1))
    for ir in range(0, len(att_comb), 1):
        i = str(att_comb[ir])
        if (len(i) == 1):
            out[ir, 0] = mu_value[ir]
        else:
            A = [int(d) for d in i]
            all_comb = combs(A)
            total_sum = 0
            for jr in range(1, len(all_comb), 1):
                curr_ele = all_comb[jr]
                curr_size = len(curr_ele)
                ele_store = []
                for istore in range(0, curr_size, 1):
                    ele_store.append(int(curr_ele[istore]))

                ele_store = sorted(ele_store)
                curr_index1 = ''

                for jr2 in range(0, len(ele_store), 1):
                    curr_index1 += str(ele_store[jr2])

                total_sum += (df_mu_temp.loc[int(curr_index1), 'mb'])
            out[ir, 0] = total_sum

    return (out)


def Interaction_Index(mu_arr, num_attribute):
    out = np.ones((num_attribute, num_attribute))
    for ir1 in range(0, num_attribute, 1):
        for ir2 in range(ir1 + 1, num_attribute, 1):
            i = ir1 + 1
            j = ir2 + 1
            A = []

            for inum in range(1, num_attribute + 1, 1):
                A.append(inum)
            A.remove(i)
            A.remove(j)

            if (len(A) > 1):
                all_comb = combs(A)
            else:
                all_comb = [[], A]

            total_sum = 0
            for jr in range(0, len(all_comb), 1):
                curr_ele = all_comb[jr]
                curr_size = len(curr_ele)
                ele_store = []
                for istore in range(0, curr_size, 1):
                    ele_store.append(int(curr_ele[istore]))

                ele_store1 = deepcopy(ele_store)
                ele_store2 = deepcopy(ele_store)
                ele_store3 = deepcopy(ele_store)
                ele_store4 = deepcopy(ele_store)

                ele_store1.append(int(i))
                ele_store1.append(int(j))

                ele_store2.append(int(i))

                ele_store3.append(int(j))

                ele_store1f = sorted(ele_store1)
                ele_store2f = sorted(ele_store2)
                ele_store3f = sorted(ele_store3)
                ele_store4f = sorted(ele_store4)

                curr_index1 = ''
                curr_index2 = ''
                curr_index3 = ''
                curr_index4 = ''
                for jr2 in range(0, len(ele_store1f), 1):
                    curr_index1 += str(ele_store1f[jr2])

                for jr2 in range(0, len(ele_store2f), 1):
                    curr_index2 += str(ele_store2f[jr2])

                for jr2 in range(0, len(ele_store3f), 1):
                    curr_index3 += str(ele_store3f[jr2])

                for jr2 in range(0, len(ele_store4f), 1):
                    curr_index4 += str(ele_store4f[jr2])

                if(isBlank(curr_index4) == 1):
                    total_sum += ((factorial(num_attribute - len(ele_store) - 2) * factorial(
                    len(ele_store))) / factorial(
                    num_attribute - 1)) * (mu_arr.loc[int(curr_index1), 'mu'] - mu_arr.loc[int(curr_index2), 'mu'] -
                                           mu_arr.loc[int(curr_index3), 'mu'])
                else:
                    total_sum += ((factorial(num_attribute - len(ele_store) - 2) * factorial(
                    len(ele_store))) / factorial(
                    num_attribute - 1)) * (mu_arr.loc[int(curr_index1), 'mu'] - mu_arr.loc[int(curr_index2), 'mu'] -
                                           mu_arr.loc[int(curr_index3), 'mu'] + mu_arr.loc[int(curr_index4), 'mu'])
            out[ir1, ir2] = total_sum
            out[ir2, ir1] = total_sum
    return (out)


def Normalize_positive(a):
    return ((a - np.min(a)) / (np.max(a) - np.min(a)))


def Normalize_negative(a):
    return ((np.max(a) - a) / (np.max(a) - np.min(a)))

def all_comb_xvalues(X_values):
    k = X_values.shape[1]
    arr = [*range(1, k + 1, 1)]
    for i in range(0, len(arr), 1):
        curr_arr = X_values[:,i]
        curr_arr = curr_arr[:, np.newaxis]
        if(i == 0):
            AllX = curr_arr
        else:
            AllX = np.hstack((AllX,curr_arr))
    
    for i in range(2, len(arr) + 1, 1):
        arr2 = rSubset(arr, i)
        for j in range(0, len(arr2), 1):
            curr_ele = arr2[j]
            curr_index = int(curr_ele[0]-1)
            curr_arr = X_values[:,curr_index]
            curr_arr = curr_arr[:, np.newaxis]

            for k in range(1, len(curr_ele), 1):
                curr_index = int(curr_ele[k]-1)
                temp = X_values[:,curr_index]
                temp = temp[:,np.newaxis]
                curr_arr = np.hstack((curr_arr,temp))

            all_prod = np.prod(curr_arr,axis=1)
            all_prod = all_prod[:,np.newaxis]
            AllX = np.hstack((AllX,all_prod))
    return (AllX)

# Data Generating procedure considering Probit Kernel for the simulation framework used in the paper
def dloopN(parm, seeddata):
    smallb = parm[0:nvarma_rum]
    smallc = parm[nvarma_rum:nvarma_rum + nvarma_chq_actual]
    if(Choquet_Model == 1):
        df_mu_temp = pd.DataFrame(smallc, index=(mu_ordering), columns=['mu'])

    if(Probit == 1):
        temp = parm[nvarma_rum + nvarma_chq_actual:nvarma_rum + nvarma_chq_actual + nCholErr]
        Psi = xpnd(temp)
        chol_psi = (chol(Psi)).T
    if(Non_comp == 1):
        att_cutoff_all = parm[nvarma_rum + nvarma_chq_actual + nCholErr:parm.shape[0]]
        att_cutoff = np.zeros((nvarma_chq,4))
        for i in range(0,nvarma_chq,1):
            if(var_cutoff_type[0,i]>0):
                if(i==0):
                    curr_cutoff_values = att_cutoff_all[0:var_cutoff_cumsum[i]]
                    curr_cutoff_values = curr_cutoff_values.T
                    att_cutoff[i,0:var_cutoff_type[0,i]] = curr_cutoff_values[0,:]
                else:
                    curr_cutoff_values = att_cutoff_all[var_cutoff_cumsum[i-1]:var_cutoff_cumsum[i]]
                    curr_cutoff_values = curr_cutoff_values.T
                    att_cutoff[i,0:var_cutoff_type[0,i]] = curr_cutoff_values

    
    np.random.seed(seeddata)
    if(Probit == 1):
        err_psid = np.random.normal(0, 1, (nind, nc - 1))
        err_psi = multi_dot([err_psid, chol_psi])
        err_psi = np.hstack((np.zeros((nind, 1)), err_psi))
    if(Logit == 1):
        rnum = np.random.random((nind,nc))
        err_psi = -np.log(-np.log(rnum))

    v1 = (np.kron(np.ones((nc, 1)), smallb)) * (Main_data.loc[:, ivgenva_rum].as_matrix().T)
    v = np.empty(shape=(nobs, nc), dtype=float)
    j = 0
    for i in range(nc):
        j = i + 1
        v[:, i] = np.sum(v1[(j - 1) * nvarma_rum:(j * nvarma_rum), :], axis=0)
    del v1

    Utility_temp = v + err_psi

    data_chq = Main_data.loc[:, ivgenva_chq].as_matrix()
    count = 0
    for i in range(0, nvarma_chq, 1):
        for j in range(0, nc, 1):
            curr_arr = data_chq[:, nvarma_chq * j + i]
            curr_arr = curr_arr[:, np.newaxis]
            if (count == 0):
                X_chq = curr_arr
            else:
                X_chq = np.hstack((X_chq, curr_arr))
            count += 1

    
    if(Non_comp == 1):
        for i in range(0, nvarma_chq, 1):
            j = i + 1
            curr_value = X_chq[:, (j - 1) * nc:(j * nc)]
            curr_cutoffs = att_cutoff[i,:]
            temp = np.reshape(curr_value,(nind*nc))
            if(var_cutoff_type[0,i]==2):
                if (var_ind_chq[0, i] == 0):

                    y = np.zeros(temp.shape[0])

                    idx = np.where((temp <= curr_cutoffs[0]))[0]
                    y[idx] = np.ones(len(idx))

                    idx = np.where((curr_cutoffs[0] < temp) & (temp <= curr_cutoffs[1]))[0]
                    y[idx] = (curr_cutoffs[1]-temp[idx])/(curr_cutoffs[1]-curr_cutoffs[0])

                    idx = np.where((temp > curr_cutoffs[1]))[0]
                    y[idx] = np.zeros(len(idx))

                    yf = np.reshape(y,(nind,nc))
                
                elif (var_ind_chq[0, i] == 1):
                    y = np.zeros(temp.shape[0])

                    idx = np.where((temp <= curr_cutoffs[0]))[0]
                    y[idx] = np.zeros(len(idx))

                    idx = np.where((curr_cutoffs[0] < temp) & (temp <= curr_cutoffs[1]))[0]
                    y[idx] = (temp[idx]-curr_cutoffs[0])/(curr_cutoffs[1]-curr_cutoffs[0])

                    idx = np.where((temp > curr_cutoffs[1]))[0]
                    y[idx] = np.ones(len(idx))

                    yf = np.reshape(y,(nind,nc))
                
                if (i == 0):
                    X_chq_nom = yf
                else:
                    X_chq_nom = np.hstack((X_chq_nom, yf))
            elif(var_cutoff_type[0,i]==3):
                y = np.zeros(temp.shape[0])

                idx = np.where((temp <= curr_cutoffs[0]))[0]
                y[idx] = np.zeros(len(idx))

                idx = np.where((curr_cutoffs[0] < temp) & (temp <= curr_cutoffs[1]))[0]
                y[idx] = (temp[idx]-curr_cutoffs[0])/(curr_cutoffs[1]-curr_cutoffs[0])

                idx = np.where((curr_cutoffs[1] < temp) & (temp <= curr_cutoffs[2]))[0]
                y[idx] = (curr_cutoffs[2]-temp[idx])/(curr_cutoffs[2]-curr_cutoffs[1])

                idx = np.where((temp > curr_cutoffs[2]))[0]
                y[idx] = np.zeros(len(idx))

                yf = np.reshape(y,(nind,nc))
                             
                if (i == 0):
                    X_chq_nom = yf
                else:
                    X_chq_nom = np.hstack((X_chq_nom, yf))
            elif(var_cutoff_type[0,i]==4):
                y = np.zeros(temp.shape[0])

                idx = np.where((temp <= curr_cutoffs[0]))[0]
                y[idx] = np.zeros(len(idx))

                idx = np.where((curr_cutoffs[0] < temp) & (temp <= curr_cutoffs[1]))[0]
                y[idx] = (temp[idx]-curr_cutoffs[0])/(curr_cutoffs[1]-curr_cutoffs[0])

                idx = np.where((curr_cutoffs[1] < temp) & (temp <= curr_cutoffs[2]))[0]
                y[idx] = np.ones(len(idx))

                idx = np.where((curr_cutoffs[2] < temp) & (temp <= curr_cutoffs[3]))[0]
                y[idx] = (curr_cutoffs[3]-temp[idx])/(curr_cutoffs[3]-curr_cutoffs[2])

                idx = np.where((temp > curr_cutoffs[3]))[0]
                y[idx] = np.zeros(len(idx))

                yf = np.reshape(y,(nind,nc))
                             
                if (i == 0):
                    X_chq_nom = yf
                else:
                    X_chq_nom = np.hstack((X_chq_nom, yf))
            elif(var_cutoff_type[0,i]==0):
                if (var_ind_chq[0, i] == 1):
                    value_nom = np.apply_along_axis(Normalize_positive, 1, curr_value)
                elif (var_ind_chq[0, i] == 0):
                    value_nom = np.apply_along_axis(Normalize_negative, 1, curr_value)

                if (i == 0):
                    X_chq_nom = value_nom
                else:
                    X_chq_nom = np.hstack((X_chq_nom, value_nom))

    else:
        for i in range(0, nvarma_chq, 1):
            j = i + 1
            curr_value = X_chq[:, (j - 1) * nc:(j * nc)]
            if(RUM_Model_basic == 0):
                if (var_ind_chq[0, i] == 1):
                    value_nom = np.apply_along_axis(Normalize_positive, 1, curr_value)
                elif (var_ind_chq[0, i] == 0):
                    value_nom = np.apply_along_axis(Normalize_negative, 1, curr_value)

            if(RUM_Model_basic == 1):
                value_nom = curr_value
            if (i == 0):
                X_chq_nom = value_nom
            else:
                X_chq_nom = np.hstack((X_chq_nom, value_nom))

    X_chq_nom_final = np.zeros((nind, nc * nvarma_chq))
    count = -1
    for i in range(0, nc, 1):
        for j in range(0, nvarma_chq, 1):
            k = j + 1
            count += 1
            X_chq_nom_final[:, count] = X_chq_nom[:, (k - 1) * nc + i]

    if((RUM_Model == 1) | (RUM_Constraint == 1)):
        for i in range(0,nc,1):
            j = i+1
            curr_exp_values = X_chq_nom_final[:,(j - 1) * nvarma_chq:(j * nvarma_chq)]
            all_poss_comb = all_comb_xvalues(curr_exp_values)
            if(i == 0):
                New_Xvalue = all_poss_comb
            else:
                New_Xvalue = np.hstack((New_Xvalue,all_poss_comb))

        v1 = (np.kron(np.ones((nc, 1)), smallc)) * (New_Xvalue.T)
        Utility_chq = np.empty(shape=(nobs, nc), dtype=float)
        j = 0
        for i in range(nc):
            j = i + 1
            Utility_chq[:, i] = np.sum(v1[(j - 1) * nvarma_chq_actual:(j * nvarma_chq_actual), :], axis=0)
        del v1

    elif(Choquet_Model == 1):
        Utility_chq = np.zeros((nind, nc))
        for i in range(0, nind, 1):
            curr_x_values = X_chq_nom_final[i, :]
            curr_x_values = np.reshape(curr_x_values, (nc, nvarma_chq))
            for ialt in range(0, nc, 1):
                A = np.zeros((nvarma_chq, 2))
                for ivar in range(0, nvarma_chq, 1):
                    A[ivar, 0] = ivar + 1
                    A[ivar, 1] = curr_x_values[ialt, ivar]

                B = A[np.argsort(A[:, 1])[::-1]]

                mu_label = []
                for k in range(0, nvarma_chq, 1):
                    if (k == 0):
                        mu_label.append(str(int(B[k, 0])))
                    else:
                        ele_store = []
                        for m in range(0, k + 1, 1):
                            ele_store.append(int(B[m, 0]))

                        ele_store = sorted(ele_store)
                        comb = ''
                        for m in range(0, len(ele_store), 1):
                            comb += str(ele_store[m])
                        mu_label.append(comb)

                chq_value = 0
                for m in range(0, nvarma_chq, 1):
                    if (i == 0):
                        curr_index2 = mu_label[m]
                        chq_value += B[m, 1] * df_mu_temp.loc[int(curr_index2), 'mu']
                    else:
                        curr_index2 = mu_label[m]
                        curr_index1 = mu_label[m - 1]
                        chq_value += B[m, 1] * (
                                    df_mu_temp.loc[int(curr_index2), 'mu'] - df_mu_temp.loc[int(curr_index1), 'mu'])

                Utility_chq[i, ialt] = chq_value
    elif(RUM_Model_basic == 1):
        v1 = (np.kron(np.ones((nc, 1)), smallc)) * (X_chq_nom_final.T)
        Utility_chq = np.empty(shape=(nobs, nc), dtype=float)
        j = 0
        for i in range(nc):
            j = i + 1
            Utility_chq[:, i] = np.sum(v1[(j - 1) * nvarma_chq_actual:(j * nvarma_chq_actual), :], axis=0)
        del v1

    Utility_temp = Utility_temp + Utility_chq

    del err_psi, v
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


# Multi-threaded Likelihood Calling procedure
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
    a_temp = list(itertools.chain.from_iterable(result_list))
    atemp_array = np.asarray(a_temp)
    if (Parametrized == 1):
        return (-np.mean(atemp_array, axis=0))

    if (Parametrized == 0):
        return (-atemp_array)


# Likelihood procedure
def lprT(iter, parm):
    st_iter = int(Data_Split[iter, 0])
    end_iter = int(Data_Split[iter, 1])
    num_obs = int(end_iter - st_iter + 1)
    obs_range = np.arange(st_iter - 1, end_iter, 1)
    obs_range = obs_range[:, np.newaxis]

    smallb = parm[0:nvarma_rum]
    smallc = parm[nvarma_rum:nvarma_rum + nvarma_chq_actual]

    if(RUM_Model_basic == 1):
        smallc = smallc

    if(Choquet_Model == 1):
        if (Parametrized == 1):
            smallc = smallc[:,np.newaxis]
            mu_value = Fuzzy(smallc)

            df_mu_temp = pd.DataFrame(mu_value, index=(mu_ordering), columns=['mu'])
        else:
            df_mu_temp = pd.DataFrame(smallc, index=(mu_ordering), columns=['mu'])
    
    if(RUM_Constraint == 1):
        if (Parametrized == 1):
            smallc = smallc[:,np.newaxis]
            smallcf = Fuzzy(smallc)                  
        else:
            smallcf = smallc

    if(Probit == 1):
        temp = parm[nvarma_rum + nvarma_chq_actual:nvarma_rum + nvarma_chq_actual + nCholErr]
        if (Parametrized == 1):
            cholPsi = (upmat(xpnd(temp)))
            Psi1 = multi_dot([cholPsi.T, cholPsi])
        else:
            Psi1 = xpnd(temp)
        del temp
        Psi = np.zeros((nc,nc))
        Psi[1:nc,1:nc] = Psi1

    if(Non_comp == 1):
        att_cutoff_all = parm[nvarma_rum + nvarma_chq_actual + nCholErr:parm.shape[0]]
        att_cutoff = np.zeros((nvarma_chq,4))
        for i in range(0,nvarma_chq,1):
            if(var_cutoff_type[0,i]>0):
                if(i==0):
                    curr_cutoff_values = att_cutoff_all[0:var_cutoff_cumsum[i]]
                    curr_cutoff_values = curr_cutoff_values.T
                    att_cutoff[i,0:var_cutoff_type[0,i]] = curr_cutoff_values[0,:]
                else:
                    curr_cutoff_values = att_cutoff_all[var_cutoff_cumsum[i-1]:var_cutoff_cumsum[i]]
                    curr_cutoff_values = curr_cutoff_values.T
                    att_cutoff[i,0:var_cutoff_type[0,i]] = curr_cutoff_values[0,:]
        if (Parametrized == 1):
            att_cutoff = np.exp(att_cutoff)
            att_cutoff = np.cumsum(att_cutoff,1)
        
    if(Probit == 1):
        iden_matrix = np.eye(nc - 1)
        one_negative = -1 * np.ones((nc - 1, 1))
        seednext = MACMLS[0, iter]

    v1 = (np.kron(np.ones((nc, 1)), smallb)) * (Main_data.loc[st_iter - 1:end_iter - 1, ivgenva_rum].as_matrix().T)
    Utility_rum = np.empty(shape=(num_obs, nc), dtype=float)
    for i in range(0, nc, 1):
        j = i + 1
        Utility_rum[:, i] = np.sum(v1[(j - 1) * nvarma_rum:(j * nvarma_rum), :], axis=0)
    del v1

    data_chq = Main_data.loc[st_iter - 1:end_iter - 1, ivgenva_chq].as_matrix()
    count = 0
    for i in range(0, nvarma_chq, 1):
        for j in range(0, nc, 1):
            curr_arr = data_chq[:, nvarma_chq * j + i]
            curr_arr = curr_arr[:, np.newaxis]
            if (count == 0):
                X_chq = curr_arr
            else:
                X_chq = np.hstack((X_chq, curr_arr))
            count += 1

    if(Non_comp == 1):
        for i in range(0, nvarma_chq, 1):
            j = i + 1
            curr_value = X_chq[:, (j - 1) * nc:(j * nc)]
            curr_cutoffs = att_cutoff[i,:]
            temp = np.reshape(curr_value,(num_obs*nc))
            if(var_cutoff_type[0,i]==2):
                if (var_ind_chq[0, i] == 0):

                    y = np.zeros(temp.shape[0])

                    idx = np.where((temp <= curr_cutoffs[0]))[0]
                    y[idx] = np.ones(len(idx))

                    idx = np.where((curr_cutoffs[0] < temp) & (temp <= curr_cutoffs[1]))[0]
                    y[idx] = (curr_cutoffs[1]-temp[idx])/(curr_cutoffs[1]-curr_cutoffs[0])

                    idx = np.where((temp > curr_cutoffs[1]))[0]
                    y[idx] = np.zeros(len(idx))

                    yf = np.reshape(y,(num_obs,nc))
                
                elif (var_ind_chq[0, i] == 1):
                    y = np.zeros(temp.shape[0])

                    idx = np.where((temp <= curr_cutoffs[0]))[0]
                    y[idx] = np.zeros(len(idx))

                    idx = np.where((curr_cutoffs[0] < temp) & (temp <= curr_cutoffs[1]))[0]
                    y[idx] = (temp[idx]-curr_cutoffs[0])/(curr_cutoffs[1]-curr_cutoffs[0])

                    idx = np.where((temp > curr_cutoffs[1]))[0]
                    y[idx] = np.ones(len(idx))

                    yf = np.reshape(y,(num_obs,nc))
                
                if (i == 0):
                    X_chq_nom = yf
                else:
                    X_chq_nom = np.hstack((X_chq_nom, yf))
            elif(var_cutoff_type[0,i]==3):
                y = np.zeros(temp.shape[0])

                idx = np.where((temp <= curr_cutoffs[0]))[0]
                y[idx] = np.zeros(len(idx))

                idx = np.where((curr_cutoffs[0] < temp) & (temp <= curr_cutoffs[1]))[0]
                y[idx] = (temp[idx]-curr_cutoffs[0])/(curr_cutoffs[1]-curr_cutoffs[0])

                idx = np.where((curr_cutoffs[1] < temp) & (temp <= curr_cutoffs[2]))[0]
                y[idx] = (curr_cutoffs[2]-temp[idx])/(curr_cutoffs[2]-curr_cutoffs[1])

                idx = np.where((temp > curr_cutoffs[2]))[0]
                y[idx] = np.zeros(len(idx))

                yf = np.reshape(y,(num_obs,nc))
                             
                if (i == 0):
                    X_chq_nom = yf
                else:
                    X_chq_nom = np.hstack((X_chq_nom, yf))
            elif(var_cutoff_type[0,i]==4):
                y = np.zeros(temp.shape[0])

                idx = np.where((temp <= curr_cutoffs[0]))[0]
                y[idx] = np.zeros(len(idx))

                idx = np.where((curr_cutoffs[0] < temp) & (temp <= curr_cutoffs[1]))[0]
                y[idx] = (temp[idx]-curr_cutoffs[0])/(curr_cutoffs[1]-curr_cutoffs[0])

                idx = np.where((curr_cutoffs[1] < temp) & (temp <= curr_cutoffs[2]))[0]
                y[idx] = np.ones(len(idx))

                idx = np.where((curr_cutoffs[2] < temp) & (temp <= curr_cutoffs[3]))[0]
                y[idx] = (curr_cutoffs[3]-temp[idx])/(curr_cutoffs[3]-curr_cutoffs[2])

                idx = np.where((temp > curr_cutoffs[3]))[0]
                y[idx] = np.zeros(len(idx))

                yf = np.reshape(y,(num_obs,nc))
                             
                if (i == 0):
                    X_chq_nom = yf
                else:
                    X_chq_nom = np.hstack((X_chq_nom, yf))
            elif(var_cutoff_type[0,i]==0):
                if (var_ind_chq[0, i] == 1):
                    value_nom = np.apply_along_axis(Normalize_positive, 1, curr_value)
                elif (var_ind_chq[0, i] == 0):
                    value_nom = np.apply_along_axis(Normalize_negative, 1, curr_value)

                if (i == 0):
                    X_chq_nom = value_nom
                else:
                    X_chq_nom = np.hstack((X_chq_nom, value_nom))

    else:
        for i in range(0, nvarma_chq, 1):
            j = i + 1
            curr_value = X_chq[:, (j - 1) * nc:(j * nc)]
            if(RUM_Model_basic == 0):
                if (var_ind_chq[0, i] == 1):
                    value_nom = np.apply_along_axis(Normalize_positive, 1, curr_value)
                elif (var_ind_chq[0, i] == 0):
                    value_nom = np.apply_along_axis(Normalize_negative, 1, curr_value)
            if(RUM_Model_basic == 1):
                value_nom = curr_value

            if (i == 0):
                X_chq_nom = value_nom
            else:
                X_chq_nom = np.hstack((X_chq_nom, value_nom))

    X_chq_nom_final = np.zeros((num_obs, nc * nvarma_chq))
    count = -1
    for i in range(0, nc, 1):
        for j in range(0, nvarma_chq, 1):
            k = j + 1
            count += 1
            X_chq_nom_final[:, count] = X_chq_nom[:, (k - 1) * nc + i]

    if((RUM_Model == 1) | (RUM_Constraint == 1)):
        for i in range(0,nc,1):
            j = i+1
            curr_exp_values = X_chq_nom_final[:,(j - 1) * nvarma_chq:(j * nvarma_chq)]
            all_poss_comb = all_comb_xvalues(curr_exp_values)
            if(i == 0):
                New_Xvalue = all_poss_comb
            else:
                New_Xvalue = np.hstack((New_Xvalue,all_poss_comb))

        if(RUM_Model == 1): v1 = (np.kron(np.ones((nc, 1)), smallc)) * (New_Xvalue.T)
        if(RUM_Constraint == 1): v1 = (np.kron(np.ones((nc, 1)), smallcf)) * (New_Xvalue.T)
        Utility_chq = np.empty(shape=(num_obs, nc), dtype=float)
        j = 0
        for i in range(nc):
            j = i + 1
            Utility_chq[:, i] = np.sum(v1[(j - 1) * nvarma_chq_actual:(j * nvarma_chq_actual), :], axis=0)
        del v1
    
    elif(Choquet_Model == 1):
        Utility_chq = np.zeros((num_obs, nc))
        for i in range(0, num_obs, 1):
            curr_x_values = X_chq_nom_final[i, :]
            curr_x_values = np.reshape(curr_x_values, (nc, nvarma_chq))
            for ialt in range(0, nc, 1):
                A = np.zeros((nvarma_chq, 2))
                for ivar in range(0, nvarma_chq, 1):
                    A[ivar, 0] = ivar + 1
                    A[ivar, 1] = curr_x_values[ialt, ivar]

                B = A[np.argsort(A[:, 1])[::-1]]

                mu_label = []
                for k in range(0, nvarma_chq, 1):
                    if (k == 0):
                        mu_label.append(str(int(B[k, 0])))
                    else:
                        ele_store = []
                        for m in range(0, k + 1, 1):
                            ele_store.append(int(B[m, 0]))

                        ele_store = sorted(ele_store)
                        comb = ''
                        for m in range(0, len(ele_store), 1):
                            comb += str(ele_store[m])
                        mu_label.append(comb)

                chq_value = 0
                for m in range(0, nvarma_chq, 1):
                    if (i == 0):
                        curr_index2 = mu_label[m]
                        chq_value += B[m, 1] * df_mu_temp.loc[int(curr_index2), 'mu']
                    else:
                        curr_index2 = mu_label[m]
                        curr_index1 = mu_label[m - 1]
                        chq_value += B[m, 1] * (
                                    df_mu_temp.loc[int(curr_index2), 'mu'] - df_mu_temp.loc[int(curr_index1), 'mu'])

                Utility_chq[i, ialt] = chq_value
    elif(RUM_Model_basic == 1):
        v1 = (np.kron(np.ones((nc, 1)), smallc)) * (X_chq_nom_final.T)
        Utility_chq = np.empty(shape=(num_obs, nc), dtype=float)
        j = 0
        for i in range(nc):
            j = i + 1
            Utility_chq[:, i] = np.sum(v1[(j - 1) * nvarma_chq_actual:(j * nvarma_chq_actual), :], axis=0)
        del v1

    Utility = Utility_rum + Utility_chq

    All_Chosen = Main_data.loc[st_iter - 1:end_iter - 1, altchm].as_matrix()
    All_Chosen_logit = Main_data.loc[st_iter - 1:end_iter - 1, altchm_logit].as_matrix()

    obs_count = -1
    pair_count = -1
    Likelihood = np.zeros((num_obs, 1))

    for i in range(0, num_obs, 1):
        obs_count = obs_count + 1
        UY = Utility[i, :]
        UY = UY[:, np.newaxis]

        if(Probit == 1):
        
            Full_error = Psi
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

            M = temp
            del temp

            B_Tild = multi_dot([M, UY])
            Error_Tild = multi_dot([M, Full_error, M.T])

            mean_gu = -B_Tild
            var_gu = Error_Tild

            om = np.diag(var_gu)
            om = om[:, np.newaxis]
            om = om ** 0.5
            mean_gu_final = np.divide(mean_gu.T, om.T)
            var_gu_final = corrvc(var_gu)

            if (nc > 3):
                seed20 = seednext
                p2, sss = cdfmvnGHK(mean_gu_final, var_gu_final, seed20)
                seednext = sss
            elif (nc == 3):
                p2 = cdfbvn(mean_gu_final[0, 0], mean_gu_final[0, 1], var_gu_final[0, 1])
            else:
                p2 = cdfn(mean_gu_final)

            if (p2 < upper_limit):
                p2 = upper_limit

            LL = np.log(p2)
            Likelihood[obs_count, 0] = LL
        
        if(Logit == 1):
            p1 = np.exp(UY)
            p2 = np.sum(p1)
            p2 = p2*np.ones((nc,1))
            p3 = p1/p2
            Alt_chosenl = All_Chosen_logit[i, :]
            Alt_chosenl = Alt_chosenl[:,np.newaxis]
            p4 = p3*Alt_chosenl
            p4 = np.sum(p4)
            if (p4 < upper_limit):
                p4 = upper_limit

            LL = np.log(p4)
            Likelihood[obs_count, 0] = LL

    return (Likelihood)


# Numerical Gradient procedure
def lgd_NM(parm_estimated, idx_estimated, idx_fixed, parm_fixed):
    eps = np.sqrt(np.finfo(np.float).eps)
    grad = ndd.approx_fprime(parm_estimated, lpr, eps, args=(idx_estimated, idx_fixed, parm_fixed), centered=False)
    return (grad)


# Procedure to construct full beta vector from various fixed and active parameter sub-vectors
def reconstruct(parm_estimated, idx_estimated, idx_fixed, parm_fixed):
    total_parm = parm_estimated.shape[0] + parm_fixed.shape[0]
    beta_reconstructed = np.empty(shape=(total_parm, 1), dtype=float)

    beta_reconstructed[idx_estimated] = parm_estimated
    beta_reconstructed[idx_fixed] = parm_fixed
    return (beta_reconstructed)

def constraint1(parm_estimated, idx_estimated, idx_fixed, parm_fixed):
    All_Element = []
    for inum in range(1, nvarma_chq + 1, 1):
        All_Element.append(inum)

    Num_2pair = nvarma_chq * (nvarma_chq - 1) * 0.5
    Num_2pair = int(Num_2pair)

    parm = reconstruct(parm_estimated, idx_estimated, idx_fixed, parm_fixed)
    smallc = parm[nvarma_rum:nvarma_rum + nvarma_chq_actual]
    df_mb_temp = pd.DataFrame(smallc, index=(mu_ordering), columns=['mb'])

    g = np.zeros((nvarma_chq))
    for ir in range(1, nvarma_chq+1, 1):
        g[ir-1] = df_mb_temp.loc[int(ir), 'mb']
    
    for iorder in range(1, nvarma_chq, 1):
        for ir in range(0, nvarma_chq, 1):
            i = ir + 1
            A = []
            A.append(i)

            A_K = deepcopy(All_Element)
            A_K.remove(i)

            curr_order_set = rSubset(A_K, iorder)
            for kr in range(0, len(curr_order_set), 1):
                temp = list(curr_order_set[kr])
                all_comb = combs(temp)
                total_sum = 0
                for jr in range(0, len(all_comb), 1):
                    curr_ele = all_comb[jr]
                    curr_size = len(curr_ele)
                    ele_store = []
                    for istore in range(0, curr_size, 1):
                        ele_store.append(int(curr_ele[istore]))

                    ele_store.append(int(i))
                    ele_store1f = sorted(ele_store)

                    curr_index1 = ''
                    for jr2 in range(0, len(ele_store1f), 1):
                        curr_index1 += str(ele_store1f[jr2])

                    total_sum += (df_mb_temp.loc[int(curr_index1), 'mb'])

                g = np.append(g,total_sum)

    return g

def constraint2(parm_estimated, idx_estimated, idx_fixed, parm_fixed):
    sum_eq = 1.0
    parm = reconstruct(parm_estimated, idx_estimated, idx_fixed, parm_fixed)
    smallc = parm[nvarma_rum:nvarma_rum + nvarma_chq_actual]
    for i in range(0,smallc.shape[0],1):
        sum_eq = sum_eq - smallc[i]
    return sum_eq


'''
-------------------------------Simulation Settings------------------------------------------------------------------------------------
''' 
global nchocc, upper_limit, nc, nind, nobs, seed, seed1, Non_IID, Num_Threads, Parametrized, D_matrix, _ranper, GHK, Data_Split, Approx_GHK, Est_Normal, Non_comp,Logit,Probit,IID_Err,Err_Full
output_path = "/rds/general/user/pb313/home/"

Choquet_Model   = 1  # Make this 1 to estimate a Choquet Integral based Model
RUM_Model       = 0  # Make this 1 to estimate a weighted sum Model with all interactions
RUM_Constraint  = 0  # Make this 1 to estimate a weighted sum Model with monotonicity constraints
Non_comp        = 1  # Make this 1 to estimate a non-compensatory model incorporating attribute cut-off
RUM_Model_basic = 0  # Make this 1 to estimate a weighted sum Model with no interactions

Logit  = 0     # Make this 1 to estimate a Logit (Extreme value distribution) Kernel based Model
Probit = 1     # Make this 1 to estimate a Probit (Normal distribution) Kernel based Model

IID_Err  = 0   # Make this 1 to estimate a IID error structure in the Probit kernel
Err_Full = 0   # Make this 1 to estimate a full-covariance error structure in the Probit kernel

nc   = 5       # Number of alternatives
nind = 3000    # Number of individuals in the sample
nobs = nind    
Num_Threads = 64  # Set the number of threads here for multithreading

Num_Choquet_Config = 6  # Number of variables in the Choquet Integral configuration 4 or 6 for the current simulation configuration
Orignal_parm       = 0  # Make this 1, if want to pass true parameter values as starting value

Output_file_name = 'Results.xlsx'   # Name of the output file where all the results are written
'''  Output file Format:
sheet_name='Param_Estimates'     : Contains all the parameter estimates
sheet_name='Standard_Error'      : Contains all the parameter's standard errros
sheet_name='Estimation_time'     : Sample estimation time in mins
sheet_name='Likelihood_estimates : Log-likelihood value at convergence
sheet_name='Interaction_Value'   : Shapley value and Interaction indicies if the model structure is Choquet
'''

st_sample = 1   # Starting sample number
Total_sam = 50  # Total number of samples. These many samples will be generated and estimated. The results for all the samples are written in the Output_file_name 


if(Choquet_Model == 1):
    print('Model Structure : Choquet Integral')
elif(RUM_Model == 1):
    print('Model Structure : Random utility unconstrained with all interactions')
elif(RUM_Constraint == 1):
    print('Model Structure : Random utility constrained with all interactions')
elif(RUM_Model_basic == 1):
    print('Model Structure : Random utility basic')

if(Non_comp == 1):
    print('Behaviour : Non-Compensatory')
else:
    print('Behaviour : Compensatory')

if(IID_Err == 1):
    print('Error Structure : IID')
else:
    if(Err_Full == 0):
        print('Error Structure : Diagonal')
    else:
        print('Error Structure : Full-Covariance')

upper_limit   = 1e-05  # Any value of CDF below this limit is considered as zero

ncol = nc
_halt_numdraws = 200
nrep = _halt_numdraws
nrephalt = nrep
allHaltDraws = HaltonSequence(nobs * (nrep + 10), ncol)
_halt_maxdraws = allHaltDraws.shape[0]

global MACMLS
MACMLS = np.array([[300000, 300001, 300002, 300003, 300004, 300005, 300006, 300007, 300008, 300009, 300010, 300011,
                    300012, 300013, 300014, 300015, 300016, 300017, 300018, 300019, 300020, 300021, 300022, 300023,
                    300024, 300025, 300026, 300027, 300028, 300029, 300030, 300031, 300032, 300033, 300034, 300035,
                    300036, 300037, 300038, 300039, 300040, 300041, 300042, 300043, 300044, 300045, 300046, 300047,
                    300048, 300049, 300050, 300051, 300052, 300053, 300054, 300055, 300056, 300057, 300058, 300059,
                    300060, 300061, 300062, 300063, 300064, 300065, 300066, 300067, 300068, 300069, 300070, 300071]])

run_no = st_sample

dataframe_col = []
global All_Parameters, All_Jacobian, All_Times, All_LL,All_Interaction_Index

sample_sucess = 0
sample_count = 0
while ((sample_sucess < Total_sam)):
    seed = 623854 + run_no - 1  # Seed for the error term
    seed1 = 358696  # The seed for the data variable generation
    seed4 = 400 + run_no + 2

    sample_count = sample_count + 1
    run_no = run_no + sample_count

    vnames = ['TT1', 'TT2', 'TT3', 'TT4', 'TT5', 
              'TC1', 'TC2', 'TC3', 'TC4', 'TC5', 
              'WT1', 'WT2', 'WT3', 'WT4', 'WT5',
              'ST1', 'ST2', 'ST3', 'ST4', 'ST5',
              'TF1', 'TF2', 'TF3', 'TF4', 'TF5',
              'TG1', 'TG2', 'TG3', 'TG4', 'TG5']

    ntot = len(vnames)
    np.random.seed(seed1)
    TT_all = np.random.uniform(low=1, high=10, size=(nobs, nc))
    np.random.seed(seed1 + 1)
    TC_all = np.random.uniform(low=1, high=10, size=(nobs, nc))
    np.random.seed(seed1 + 2)
    WT_all = np.random.uniform(low=1, high=10, size=(nobs, nc))
    np.random.seed(seed1 + 3)
    ST_all = np.random.uniform(low=1, high=10, size=(nobs, nc))
    np.random.seed(seed1 + 4)
    TF_all = np.random.uniform(low=1, high=10, size=(nobs, nc))
    np.random.seed(seed1 + 5)
    TG_all = np.random.uniform(low=1, high=10, size=(nobs, nc))

    X_Var = np.hstack((TT_all, TC_all, WT_all, ST_all,TF_all,TG_all))

    
    '''
    X_Var1 = np.hstack((np.reshape(TT_all,(nobs*nc,1)), np.reshape(TC_all,(nobs*nc,1)), np.reshape(WT_all,(nobs*nc,1)), np.reshape(ST_all,(nobs*nc,1))))

    print("30th Percentile of arr: ", np.percentile(X_Var1, 30, axis=0))
    print("60th Percentile of arr: ", np.percentile(X_Var1, 60, axis=0))
    print("90th Percentile of arr: ", np.percentile(X_Var1, 90, axis=0))
    sys.exit()
    '''

    invar2 = ['Chosen', 'Mode_1', 'Mode_2', 'Mode_3', 'Mode_4', 'Mode_5', 'PID', 'uno', 'sero']
    outvar = vnames + invar2
    outdata = np.hstack((X_Var, np.zeros((nobs, len(invar2) - 2)), np.ones((nobs, 1)), np.zeros((nobs, 1))))
    global Main_data
    Main_data = pd.DataFrame(outdata, index=range(0, nobs, 1), columns=outvar)
    del TT_all, TC_all, WT_all, ST_all, X_Var, outdata
    global nCholErr, ncholomega, mix_ele, mu_ordering, var_ind_chq

    # Utility expression for variables with weighted combination
    Mode_1_0 = ['sero', 'sero', 'sero', 'sero']
    Mode_2_0 = ['uno',  'sero', 'sero', 'sero']
    Mode_3_0 = ['sero', 'uno',  'sero', 'sero']
    Mode_4_0 = ['sero', 'sero', 'uno',  'sero']
    Mode_5_0 = ['sero', 'sero', 'sero', 'uno']

    if(Num_Choquet_Config == 6):
        Mode_1_1 = ['TT1', 'TC1', 'WT1', 'ST1', 'TF1', 'TG1']
        Mode_2_1 = ['TT2', 'TC2', 'WT2', 'ST2', 'TF2', 'TG2']
        Mode_3_1 = ['TT3', 'TC3', 'WT3', 'ST3', 'TF3', 'TG3']
        Mode_4_1 = ['TT4', 'TC4', 'WT4', 'ST4', 'TF4', 'TG4']
        Mode_5_1 = ['TT5', 'TC5', 'WT5', 'ST5', 'TF5', 'TG5']
    elif(Num_Choquet_Config == 4):
        Mode_1_1 = ['TT1', 'TC1', 'WT1', 'ST1']
        Mode_2_1 = ['TT2', 'TC2', 'WT2', 'ST2']
        Mode_3_1 = ['TT3', 'TC3', 'WT3', 'ST3']
        Mode_4_1 = ['TT4', 'TC4', 'WT4', 'ST4']
        Mode_5_1 = ['TT5', 'TC5', 'WT5', 'ST5']

    global ivgenva_rum, nvarma_rum, ivgenva_chq, nvarma_chq, var_cutoff_type,var_cutoff_cumsum
    ivgenva_rum = Mode_1_0 + Mode_2_0 + Mode_3_0 + Mode_4_0 + Mode_5_0
    nvarma_rum = len(Mode_1_0)

    ivgenva_chq = Mode_1_1 + Mode_2_1 + Mode_3_1 + Mode_4_1 + Mode_5_1
    nvarma_chq = len(Mode_1_1)
    nvarma_chq_actual = int(pow(2, nvarma_chq) - 1)

    if(RUM_Model_basic == 1):
        nvarma_chq_actual = nvarma_chq

    if(RUM_Model_basic == 0):
        mu_ordering = all_comb(nvarma_chq)
        mu_ordering = [int(i) for i in mu_ordering]

    var_ind_chq = np.array([[0, 1, 0, 1, 0, 1]])
    var_ind_chq = var_ind_chq[:,0:nvarma_chq]
    var_cutoff_type = np.array([[2,2,4,4, 2, 4]]) # 0: No cutoff, 2: 2 point cutoff (1 or 0 followed by a slope), 3 and 4 
    var_cutoff_type = var_cutoff_type[:,0:nvarma_chq]
    var_cutoff_cumsum = np.cumsum(var_cutoff_type)
    total_cutoff = np.sum(var_cutoff_type)

    # *****************************************************************************
    #                 True Value of Parameters
    # *****************************************************************************
    dgp_beta_rum = np.vstack( (-0.70, -0.60, -0.50, -0.40))  # Exogeneous variables coefficients for DC variable in RUM framework

    # ['1', '2', '3', '4', '12', '13', '14', '23', '24', '34', '123', '124', '134', '234', '1234'] Choquet fuzzy measure ordering
    if(Num_Choquet_Config == 4):
        dgp_beta_chq = np.vstack((0.3, 0.25, 0.2, 0.1, 0.58, 0.53, 0.44, 0.49, 0.36, 0.33, 0.79, 0.68, 0.64, 0.59, 1.00))
    elif(Num_Choquet_Config == 6):
        dgp_beta_chq = np.vstack(([0.17, 0.18, 0.20, 0.16, 0.19, 0.18, 0.33, 0.35, 0.31, 0.34, 0.33, 0.36, 0.32, 0.35,
                                0.34, 0.34, 0.37, 0.36, 0.33, 0.32, 0.35, 0.51, 0.47, 0.50, 0.49, 0.49, 0.52, 0.51,
                                0.48, 0.47, 0.50, 0.50, 0.53, 0.52, 0.49, 0.48, 0.51, 0.51, 0.50, 0.53, 0.49, 0.65,
                                0.68, 0.67, 0.64, 0.63, 0.66, 0.66, 0.65, 0.68, 0.64, 0.67, 0.66, 0.69, 0.65, 0.67,
                                0.82, 0.81, 0.84, 0.80, 0.82, 0.83, 1.00]))

    if(RUM_Model_basic == 1):
        dgp_beta_chq = dgp_beta_chq[0:nvarma_chq_actual]

    if(RUM_Model_basic == 0):
        dgp_beta_chq_mb = Mobius(dgp_beta_chq)

    if(IID_Err == 0):
        dgp_Psi = np.array([[1.00,
                             0.50, 1.10,
                             0.50, 0.50, 1.20,
                             0.50, 0.50, 0.50, 1.30]]).T  # Covariance matrix for the differenced error term vector
        dgp_Psi = xpnd(dgp_Psi)
    else:
        dgp_Psi = 0.5*np.eye(nc-1) + 0.5*np.ones((nc-1,nc-1))


    
    row_psi = dgp_Psi.shape[0]
    nCholErr = int((nc) * (nc-1) * 0.5)

    if(IID_Err == 0):
        if(Err_Full == 0):
            psi_active = np.array([[0,
                                    0, 1,
                                    0, 0, 1,
                                    0, 0, 0, 1]]).T
        else:
            psi_active = np.array([[0,
                                    1, 1,
                                    1, 1, 1,
                                    1, 1, 1, 1]]).T

    else:
        psi_active = np.zeros((nCholErr,1))
    

    if(Logit == 1):
        nCholErr = 0


    if(Non_comp == 1):
        dgp_beta_cutoff = np.vstack((3.0, 7.0,
                                     3.5, 6.5,
                                     2.0, 4.0, 6.0, 7.0,
                                     3.5, 5.5, 7.5, 8.5,
                                     3.3, 6.8,
                                     2.5, 5.0, 6.5, 7.5))
        dgp_beta_cutoff = dgp_beta_cutoff[0:total_cutoff]
        cutoff_num = dgp_beta_cutoff.shape[0]

    # ****************************************************************************************************************************************************************************************
    #                 Packing of all parameters in a single vector ( Do not change anything below this line)
    # *****************************************************************************************************************************************************************************************
    dgp_X = dgp_beta_rum
    dgp_X = np.vstack((dgp_X, dgp_beta_chq))
    if(Probit == 1):
        dgp_X = np.vstack((dgp_X, vech(dgp_Psi)))
    if(Non_comp == 1):
        dgp_X = np.vstack((dgp_X, dgp_beta_cutoff)) 

    global req_col, altchm, continuous_var
    req_col = ['Mode_1', 'Mode_2', 'Mode_3', 'Mode_4', 'Mode_5']
    altchm = ['Chosen']
    altchm_logit = ['Mode_1', 'Mode_2', 'Mode_3', 'Mode_4', 'Mode_5']
    Chosen_Alt = dloopN(dgp_X, seed)

    Share_data = Main_data.loc[:, req_col].as_matrix()
    Share = np.sum(Share_data, axis=0)
    Share = Share / nobs
    ID_data = Main_data.loc[:, vnames].as_matrix()
    ID_Share = np.sum(ID_data, axis=0)
    ID_Share = ID_Share / nobs
    
    print("Sample Number: ", sample_count)
    print('-------------------------------------------')
    print("Alternative Share :")
    for i in range(0, nc, 1):
        print('{0:20s}'.format(req_col[i]).ljust(19) + '{0:1.2f}'.format(Share[i]).ljust(10))
    # print("ID Share :")
    # for i in range(0,len(vnames),1):
    #   print('{0:10s}'.format(vnames[i]).ljust(19) + '{0:1.2f}'.format(ID_Share[i]).ljust(10))
    print('-------------------------------------------')
    
    if (Orignal_parm == 1):
        dgp_bd1 = dgp_beta_rum
        dgp_bcont1 = dgp_beta_chq_mb
        dgp_Psi1 = dgp_Psi
        if(Non_comp == 1):
            dgp_cutoff1 = np.log((1/dgp_beta_cutoff)-1)
    
    else:
        dgp_bd1 = 0.1 * np.ones((dgp_beta_rum.shape[0], 1))
        dgp_bcont1 = 0.1 * np.ones((nvarma_chq_actual, 1))
        dgp_Psi1 = 0.5*np.eye(dgp_Psi.shape[0]) + 0.5*np.ones((nc-1,nc-1))
        if(Non_comp == 1):
            cutoff_start = 0.5*np.ones((cutoff_num,1))
            dgp_cutoff1 = cutoff_start
    
    dgp_X1 = dgp_bd1
    dgp_X1 = np.vstack((dgp_X1, dgp_bcont1))
    if(Probit == 1):
        dgp_X1 = np.vstack((dgp_X1, vech(dgp_Psi1)))
    if(Non_comp == 1):
        dgp_X1 = np.vstack((dgp_X1, dgp_cutoff1))

    
    bb = dgp_X1[0:nvarma_rum + nvarma_chq_actual]
    if(Probit == 1):
        bb = np.vstack((bb, vech(chol(xpnd(dgp_X1[nvarma_rum + nvarma_chq_actual:nvarma_rum + nvarma_chq_actual + nCholErr])))))
    if(Non_comp == 1):
        bb = np.vstack((bb, dgp_X1[nvarma_rum + nvarma_chq_actual + nCholErr:dgp_X1.shape[0]]))
    
    max_active = np.ones((nvarma_rum + nvarma_chq_actual, 1))
    if(Probit == 1):
        max_active = np.vstack((max_active, psi_active))
    if(Non_comp == 1):
        max_active = np.vstack((max_active, np.ones((cutoff_num, 1))))
    
    idx_estimated = np.where(max_active == 1)  # Index of estimated variables in beta vector
    idx_fixed = np.where(max_active == 0)  # Index of fixed variables in beta vector
    
    idx_estimatedt = idx_estimated[0]
    idx_estimatedt = idx_estimatedt[:, np.newaxis]
    
    # Split the starting value vector into parameters that are estimated and fixed
    init_estimated = bb[idx_estimated]
    init_fixed = bb[idx_fixed]
    
    # Defining variable labels
    
    Param_nam_mb = []
    for i in range(0, nvarma_rum, 1):
        if (i < 9):
            Param_nam_mb.append('ASC0' + str(i + 1))
        else:
            Param_nam_mb.append('ASC' + str(i + 1))
    
    for i in range(0, nvarma_chq_actual, 1):
        if((Choquet_Model == 1) | (RUM_Constraint == 1)):
            Param_nam_mb.append('MB_' + str(mu_ordering[i]))
        elif(RUM_Model == 1):
            Param_nam_mb.append('Beta_' + str(mu_ordering[i]))
        elif(RUM_Model_basic == 1):
            Param_nam_mb.append('Beta_' + str(i+1))

    
    if(Probit == 1):
        Param_nam_mb.append('Psi11')
        for i in range(1, dgp_Psi.shape[0], 1):
            for j in range(0, i + 1, 1):
                Param_nam_mb.append('Psi' + str(i + 1) + str(j + 1))
    
    if(Non_comp == 1):
        for i in range(0, nvarma_chq, 1):
            if(var_cutoff_type[0,i]>0):
                for j in range(0,var_cutoff_type[0,i],1):
                    Param_nam_mb.append('Thr_' + str(i + 1) + '_'+ str(j + 1))


    Active_name_mb = []
    for i in range(0, int(np.sum(max_active, axis=0)[0]), 1):
        Active_name_mb.append(Param_nam_mb[idx_estimatedt[i, 0]])
    
    Param_nam_mu = []
    for i in range(0, nvarma_rum, 1):
        if (i < 9):
            Param_nam_mu.append('ASC0' + str(i + 1))
        else:
            Param_nam_mu.append('ASC' + str(i + 1))
    
    for i in range(0, nvarma_chq_actual, 1):
        if(Choquet_Model == 1):
            Param_nam_mu.append('MU_' + str(mu_ordering[i]))
        elif((RUM_Model == 1) | (RUM_Constraint == 1)):
            Param_nam_mu.append('Beta_' + str(mu_ordering[i]))
        elif(RUM_Model_basic == 1):
            Param_nam_mu.append('Beta_' + str(i+1))
    
    
    if(Probit == 1):
        Param_nam_mu.append('Psi11')
        for i in range(1, dgp_Psi.shape[0], 1):
            for j in range(0, i + 1, 1):
                Param_nam_mu.append('Psi' + str(i + 1) + str(j + 1))
    
    if(Non_comp == 1):
        for i in range(0, nvarma_chq, 1):
            if(var_cutoff_type[0,i]>0):
                for j in range(0,var_cutoff_type[0,i],1):
                    Param_nam_mu.append('Thr_' + str(i + 1) + '_'+ str(j + 1))

    Active_name_mu = []
    for i in range(0, int(np.sum(max_active, axis=0)[0]), 1):
        Active_name_mu.append(Param_nam_mu[idx_estimatedt[i, 0]])

    Data_Split = np.zeros((Num_Threads, 2))
    for i in range(1, Num_Threads + 1, 1):
        Data_Split[i - 1, 0] = int(ceil((i - 1) * (nind / Num_Threads)) + 1)
        Data_Split[i - 1, 1] = int(ceil((i) * (nind / Num_Threads)))
    
    max_gradTol = 1e-05
    max_iter = 150
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
                print('{0:10s}'.format(Active_name_mb[i]).ljust(19) + '{0:3.4f}'.format(Xi[i]).ljust(10))
            if (Xi[i] >= 0):
                print('{0:10s}'.format(Active_name_mb[i]).ljust(20) + '{0:3.4f}'.format(Xi[i]).ljust(10))
        print()
    
        Nfeval += 1
    

    print("Optimization has Started.............")
    start_time = time.time()
    if(Choquet_Model == 1):
        [xopt, fopt, num_iter, imode, smode] = fmin_slsqp(lpr, init_estimated, f_eqcons=constraint2, f_ieqcons=constraint1, fprime=lgd_NM,
                                                                                   args=(idx_estimated, idx_fixed, init_fixed),  iter=max_iter, acc=max_gradTol,
                                                                                   iprint=2, full_output=1, disp=1,callback=callbackF)
        print(smode)
    elif((RUM_Model == 1) | (RUM_Model_basic == 1)):
        [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg, allvecs] = fmin_bfgs(lpr, init_estimated, fprime=lgd_NM, 
                                                                                        args=(idx_estimated, idx_fixed, init_fixed), gtol=max_gradTol, 
                                                                                        epsilon=1.4901161193847656e-08, maxiter=max_iter, full_output=1, disp=1, 
                                                                                        retall=1, callback=callbackF)
    elif(RUM_Constraint == 1):
        [xopt, fopt, num_iter, imode, smode] = fmin_slsqp(lpr, init_estimated, f_eqcons=constraint2, f_ieqcons=constraint1, fprime=lgd_NM,
                                                                                   args=(idx_estimated, idx_fixed, init_fixed),  iter=max_iter, acc=max_gradTol,
                                                                                   iprint=2, full_output=1, disp=1,callback=callbackF)

    
    
    xComplete = bb
    xComplete[idx_estimated] = xopt
    xComplete[idx_fixed] = init_fixed

    if(Probit == 1):
        temp = xComplete[nvarma_rum + nvarma_chq_actual:nvarma_rum + nvarma_chq_actual + nCholErr]
        cholPsi = (upmat(xpnd(temp)))
        Psi = multi_dot([cholPsi.T, cholPsi])
        check1 = np.isfinite(cond(Psi))
        det1 = det(Psi)
        check2 = det1 > 0.01
        check3 = is_pos_def(Psi)
        check_error = check1 & check2 & check3
        check = check_error

        Psi = vech(Psi)
        xComplete[nvarma_rum + nvarma_chq_actual:nvarma_rum + nvarma_chq_actual + nCholErr, 0] = Psi[:, 0]
    if(Logit == 1):
        check = True

    if((Choquet_Model == 1) | (RUM_Constraint == 1)):
        mb_values = xComplete[nvarma_rum:nvarma_rum + nvarma_chq_actual]
        mb_values = mb_values[:,np.newaxis]
        mb_to_mu = Fuzzy(mb_values)
        xComplete[nvarma_rum:nvarma_rum + nvarma_chq_actual,0] = mb_to_mu[:, 0]
    if(Non_comp == 1):
        att_cutoff_all = xComplete[nvarma_rum + nvarma_chq_actual + nCholErr:xComplete.shape[0]]
        att_cutoff = np.zeros((nvarma_chq,4))
        for i in range(0,nvarma_chq,1):
            if(var_cutoff_type[0,i]>0):
                if(i==0):
                    curr_cutoff_values = att_cutoff_all[0:var_cutoff_cumsum[i]]
                    curr_cutoff_values = curr_cutoff_values.T
                    att_cutoff[i,0:var_cutoff_type[0,i]] = curr_cutoff_values[0,:]
                else:
                    curr_cutoff_values = att_cutoff_all[var_cutoff_cumsum[i-1]:var_cutoff_cumsum[i]]
                    curr_cutoff_values = curr_cutoff_values.T
                    att_cutoff[i,0:var_cutoff_type[0,i]] = curr_cutoff_values[0,:]
        att_cutoff = np.exp(att_cutoff)
        att_cutoff = np.cumsum(att_cutoff,1)

        count_check = 0
        for i in range(0,nvarma_chq,1):
            if(var_cutoff_type[0,i]>0):
                if(i==0):
                    curr_cutoff_values = att_cutoff[i,0:var_cutoff_type[0,i]]  
                    count_check += 1                  
                else:
                    curr_cutoff_values = att_cutoff[i,0:var_cutoff_type[0,i]]
                    count_check += 1
                if(count_check == 1):
                    All_final_cutoff = curr_cutoff_values[:,np.newaxis]
                else:
                    All_final_cutoff = np.vstack((All_final_cutoff,curr_cutoff_values[:,np.newaxis]))

        xComplete[nvarma_rum + nvarma_chq_actual + nCholErr:xComplete.shape[0],0] = All_final_cutoff[:,0]

    #print(xComplete)
    if (check):
        Parametrized = 0

        parm_estimated = xComplete[idx_estimated]
        parm_fixed = xComplete[idx_fixed]

        MNP_lpr = lpr(parm_estimated, idx_estimated, idx_fixed, parm_fixed)
        MNP_lgd = lgd_NM(parm_estimated, idx_estimated, idx_fixed, parm_fixed)

        # Hess_call = nd.Hessian(lpr)
        # MNP_Hess = Hess_call(parm_estimated, idx_estimated, idx_fixed, parm_fixed)
        # print(MNP_Hess.shape)
        # print(np.sum(max_active))

        IM_all = np.zeros((max_active.shape[0], 1))

        Jacobian = multi_dot([MNP_lgd.T, MNP_lgd])

        check1 = np.isfinite(cond(Jacobian))
        det1 = det(Jacobian)
        check2 = det1 > 0.01
        check3 = is_pos_def(Jacobian)
        check_error = check1 & check2 & check3
        check_Jac = check_error


        if (check_Jac):
            IM = np.sqrt(np.diag(np.linalg.inv(Jacobian)))  
            sample_sucess = sample_sucess + 1
                  

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
            print('{0:10s}'.format('Parm_name').ljust(19) + '{0:10s}'.format('Estimate').ljust(16) + '{0:10s}'.format('T-stat').ljust(14))
            for i in range(0, max_active.shape[0], 1):
                if (xComplete[i, 0] < 0 and max_active[i, 0] == 1):
                    if (abs(xComplete[i, 0]) >= 10):
                        print('{0:10s}'.format(Param_nam_mu[i]).ljust(19) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(14) + '{0:3.3f}'.format(xComplete[i, 0] / IM_all[i, 0]).ljust(14))
                    if (abs(xComplete[i, 0]) < 10):
                        print('{0:10s}'.format(Param_nam_mu[i]).ljust(19) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(15) + '{0:3.3f}'.format(xComplete[i, 0] / IM_all[i, 0]).ljust(14))
                if (xComplete[i, 0] > 0 and max_active[i, 0] == 1):
                    if (xComplete[i, 0] >= 10):
                        print('{0:10s}'.format(Param_nam_mu[i]).ljust(20) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(14) + '{0:3.3f}'.format(xComplete[i, 0] / IM_all[i, 0]).ljust(15))
                    if (xComplete[i, 0] < 10):
                        print('{0:10s}'.format(Param_nam_mu[i]).ljust(20) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(15) + '{0:3.3f}'.format(xComplete[i, 0] / IM_all[i, 0]).ljust(15))
                if (xComplete[i, 0] == 0 or max_active[i, 0] == 0):
                    print('{0:10s}'.format(Param_nam_mu[i]).ljust(20) + '{0:3.3f}'.format(xComplete[i, 0]).ljust(15) + '{0:3.3f}'.format(0.000).ljust(15))

            if(Choquet_Model == 1):
                mu_temp = xComplete[nvarma_rum:nvarma_rum + nvarma_chq_actual,0]
                Mu_dataframe = pd.DataFrame(mu_temp,index=mu_ordering,columns=['mu'])
                Shapley_Value = Shapley(Mu_dataframe,nvarma_chq)
                Interaction_Value = Interaction_Index(Mu_dataframe,nvarma_chq)

                for iparam in range(0,nvarma_chq,1):
                    Interaction_Value[iparam,iparam] = Shapley_Value[iparam,0]

                F_id = vech(Interaction_Value)

            if (sample_sucess <= 9):
                dataframe_col.append('Sample_0' + str(sample_sucess))
            else:
                dataframe_col.append('Sample_' + str(sample_sucess))

            if (sample_sucess == 1):
                All_Parameters = xComplete
                All_Jacobian = IM_all
                All_Times = total_time
                All_LL = Total_LL
                if(Choquet_Model == 1):
                    All_Interaction_Index = F_id
            else:
                All_Parameters = np.hstack((All_Parameters, xComplete))
                All_Jacobian = np.hstack((All_Jacobian, IM_all))
                All_Times = np.vstack((All_Times, total_time))
                All_LL = np.vstack((All_LL, Total_LL))
                if(Choquet_Model == 1):
                    All_Interaction_Index = np.hstack((All_Interaction_Index,F_id))


All_betas = pd.DataFrame(All_Parameters, index=range(0, All_Parameters.shape[0], 1), columns=dataframe_col)
All_error = pd.DataFrame(All_Jacobian, index=range(0, All_Jacobian.shape[0], 1), columns=dataframe_col)
All_times = pd.DataFrame(All_Times, index=range(0, All_Times.shape[0], 1), columns=['Time'])
All_LL = pd.DataFrame(All_LL, index=range(0, All_LL.shape[0], 1), columns=['Log-Likelihood'])
if(Choquet_Model == 1):
    All_Interaction_Index = pd.DataFrame(All_Interaction_Index, index=range(0, All_Interaction_Index.shape[0], 1), columns=dataframe_col)

if(Choquet_Model == 1):
    writer = pd.ExcelWriter(output_path+Output_file_name, engine='xlsxwriter')
    All_betas.to_excel(writer,sheet_name='Param_Estimates')
    All_error.to_excel(writer,sheet_name='Standard_Error')
    All_times.to_excel(writer,sheet_name='Estimation_time')
    All_LL.to_excel(writer,sheet_name='Likelihood_estimates')
    All_Interaction_Index.to_excel(writer,sheet_name='Interaction_Value')
    writer.save()
elif(RUM_Model == 1):
    writer = pd.ExcelWriter(output_path+Output_file_name, engine='xlsxwriter')
    All_betas.to_excel(writer,sheet_name='Param_Estimates')
    All_error.to_excel(writer,sheet_name='Standard_Error')
    All_times.to_excel(writer,sheet_name='Estimation_time')
    All_LL.to_excel(writer,sheet_name='Likelihood_estimates')
    writer.save()
elif(RUM_Constraint == 1):
    writer = pd.ExcelWriter(output_path+Output_file_name, engine='xlsxwriter')
    All_betas.to_excel(writer,sheet_name='Param_Estimates')
    All_error.to_excel(writer,sheet_name='Standard_Error')
    All_times.to_excel(writer,sheet_name='Estimation_time')
    All_LL.to_excel(writer,sheet_name='Likelihood_estimates')
    writer.save()
elif(RUM_Model_basic == 1):
    writer = pd.ExcelWriter(output_path+Output_file_name, engine='xlsxwriter')
    All_betas.to_excel(writer,sheet_name='Param_Estimates')
    All_error.to_excel(writer,sheet_name='Standard_Error')
    All_times.to_excel(writer,sheet_name='Estimation_time')
    All_LL.to_excel(writer,sheet_name='Likelihood_estimates')
    writer.save()



