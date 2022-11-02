# Paper Title
The code is for the paper titled _'**A multinomial probit model with Choquet integral and attribute cut-offs**'_. 
You can access the paper [**Here**](https://www.sciencedirect.com/science/article/pii/S0191261522000261). It is published under a open access license and hence can be downloaded for free. 

# Paper Abstract
_Several non-linear functions and machine learning methods have been developed for flexible specification of the systematic utility in discrete choice models. However, they lack interpretability, do not ensure monotonicity conditions, and restrict substitution patterns. We address the first two challenges by modeling the systematic utility using the Choquet Integral (CI) function and the last one by embedding CI into the multinomial probit (MNP) choice probability kernel. We also extend the MNP-CI model to account for attribute cut-offs that enable a modeler to approximately mimic the semi-compensatory behavior using the traditional choice experiment data. The MNP-CI model is estimated using a constrained maximum likelihood approach, and its statistical properties are validated through a comprehensive Monte Carlo study. The CI-based choice model is empirically advantageous as it captures interaction effects while maintaining monotonicity. It also provides information on the complementarity between pairs of attributes coupled with their importance ranking as a by-product of the estimation. These insights could potentially assist policymakers in making policies to improve the preference level for an alternative. These advantages of the MNP-CI model with attribute cut-offs are illustrated in an empirical application to understand New Yorkers’ preferences towards mobility-on-demand services._

# Code Content
The code provides user the oppurtunity to replicate the simulation study reported int he paper. The code is properly commented for a easy follow through. With a little effort, users can modify the code to include external dataset.
The file _Choquet_Simulation.py_ contains all the necessary functions and no additional file is required.

# Dependencies
```
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
import time
import sys
import operator
from functools import reduce
import itertools
from itertools import combinations
from scipy.special import factorial
from copy import deepcopy
import warnings
```

# Code settings
### Here is the list of variables that can be changed to set various model configuration 
```
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
```