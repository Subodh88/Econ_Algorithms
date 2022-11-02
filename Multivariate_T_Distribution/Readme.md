# Paper Title
The code is for the paper titled _**A generalized continuous-multinomial response model with a t-distributed error kernel**_. You can access the paper [**Here**](https://www.sciencedirect.com/science/article/abs/pii/S0191261519304205). This model is useful in situations where data is not normally distributed and exhibit a heavy tailed distribution.  

# Paper Abstract
_In multinomial response models, idiosyncratic variations in the indirect utility are generally modeled using Gumbel or normal distributions. This study makes a strong case to substitute these thin-tailed distributions with a t-distribution. First, we demonstrate that a model with a t-distributed error kernel better estimates and predicts preferences, especially in class-imbalanced datasets. Our proposed specification also implicitly accounts for decision-uncertainty behavior, i.e. the degree of certainty that decision-makers hold in their choices relative to the variation in the indirect utility of any alternative. Second – after applying a t-distributed error kernel in a multinomial response model for the first time – we extend this specification to a generalized continuous-multinomial (GCM) model and derive its full-information maximum likelihood estimation procedure. The likelihood involves an open-form expression of the cumulative density function of the multivariate t-distribution, which we propose to compute using a combination of the composite marginal likelihood method and the separation-of-variables approach. Third, we establish finite sample properties of the GCM model with a t-distributed error kernel (GCM-t) and highlight its superiority over the GCM model with a normally-distributed error kernel (GCM-N) in a Monte Carlo study. Finally, we compare GCM-t and GCM-N in an empirical setting related to preferences for electric vehicles (EVs). We observe that accounting for decision-uncertainty behavior in GCM-t results in lower elasticity estimates and a higher willingness to pay for improving the EV attributes than those of the GCM-N model. These differences are relevant in making policies to expedite the adoption of EVs.._

# Code Content
The code provides user the oppurtunity to replicate the simulation study reported int he paper. The code is properly commented for a easy follow through. With a little effort, users can modify the code to include external dataset. The file _MNTF.py_ contains all the necessary functions and no additional file is required.

# Dependencies
```
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
```

# Code settings
### Here is the list of variables that can be changed to set various model configuration 
```
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
```