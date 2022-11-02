# Paper Title
The code is for the paper titled _**A new estimation approach to integrate latent psychological constructs in choice modeling**_. You can access the paper [**Here**](https://www.sciencedirect.com/science/article/abs/pii/S0191261514000678). This model integrates a classification algorithm (Multinomial probit) with latent variable model (Structural Equation model). Such an integration allows for capturing the effect of attitudes and habits on the choice variable. T

# Paper Abstract
_In the current paper, we propose a new multinomial probit-based model formulation for integrated choice and latent variable (ICLV) models, which, as we show in the paper, has several important advantages relative to the traditional logit kernel-based ICLV formulation. Combining this MNP-based ICLV model formulation with Bhat’s maximum approximate composite marginal likelihood (MACML) inference approach resolves the specification and estimation challenges that are typically encountered with the traditional ICLV formulation estimated using simulation approaches. Our proposed approach can provide very substantial computational time advantages, because the dimensionality of integration in the log-likelihood function is independent of the number of latent variables. Further, our proposed approach easily accommodates ordinal indicators for the latent variables, as well as combinations of ordinal and continuous response indicators. The approach can be extended in a relatively straightforward fashion to also include nominal indicator variables. A simulation exercise in the virtual context of travel mode choice shows that the MACML inference approach is very effective at recovering parameters. The time for convergence is of the order of 30–80 min for sample sizes ranging from 500 observations to 2000 observations, in contrast to much longer times for convergence experienced in typical ICLV model estimations._

# Code Content
The code provides user the oppurtunity to replicate the simulation study reported int he paper. The code is properly commented for a easy follow through. With a little effort, users can modify the code to include external dataset. The file _ICLV_finalversion.gss_ contains all the necessary functions and no additional file is required.

# Dependencies
Users will require Gauss IDE to run the code along with Maimum likelihood module. See [here for more details](https://www.aptech.com/).


# Code settings
Detailed instruction are provided in the doc titled 'ICLV_documentation.docx'.