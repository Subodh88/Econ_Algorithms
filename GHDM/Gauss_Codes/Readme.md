# Paper Title
The code is for the paper titled _**A new generalized heterogeneous data model (GHDM) to jointly model mixed types of dependent variables**_. You can access the paper [**Here**](https://www.sciencedirect.com/science/article/abs/pii/S0191261515001198). This model is useful in situations where one wish to model inter-dependencies between various type of dependent variables (continuous, count, ordinal, and un-ordered (binary and multinomial both).

# Paper Abstract
_This paper formulates a generalized heterogeneous data model (GHDM) that jointly handles mixed types of dependent variables—including multiple nominal outcomes, multiple ordinal variables, and multiple count variables, as well as multiple continuous variables—by representing the covariance relationships among them through a reduced number of latent factors. Sufficiency conditions for identification of the GHDM parameters are presented. The maximum approximate composite marginal likelihood (MACML) method is proposed to estimate this jointly mixed model system. This estimation method provides computational time advantages since the dimensionality of integration in the likelihood function is independent of the number of latent factors. The study undertakes a simulation experiment within the virtual context of integrating residential location choice and travel behavior to evaluate the ability of the MACML approach to recover parameters. The simulation results show that the MACML approach effectively recovers underlying parameters, and also that ignoring the multi-dimensional nature of the relationship among mixed types of dependent variables can lead not only to inconsistent parameter estimation, but also have important implications for policy analysis._

# Code Content
The code provides user the oppurtunity to replicate the simulation study reported int he paper. The code is properly commented for a easy follow through. With a little effort, users can modify the code to include external dataset. The file _GHDM_Aspatial_Final.gss_ contains all the necessary functions and no additional file is required.

# Dependencies
Users will require Gauss IDE to run the code along with Maimum likelihood module. See [here for more details](https://www.aptech.com/).


# Code settings
Detailed instruction are provided in the doc titled 'Code Documentation for the GHDM.docx'.