# Computational Methods for the manuscript "Spatial discordances between mRNAs and proteins in the intestinal epithelium"

The repository contains code associated with the parameter estimation process in the manuscript "Spatial discordances between mRNAs and proteins in the intestinal epithelium" by Yotam Harnik, Lisa Buchauer, Shani Ben-Moshe, Yishai Levine, Alon Savidor, Raya Eilam, Andreas E. Moor and Shalev Itzkovitz.

## Structure of the repository

The repository contains analysis scripts associated with different parts of the study. Each part is contained in its own directory as follows:

- parameter_estimation - Protein translation and decay rate estimation based on mRNA and protein profiles. This part requires python 3.8 and the third-party packages numpy, matplotlib, scipy, seaborn, pandas, scipy, emcee and corner. The parameter process is outlined in five jupyter notebooks (N1 - N5) which detail data preprocessing, prior construction, MCMC sampling and result validation. The directory further contains three external data sets used for comparison to the results derived here and some code meant to facilitate large-scale parameter estimation on computational clusters.

## Additional data

All data which is directly required for executing the scripts in this repository is likewise contained in the repository. Raw mRNA sequencing data have been deposited in the GenBank GEO database under accession code GSE164746 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164746). The full collection of model fit results and parameter estimates, which includes MCMC-chains approximating the posterior parameter distribution and figures showing the model's fit to the data for each gene, can be accessed in a zenodo repository (https://zenodo.org/record/4438689).
