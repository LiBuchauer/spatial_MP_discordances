# -*- coding: utf-8 -*-
# python 3.8
""" Entry point to MCMC to be called from bash script, receives gene arguments
and runs MCMC chain for each of them."""

import sys
# for importing module from parent directory
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from MCMC_pipe import fit_one_gene, import_data
import numpy as np

gene_list = sys.argv[1:]
print(gene_list)

# import data once globally to avoid multiple import
M_data, P_data, Psem_data, M_interp_dict, beta_gamma_dist, delta_gamma_dist, start_values, TE_fun_norm = \
    import_data(import_folder='../processed_data')

# go through genes and start the fit process
for gene in gene_list:
    print(gene)
    fit_one_gene(
        gene,
        M_data,
        P_data,
        Psem_data,
        M_interp_dict,
        beta_gamma_dist,
        delta_gamma_dist,
        start_values,
        TE_fun_norm,
        nwalkers=32,
        Nsteps=6000,
        Ndiscard=500,
        thin=15)
