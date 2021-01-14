# -*- coding: utf-8 -*-
# python 3.8
""" Imports all genes and creates bash job files with batches of genes
from that list."""
import numpy as np
import pickle

# first read gene list from pickle
with open("processed_data/gene_names", "rb") as file:
    gene_names = pickle.load(file)

# now, for defined batch size, write a job file
batch_size = 25
iterations = int(np.ceil(len(gene_names)/batch_size))

for i in range(iterations):
    # get gene names for this batch job
    genes_sub = gene_names[i*batch_size:(i+1)*batch_size]

    with open("jobs/dynGE_job_{}.sh".format(i), "w") as f:
        f.write("""module load anaconda/2020.02/python/3.7
conda activate dynGE

python job_head_MCMC.py {}
                """.format(" ".join(genes_sub)))
