# python 3.8

""" File that offers functions required for performing MCMC sampling
on a given gene. Provides method fit_one_gene() for calling by batch
parameter estimation jobs. """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm, gamma
import dill as pickle
import emcee
import corner
import seaborn as sns


def fit_one_gene(
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
        thin=15):
    # get data for this gene as array
    protein_vals = P_data.loc[gene].values
    protein_errors = Psem_data.loc[gene].values
    mRNA_vals = M_data.loc[gene].values
    mRNA_fun = M_interp_dict[gene]
    log_beta_0 = np.log(start_values.loc[gene]['beta_0'])
    log_delta_0 = np.log(start_values.loc[gene]['delta_0'])

    # define time vector
    time_vec = np.linspace(0, 96, 6)

    # initialize sampling ensemble with 32 walkers
    nwalkers = 32
    ndim = 3
    pos = np.array([log_beta_0, log_delta_0, protein_vals[0]]) \
        + 1e-4 * np.random.randn(nwalkers, ndim)

    # sample!
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(time_vec, protein_vals,
                                          protein_errors, mRNA_fun,
                                          beta_gamma_dist, delta_gamma_dist,
                                          TE_fun_norm))
    sampler.run_mcmc(pos, Nsteps)

    # plot autocorrelation plots from the unaltered sampler
    plot_autocorr(gene, sampler)

    # extract samples
    flat_samples = sampler.get_chain(discard=Ndiscard, thin=thin, flat=True)
    flat_probs = sampler.get_log_prob(discard=Ndiscard, thin=thin, flat=True)

    # export part of the chain
    sample_df = pd.DataFrame(data=np.hstack((flat_samples,
                                             flat_probs.reshape((-1, 1)))),
                             columns=['log_beta', 'log_delta', 'Pzero', 'log_prob'])

    # extract medians
    beta_med, delta_med, Pzero_med = np.median(flat_samples, axis=0)

    # plot corner plots
    plot_corner(gene, flat_samples)

    # plot profiles
    plot_model_profiles(gene,
                        sample_df,
                        mRNA_vals,
                        protein_vals,
                        protein_errors,
                        mRNA_fun,
                        TE_fun_norm)

    sample_df.to_csv('MCMC_results/{}_chain_sample.csv'.format(gene))


def import_data(
        import_folder='processed_data'):
    """ Given the path to the import data, gets pandas dataframes for
    mRNA and preotein profiles as well as protein SEMs and the mRNA
    interpolation obect plus frozen gamma distributions from the specified
    parameters (beta and delta) and returns them in this order."""
    # load three dataframes
    M_data = pd.read_csv(
        import_folder+"/M_data_sc_based_scaled.csv",
        index_col=0)
    P_data = pd.read_csv(
        import_folder+"/P_data_vol_corrected_scaled.csv",
        index_col=0)
    Psem_data = pd.read_csv(
        import_folder+"/Psem_data_vol_corrected_scaled.csv",
        index_col=0)
    # and the dict with linear mrna interpolations
    with open(import_folder+"/M_interp_dict", 'rb') as file:
        M_interp_dict = pickle.load(file)

    # load parameters specifying gamma distribution for log beta and
    # log delta priors
    beta_df = pd.read_csv(
        import_folder+'/log_beta_gamma_fit.csv',
        index_col=0)
    delta_df = pd.read_csv(
        import_folder+'/log_delta_gamma_fit.csv',
        index_col=0)
    # make frozen gamma distributions from these
    beta_gamma_dist = gamma(a=beta_df.loc['a'].value,
                            loc=beta_df.loc['loc'].value,
                            scale=beta_df.loc['scale'].value)
    delta_gamma_dist = gamma(a=delta_df.loc['a'].value,
                             loc=delta_df.loc['loc'].value,
                             scale=delta_df.loc['scale'].value)
    
    # load beta_0 and delta_0 values for as MCMC chain starting positions
    # these values were obtained during a preliminary model run and serve
    # to increase sampling efficiency and reduce burn-in time
    start_values = pd.read_csv(
        import_folder+'/beta_delta_start_values.csv',
        index_col=0)

    # load the function describing translation efficiency decline
    with open(import_folder+"/TE_decay_function", 'rb') as file:
        TE_fun_norm = pickle.load(file)

    return M_data, P_data, Psem_data, M_interp_dict, beta_gamma_dist,\
        delta_gamma_dist, start_values, TE_fun_norm


""" Model functions """


def protein_ODE(
        p,
        t,
        mRNA_fun,
        beta,
        delta,
        TE_fun_norm):
    """ Protein dynamics given current mRNA levels and translation,
    secretion and decay rates
    """
    dp = beta * mRNA_fun(t) * TE_fun_norm(t) - delta * p
    return dp


def log_likelihood(
        theta,
        time_vec,
        protein_vals,
        protein_errors,
        mRNA_fun,
        TE_fun_norm):
    # unpack parameters
    log_beta, log_delta, Pzero = theta

    # calculate model values
    model_vals = odeint(
        protein_ODE,
        Pzero,
        time_vec,
        args=(mRNA_fun, np.exp(log_beta), np.exp(log_delta), TE_fun_norm)).T[0]

    # calculate log likelihood from this (omitting constants)
    log_like = -0.5 * np.sum(((protein_vals - model_vals) /
                              protein_errors)**2)

    return log_like


def log_prior(
        theta,
        beta_gamma_dist,
        delta_gamma_dist):
    log_beta, log_delta, Pzero = theta

    # get the probability for these beta and delta values independetly
    p_log_beta = beta_gamma_dist.pdf(log_beta)
    p_log_delta = delta_gamma_dist.pdf(log_delta)
    # for P0 assume uniform distribution over reasonable values
    if 0 <= Pzero < 3e8:
        p_Pzero = 1/3e8
    else:
        p_Pzero = 0

    # if one or more of them are 0, return -inf
    if (p_log_beta < 1e-9) or (p_log_delta < 1e-9) or (p_Pzero == 0):
        return -np.inf

    # else return sum of the logs
    return np.log(p_log_beta) + np.log(p_log_delta) + np.log(p_Pzero)


def log_probability(
        theta,
        time_vec,
        protein_vals,
        protein_errors,
        mRNA_fun,
        beta_gamma_dist,
        delta_gamma_dist,
        TE_fun_norm):
    lp = log_prior(theta, beta_gamma_dist, delta_gamma_dist)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, time_vec, protein_vals,
                               protein_errors, mRNA_fun, TE_fun_norm)


""" Plot functions """


def plot_corner(
        gene,
        flat_samples,
        labels=['log_beta', 'log_delta'],
        show=False):
    # we include only delta and beta in the corner plot,
    # as P0 values are of little general interest
    fig = corner.corner(flat_samples[:, 0:2],
                        labels=labels)
    fig.suptitle(gene)
    plt.savefig("figures/{}_corner.pdf".format(gene))
    if show:
        plt.show()
    else:
        plt.close()


def plot_autocorr(
        gene,
        sampler,
        show=False):
    fig, axes = plt.subplots(3, figsize=(10, 10), sharex=True)
    samples = sampler.get_chain()
    labels = ["log_beta", "log_delta", "Pzero"]
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig('figures/{}_autocorr.png'.format(gene), dpi=75)
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_profiles(
        gene,
        sample_df,
        mRNA_vals,
        protein_vals,
        protein_errors,
        mRNA_fun,
        TE_fun_norm,
        show=False):
    """ This function not only produces figures, but also calculates posterior
    predictive p values as a measure of model fitness. Model fitness
    information is exported as csv. """
    # define time vector
    time_vec = np.linspace(0, 96, 6)

    # get mode vals for plot
    beta_mode = np.exp(
        sample_df.iloc[sample_df['log_prob'].argmax()]['log_beta'])
    delta_mode = np.exp(
        sample_df.iloc[sample_df['log_prob'].argmax()]['log_delta'])
    Pzero_mode = sample_df.iloc[sample_df['log_prob'].argmax()]['Pzero']

    # create profiles for a number of draws and store them
    n_draws = 2000
    mod_array = np.zeros((n_draws, 6))
    # get indices of n_draws samples
    rng = np.random.default_rng()
    inds = rng.choice(len(sample_df), n_draws, replace=False)

    # below, we want to count the number of realizations in which
    # chi2_pred > chi2_obs
    chi2_counter = 0

    chi2_obs_list = []
    chi2_pred_list = []

    # for each one, calculate the profile and store it - also use the
    # loop to get posterior predictive p value [Gelman et al 1996]
    for i, ind in enumerate(inds):
        sample = sample_df.iloc[ind][['log_beta', 'log_delta', 'Pzero']].values
        model_vals = odeint(
            protein_ODE,
            sample[2],
            time_vec,
            args=(mRNA_fun, np.exp(sample[0]), np.exp(sample[1]), TE_fun_norm)).T[0]
        mod_array[i, :] = model_vals

        # also, calculate the chi2 or the observed data with respect
        # to this model
        chi2_obs = np.sum(((model_vals - protein_vals) /
                          np.maximum(protein_errors, 0.0001))**2)
        chi2_obs_list.append(chi2_obs)

        # calculate relative errors from data and propagate them
        # to the predicted data
        rel_errors = protein_errors / np.maximum(protein_vals, 0.0001)
        pred_errors = model_vals * rel_errors

        # errors should be strictly positive
        pred_errors = np.maximum(pred_errors, 0.0001)

        # now, assuming gaussian errors, get a data realisation
        # at each timepoint
        protein_drawn = np.array([norm(model_vals[i],
                                  pred_errors[i]).rvs() for i in range(6)])
        # this could potentially contain sub-0 entries because of the
        # normal dist, set these to 0
        protein_drawn = np.maximum(protein_drawn, 0)

        # calculate chi2 for this predicted data set
        chi2_pred = np.sum(((model_vals - protein_drawn) /
                            pred_errors)**2)
        chi2_pred_list.append(chi2_pred)

        # if larger than chi2_obs, trigger the counter
        if chi2_pred > chi2_obs:
            chi2_counter += 1

    p_val = chi2_counter/n_draws

    # for each time point, get the HDI intervals 68% and 95% ptobability mass
    # and write them to file
    res_dict = {}
    for pmass in [0.68, 0.95]:
        for j, tp in enumerate(['tp1', 'tp2', 'tp3', 'tp4', 'tp5', 'tp6']):
            # get HDI for each timepoint
            left, right = HDI(mod_array[:, j], pmass)
            res_dict['model_HDI{}_left_{}'.format(pmass, tp)] = left
            res_dict['model_HDI{}_right_{}'.format(pmass, tp)] = right
    # add the posterior predictive p value to the dict
    res_dict['pp_pval'] = p_val

    # make dataframe from the dict and export it
    res_df = pd.DataFrame.from_dict(res_dict,
                                    orient='index',
                                    columns=['value'])
    res_df.to_csv('MCMC_results/{}_posterior_predictive_results.csv'.format(
        gene))

    # plot their profiles, in order to get mRNA and protein onto the same
    # scale, normalise everything by mean
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(5, 5))
    # now, plot trace for each realisation
    leg = True
    for i in range(np.min((n_draws, 300))):
        if leg:
            ax.plot(time_vec, mod_array[i, :]/np.mean(mod_array[i, :]), '-',
                    color='grey', linewidth=1, alpha=0.2)
            leg = False
        else:
            ax.plot(time_vec, mod_array[i, :]/np.mean(mod_array[i, :]), '-',
                    color='grey', linewidth=1, alpha=0.2)

    # plot mRNA
    ax.plot(time_vec, mRNA_vals/np.mean(mRNA_vals), '-o', color='darkblue',
            label='mRNA', linewidth=3)

    # plot protein sem
    ax.fill_between(
        time_vec,
        (protein_vals-protein_errors)/np.mean(protein_vals),
        (protein_vals+protein_errors)/np.mean(protein_vals),
        color='crimson',
        label='protein SEM',
        alpha=0.3)

    # plot protein
    ax.plot(time_vec, protein_vals/np.mean(protein_vals), '-o',
            color='crimson', label='protein', linewidth=3)

    # plot model MAP
    model_vals = odeint(
        protein_ODE,
        Pzero_mode,
        time_vec,
        args=(mRNA_fun, beta_mode, delta_mode, TE_fun_norm)).T[0]
    ax.plot(time_vec, model_vals/np.mean(model_vals), '--',
            color='black', label='model MAP', linewidth=3, alpha=1)
    
    ax.set_title(gene + ", half life = {} h, pval={}".format(
        np.round(np.log(2)/delta_mode, 3), p_val))

    ax.set_ylabel('mean-normalized expression')
    ax.set_xlabel('time (hours)')

    sns.despine()
    plt.legend(loc=0)

    plt.savefig('figures/{}_model_profiles.pdf'.format(gene),
                dpi=75, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# For unimodal distributions, a meaningful highest density interval containing
# a percentage of the probability mass can accompany the mode as MAP estimator
def HDI(posterior_samples, credible_mass):
    """ Computes highest density interval from a sample of representative
    values, estimated as the shortest credible interval Takes Arguments
    posterior_samples (samples from posterior) and credible mass."""
    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0]*nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
    return(HDImin, HDImax)
