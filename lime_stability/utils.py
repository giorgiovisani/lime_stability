"""
This is a script containing bunch of utility functions,
helpful to build the stability indices for Lime
"""

import numpy as np
from itertools import combinations
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import WLS


class LocalModelError(Exception):
    """Custom Exception raised when the model is not Weighted Ridge Regression"""
    pass


def compute_WLS_stdevs(X, Y, weights, alpha):
    """Function to calculate standard deviations of Weighted Ridge coefficients

    Args:
        X: dataset containing the explanatory variables
        Y: vector of the response variable
        weights: vector of weights (one for each tuple of the X dataset)
        alpha: regularization parameter

    Returns:
        stdevs_beta: list containing the standard deviations of the coefficients
        """

    # Build Weighted Regression (WLS) model
    X_enh = add_constant(X)
    wls_model = WLS(Y, X_enh, weights=weights)
    results = wls_model.fit()
    errors_wls_weighted = results.wresid

    # Estimate of the sigma squared quantity
    sigma2 = np.dot(errors_wls_weighted.T, errors_wls_weighted) / (X.shape[0] - X.shape[1])
    weights_matr = np.diag(weights)  # reformulate weights as diagonal matrix

    # Standard deviations of the coefficients
    partial_ = np.linalg.inv(np.linalg.multi_dot([X.T, weights_matr, X]) +
                             alpha * np.diag([1, ] * X.shape[1]))
    variances_beta_matrix = sigma2 * np.linalg.multi_dot(
        [partial_, X.T, weights_matr, X, partial_.T])
    variances_beta = np.diag(variances_beta_matrix)
    stdevs_beta = list(np.sqrt(variances_beta))

    return stdevs_beta


def refactor_confints_todict(means, st_devs, feat_names):
    """Refactor means and confidence intervals into a dictionary

    Args:
        means: list of the means of the WRR coefficients
        st_devs: list of the standard deviations of the WRR coefficients
        feat_names: list of feature names associated with the coefficients

    Returns:
        conf_int: dictionary,
            key = the feature name
            value = confidence interval for the feature (upper, lower bound)
    """

    conf_int = {}
    for name, mean, stdev in zip(feat_names, means, st_devs):
        conf_int[name] = [mean - 1.96 * stdev, mean + 1.96 * stdev]
    return conf_int


def compare_confints(confidence_intervals, index_verbose=False):
    """Function to compare confidence intervals obtained through different WRR,
        which are built with the same number of features (possibly different ones, but the same number).
        Core function of the package: calculates the two complementary indices CSI, VSI.

    Args:
        confidence_intervals: list of dictionaries,
            each dictionary is the output of the confint function.
        index_verbose: Controls for the verbosity at the stability indices level,
            when set to True gives information about partial values related to stability.

    Returns:
        csi: Coefficients stability index
        vsi: Variables stability index
    """

    n_features = len(confidence_intervals[0].keys())
    features_limes = []
    for conf_int in confidence_intervals:
        features_limes.append(conf_int.keys())
    unique_features = list(set([l for ll in features_limes for l in ll]))

    # Calculate CSI
    overlapping_tot = []
    for feat in unique_features:
        conf_int_feat = []
        for conf_int in confidence_intervals:
            conf_int_feat.append(conf_int.get(feat))

        if len(conf_int_feat) < 2:
            pass
        else:
            overlapping = []
            for pair_intervals in combinations(conf_int_feat, 2):
                i1, i2 = pair_intervals
                is_overlap = True if (i1[0] < i2[1] and i2[0] < i1[1]) else False
                overlapping.append(is_overlap)
            frac_overlapping = round(sum(overlapping) / len(overlapping) * 100, 2)
            overlapping_tot.append(frac_overlapping)
            if index_verbose:
                print("""Percentage of overlapping confidence intervals, variable {}: {}%\n""".format(
                    feat, frac_overlapping))

    csi = round(np.mean(overlapping_tot), 2)

    # Calculate VSI
    same_vars = 0
    n_combs = 0
    for pair_vars in combinations(features_limes, 2):
        var1, var2 = pair_vars
        same_vars += len(set(var1) & set(var2))
        n_combs += 1
    vsi = round(same_vars / (n_combs * n_features) * 100, 2)
    if index_verbose:
        print("""Percentage same variables across repeated LIME calls: {}%\n""".format(vsi))

    return csi, vsi
