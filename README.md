# Stability Indices for LIME

[![Build Status](https://travis-ci.org/giorgiovisani/lime_stability.svg?branch=master)](https://travis-ci.org/giorgiovisani/lime_stability)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/giorgiovisani/lime_stability/master)
![PyPI - License](https://img.shields.io/pypi/l/lime_stability?style=plastic)
![PyPI](https://img.shields.io/pypi/v/lime_stability?style=plastic)
![PyPI - Downloads](https://img.shields.io/pypi/dw/lime-stability?style=plastic)


This project is about measuring the stability of explanations obtained through the [LIME](https://github.com/marcotcr/lime) tool.

LIME (Local Interpretable Model-agnostic Explanations) is a quite well-established albeit recent technique, which enables to understand Machine Learning model's reasoning.  
*For a deeper understanding of the technique, we suggest to read the [paper]() by Ribeiro, as well as to watch its [video](https://www.youtube.com/watch?v=hUnRCxnydCc)*.

Even though LIME is a great tool, it **suffers from a lack of stability**, namely repeated applications of the method, under the same conditions, may obtain different results.
Even worse, many times the stability issue is not spotted at all by the practitioner, e.g. when just a single call to the method is performed and the result is considered to be okay without further checks.

**An explanation can be considered reliable only if unambiguous**.  
Guided by this notion, we developed a pair of complementary indices to evaluate LIME stability: **Variables Stability Index (VSI)** and **Coefficients Stability Index (CSI)**.  

The method creates repeated LIME explanations for the same data point to be explained.  
The VSI index checks whether the different LIMEs give back the same variables as explanation.  
The CSI index controls whether the coefficients of each variable, under the repeated LIME calls, can be considered equal.

The indices can be calculated on every trained LIME method. Both of them range from 0 to 100, where higher values mean that the tested LIME is stable. They are designed to be used together, since each one tests a different concept of stability.

Using the indices will bring enhanced confidence in LIME's results: the practitioner may find out possible instability in its trained LIME, or he may vouch for its consistency.

To get a deeper understanding about the approach, we suggest reading the paper [Statistical stability indices for LIME: obtaining reliable explanations
for Machine Learning models [1]](https://arxiv.org/pdf/2001.11757.pdf).

&nbsp;

[1] Visani, Giorgio, et al. "Statistical stability indices for LIME: obtaining reliable explanations for Machine Learning models." arXiv preprint arXiv:2001.11757 (2020).

## Installation

Installing through pip package manager:

```bash
pip install lime-stability
```

### Prerequisites
List of dependencies:  

* lime
* statsmodels
* sklearn
* numpy

## Types of data to use the indices on

The stability indices in `lime_stability` are currently available only for tabular data (LimeTabularExplainer class). Although the theory behind the indices allow for their usage also with other types of data, but the implementation is not available yet.

## Authors

* **Giorgio Visani**: [institutional page](https://www.unibo.it/sitoweb/giorgio.visani2/en)

## Acknowldegments

We would like to thank [CRIF S.p.A.](https://www.crif.com/) and [Università degli Studi di Bologna](https://www.unibo.it/en), which financially supported the project.




