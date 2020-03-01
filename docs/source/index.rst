.. DCDF documentation master file, created by
   sphinx-quickstart on Mon Feb  3 15:36:41 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DCDF Documentation Homepage
===========================

.. toctree::
   :includehidden:
   :maxdepth: 2
   :glob:
   :caption: Contents:

   pages/user-guide.rst
   pages/code.rst

General Framework
=================

In neuroimaging, summary statistics are frequently used in an attempt to describe various aspects of a patient’s neuroanatomy.  For example, one might measure the mean FA intensity across some white matter region of interest in order to quantify structural integrity.  Similarly, PSMD, defined as the width between the 5th and 95th percentile in a skeletonized mean diffusivity map, has recently been used to quantify vascular burden in an SVD cohort.  Such statistics are generally intended to maximally describe the underlying data generating process. Unless a statistic is sufficient, however, there is no guarantee that it will be able to capture information which correlates well with independent variables of interest. Furthermore, non-trivial sufficient statistics require the underlying distribution to be known in parametric form, and, even then, the theoretical derivations can be extremely complicated.

To date, no group has parameterized the distribution of DTI metrics (AD, FA, MD, & RD) within white matter, and so no non-trivial sufficient statistic is known for these distributions.  To bridge this gap, we have devised a general framework which allows differences between a target and a reference distribution to be weighted according to a user supplied function.  This allows custom statistics to be developed which may carry more information about independent variables of interest than traditional summary statistics are able to convey.

We begin by considering two cumulative distribution functions (CDF).  Let FR denote the CDF of some reference distribution.  This could be the distribution of a group of healthy controls or an entire population.  Let FS denote the CDF of a single sample or subject.  Our framework proposes statistics which take the form:

.. math::
	\int_{l}^{u} \phi\big(F_{R}^{-1}(x) - F_{S}^{-1}(x)\big)dx

Where :math:`\phi` is a weighting function applied to the differences of the inverse CDFs.  Using the inverse CDF is preferred here, as the differences refer to the measure difference between corresponding quantiles.  That is, :math:`F_{R}^{-1}(0.5) - F_{S}^{-1}(0.5)` can be interpreted as the measured difference between the medians of the two distributions.  Furthermore, the use of inverse CDFs allows the integral to be defined cleanly in terms of quantiles as opposed to measure specific values.

It is worth noting the case when :math:`\phi` is the identity function, that is :math:`\phi(x) = x`, we have

.. math::
	\int_{l}^{u} \big(F_{R}^{-1}(x) - F_{S}^{-1}(x)\big)dx &= \int_{l}^{u} F_{R}^{-1}(x) - \int_{l}^{u} F_{S}^{-1}(x) \\
				&= \mathbb{E}_R\big(X|F_{R}^{-1}(l) < X < F_{R}^{-1}(u)\big) - \mathbb{E}_S\big(X|F_{S}^{-1}(l) < X < F_{S}^{-1}(u)\big)


	
	

Noting that :math:`\mathbb{E}_S(X|F_{S}^{-1}(l) < X < F_{S}^{-1}(u))` is the conditional expectation of a random variable :math:`X`  with respect to distribution :math:`F`. This is simply the difference between the truncated means of the two distributions.  When comparing subjects evaluated against the same reference, the truncated mean of the reference distribution can be treated as constant and thus ignored.

Implementation
==============
This was developed to perform numerical integration using Reimann sums over user specified functions (:math:`\phi`).  These functions can be defined using Python 3 syntax and any functions available through the Numpy library.  The reference distribution is first estimated by calculating the running average of cumulative bin frequencies over a user supplied reference list of images.  Subjects are then evaluated in parallel against the generated reference.



.. autosummary::



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
