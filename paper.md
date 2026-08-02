---
title: 'pygifi: A Python Implementation of the Gifi System for Optimal Scaling'
tags:
  - Python
  - optimal scaling
  - multivariate analysis
  - categorical data analysis
  - computational mathematics 
authors:
  - name: Vamanie Perumal
    affiliation: '1'
  - name: Sanjeev
    affiliation: '2'
  - name: Bhavesh
    affiliation: '2'
affiliations:
  - index: 1
    name: Indian Institute of Technology Madras, India
  - index: 2
    name: Amrita College of Engineering, Chennai, India
date: 2 August 2026
bibliography: paper.bib
---

# Summary

Optimal scaling extends classical multivariate analysis to data that are not
naturally continuous: nominal categories, ordinal ratings, and mixed-type
variables of the kind produced by surveys, clinical instruments, and
behavioral coding schemes. The Gifi system [@gifi1990; @michailidis1998]
addresses this by jointly estimating a numeric quantification for every
category, subject to the variable's measurement level, and a low-dimensional
representation of the objects being measured, using an alternating least
squares (ALS) algorithm. This family of methods includes homogeneity analysis
(comparable to multiple correspondence analysis), optimal-scaling principal
component analysis, and monotone regression, and has long been implemented in
R's `Gifi` package [@gifiRpackage].

`pygifi` is a native Python implementation of the same computational
framework: `Homals` (homogeneity analysis), `Princals` (optimal-scaling
PCA), and `Morals` (monotone regression), built on a shared ALS engine that
handles nominal, ordinal, and spline-based transformations, missing data, and
tie-breaking rules. It exposes a scikit-learn-compatible `fit`/`transform`
interface and integrates directly with NumPy, SciPy, pandas, and matplotlib
[@harris2020numpy; @virtanen2020scipy; @mckinney2010pandas;
@pedregosa2011scikit], so users already working in Python's data-science
stack do not need to switch languages or maintain an R subprocess to use
Gifi-style methods.

# Statement of need

Python has become a dominant language for data analysis, but native support
for the Gifi family of optimal-scaling methods remains limited: researchers
who need homogeneity analysis, optimal-scaling PCA, or monotone-regression
modeling typically rely on R's `Gifi` package. This is a real barrier for
users whose broader pipelines — preprocessing, modeling, visualization — are
already built in Python, and it complicates reproducibility for projects
that must combine both languages. `pygifi` is intended for statisticians,
psychometricians, and social, educational, and health-science researchers
who work with categorical or mixed-scale data and want optimal scaling
available natively in Python, as well as for developers who need a
transparent, inspectable reference implementation to build on, or who need
optimally-scaled numeric representations of mixed-type data as an input
stage for downstream methods that require purely numeric input.

# State of the field

R's `Gifi` package [@gifiRpackage] remains the reference implementation and
the direct basis for this port. Within Python, the closest existing tool is
`prince` [@princepackage], which provides multiple correspondence analysis and factor
analysis of mixed data via direct singular value decomposition of
(possibly disjunctive-coded) data matrices. That is a different algorithmic
family from Gifi's iterative ALS optimal scaling, which additionally
supports spline-based ordinal transformations, explicit tie-breaking rules,
and monotone regression with an optimally scaled response — `prince` has no
equivalent of `Morals`. Extending an SVD-based package to cover ALS-based
optimal scaling would mean replacing its computational core rather than
adding a method to it, so a dedicated, from-scratch port of the Gifi engine
was the more direct path to a faithful, verifiable implementation, rather
than an approximation grafted onto architecturally different existing
software.

# Software design

`pygifi` decomposes each model into shared infrastructure — data structure
construction, spline and indicator basis preparation, cone projections for
nominal/ordinal/spline restrictions, and a common ALS engine — with
model-specific classes assembling these pieces the way R's `Gifi` does
internally, rather than re-implementing the ALS loop per model. This
shared-core design means a fix or extension to the engine benefits every
model at once, and it made the port independently testable at the component
level (Gram-Schmidt orthogonalization, isotonic regression, B-spline bases)
rather than only as an opaque end-to-end fit.

That testability mattered in practice. A faithful port is only credible if
it demonstrably agrees with the reference implementation, not merely if it
exposes a similar API — plausible-looking code can converge to a different
local optimum, or a subtly wrong default can degrade a result without
raising any error. Comparing intermediate quantities against live R output
during development, rather than only final results, surfaced two defects
that a single end-to-end comparison would have hidden inside compensating
errors: a spline-knot default in `Morals` that diverged from R's actual
default, and an alternative majorization solver whose update formula
violated its own documented monotone-convergence guarantee under unequal
category sizes. `pygifi` ships the validation pipeline that caught these
(`tests/parity/`), which runs the real CRAN `Gifi` package and `pygifi` on
the same twelve bundled datasets using a ported implementation of R's random
number generator, so both languages start from either an identical or an
independently-equivalent initial configuration. Of the 24 `Homals`/`Princals`
runs, 22 match R to under 0.15% relative difference on stress and
eigenvalues (most at floating-point precision, $10^{-9}$% or below); the
`Morals` models match to floating-point tolerance across three regression
datasets. The two remaining cases — one where neither implementation
converges within the iteration budget, one where both reach a valid but
different local optimum on the largest dataset — are recorded, not hidden,
alongside the full comparison in `results/` and documented in
`tests/parity/README.md`. The full test suite (220 tests) and this pipeline
run for every change.

# Research impact statement

`pygifi` is a new package; its immediate evidence of readiness is the
validation pipeline and test suite described above, its public GitHub
repository and issue tracker, and this submission itself. Its authors plan
to use it as a preprocessing stage for Self-Organizing Maps (SOMs), which
require purely numeric input and have no native mechanism for mixed
nominal/ordinal/continuous data; `pygifi`'s optimally scaled quantifications
are intended to supply that numeric representation while preserving each
variable's measurement-level constraints, in place of ad hoc dummy- or
integer-coding.

# AI usage disclosure

An AI coding assistant (Claude, Anthropic) was used throughout this
project's development: diagnosing and fixing the numerical defects described
above, cleaning up the repository structure, and drafting this paper. 
No AI-generated claim was taken on faith — every code fix was checked against the live CRAN `Gifi`
package output, every documentation example in this repository was executed to confirm it runs as written, and the full test suite was run after each change. The numerical results
reported in this paper were read directly from the generated comparison
files (`results/comparison_summary.csv`), not estimated.

# Acknowledgements

We thank the authors of the original R `Gifi` package — Patrick Mair, Jan De
Leeuw, and Patrick J. F. Groenen — whose implementation and documentation
made this port possible, and the broader line of work on alternating least
squares optimal scaling [@deleeuw1977] that the Gifi system builds on.

# References
