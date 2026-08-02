# pygifi: A Python Implementation of the Gifi System for Optimal Scaling

## Summary

Optimal scaling methods extend classical multivariate analysis to data that are not naturally continuous, including nominal, ordinal, and mixed-type variables. In many empirical settings, researchers work with survey responses, clinical ratings, educational scores, behavioral categories, and other structured observations that cannot be handled appropriately by ordinary principal component analysis, linear regression, or canonical correlation without imposing questionable numeric assumptions. The Gifi system, developed in R, addresses this problem by transforming variables optimally under measurement restrictions and then applying low-dimensional multivariate methods within an alternating least squares framework.

`pygifi` is a Python implementation of core functionality provided by the R `Gifi` package. The project ports key methods from the Gifi framework into a Python environment while preserving the conceptual structure of optimal scaling, category quantification, ordinal restrictions, spline-based transformations, and iterative estimation. The package provides a native Python interface for methods such as homogeneity analysis (`Homals`), optimal-scaling principal components analysis (`Princals`), monotone regression (`Morals`), and related utilities for coding, spline preparation, cross-validation, visualization, and validation against R outputs.

The goal of the project is not merely to wrap R functionality but to reproduce the underlying computational workflow in Python so that users in scientific computing, data analysis, and machine learning ecosystems can work with Gifi-style methods directly alongside tools such as NumPy, pandas, SciPy, matplotlib, and scikit-learn. This makes optimal scaling methods more accessible to Python users while also supporting reproducible comparisons between R and Python implementations.

## Statement of Need

Although Python has become a dominant language for data science and machine learning, native support for optimal scaling methods remains limited compared with R. Researchers who want to perform nonlinear principal components analysis, homogeneity analysis, or monotone transformation-based multivariate modeling often rely on the R `Gifi` package or related R implementations. This creates a barrier for users whose broader workflows are already built in Python, especially when they need to integrate preprocessing, modeling, validation, and visualization within a single computational environment.

A Python implementation is useful for at least three reasons. First, it lowers the practical cost of adopting optimal scaling methods in Python-centered research workflows. Users can remain within the Python ecosystem rather than switching languages or maintaining cross-language bridges. Second, it improves interoperability with widely used Python libraries for machine learning and scientific computing. Third, it supports pedagogical and methodological transparency by making the structure of the algorithms available in a language familiar to a large user base.

The conversion from R `Gifi` to Python also addresses a reproducibility need. In practice, ports of statistical methods are only credible when they demonstrate agreement with an established reference implementation. For that reason, the project includes parity-oriented validation against the R package, including numerical comparisons of transformed outputs, quantifications, and related model quantities. This is especially important for iterative multivariate procedures, where small differences in initialization, transformation handling, or numerical conventions can produce divergent solutions.

The intended audience includes statisticians, psychometricians, social scientists, educational researchers, health researchers, and method developers who work with categorical or mixed-scale data and want direct access to optimal scaling methods in Python. It is also relevant for developers who need a transparent, inspectable implementation of Gifi-style methods for downstream experimentation or integration.

## Implementation

`pygifi` is organized around model classes and shared computational infrastructure. At the user level, classes such as `Princals`, `Homals`, and `Morals` expose fit-oriented interfaces that are familiar to Python users. Underneath these model classes, the package constructs Gifi-style data structures, prepares spline and indicator bases, applies ordinal or nominal restrictions, and invokes a shared alternating least squares engine. This design mirrors the conceptual decomposition of the original R framework while adapting it to Pythonic data structures.

The implementation includes support utilities for categorical encoding and decoding, spline knot generation, isotonic and cone-based projections, and cross-validation helpers. Built-in datasets and plotting functions support demonstration and interpretation. The package also includes a dedicated random-number compatibility component used for exact parity-oriented initialization when comparing selected procedures to the R implementation. This is important because iterative optimal scaling methods can be sensitive to starting values.

From a software engineering perspective, the project aims to provide more than a one-to-one syntax translation. The conversion required adapting the algorithmic structure to Python's numerical stack, managing differences in data representation, handling categorical encodings in pandas, and reproducing transformation logic in a way that remains inspectable and testable. The resulting implementation is intended to function as a native Python library rather than as a language bridge.

## Validation and Testing

Because this project is a methodological port, validation is central to its contribution. The repository includes automated tests for core numerical utilities, model behavior, and parity-oriented checks. In addition to unit tests for internal components such as linear algebra helpers, coding utilities, isotonic procedures, and spline functions, the project includes comparisons against the R `Gifi` implementation for selected workflows.

This validation strategy serves two purposes. First, it checks internal correctness of the Python implementation at the component level. Second, it evaluates whether the Python outputs are consistent with the established R reference for practically meaningful model results such as category quantifications, transformed datasets, and eigenvalue-based summaries. This kind of validation is particularly valuable for a porting project because correctness cannot be assessed solely by API similarity; it must be supported by numerical agreement and behavioral consistency.

## Comparison to Existing Software

The main point of comparison for `pygifi` is the R `Gifi` package, which remains the primary reference implementation for these methods. The original package provides a mature environment for optimal scaling techniques and has long served as an important resource for researchers working with categorical multivariate data. However, its use requires working in R, which limits direct integration with Python-first analysis pipelines.

To the best of this project's design goals, `pygifi` fills the gap of a native Python implementation rather than offering a wrapper around R. This distinction matters for usability, portability, and long-term maintainability. A wrapper would still require an R runtime and would reduce transparency for Python users. A native implementation, by contrast, can be inspected, extended, tested, and integrated within standard Python workflows.

The contribution of `pygifi` is therefore not novelty in the statistical methods themselves, which originate in the Gifi framework, but accessibility, interoperability, and reproducibility in a Python setting. In that sense, the project complements rather than replaces the original R package: R `Gifi` remains the methodological reference, while `pygifi` provides Python users with direct access to the same class of tools.

## Conclusion

`pygifi` brings the Gifi system of optimal scaling methods into Python through a native implementation of key models, shared computational components, and validation workflows against the established R package. The project addresses a practical and methodological need for Python users who work with categorical, ordinal, and mixed-type data and want access to optimal scaling methods without leaving the Python ecosystem. By combining model implementations, supporting utilities, and parity-oriented validation, the package provides a foundation for broader use, further development, and reproducible application of Gifi methods in Python.

## Placeholders To Complete Before Submission

- Add the final author list and affiliations.
- Add the archived software DOI.
- Add citations to the original `Gifi` package and core methodological references.
- Add any published or in-progress applications of `pygifi`, if available.
