=======
omigami
=======
<h4 align="center">Minimally biased features selection for untargeted metabolomics.</h4>


.. image:: https://img.shields.io/pypi/v/omigami.svg
        :target: https://pypi.python.org/pypi/omigami

..
    .. image:: https://img.shields.io/travis/datarevenue-berlin/omigami.svg
        :target: https://travis-ci.org/datarevenue-berlin/omigami

..
    .. image:: https://readthedocs.org/projects/omigami/badge/?version=latest
        :target: https://omigami.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Easy and powerful tool to select features in untargeted metabolomics studies – with less risk for bias. How?
Omigami uses multivariate models to perform a recursive feature elimination within a repeated double cross-validation.

Install
-------

```sh
pip install omigami
```

Motivation
-------
- It's hard to protect against selection bias on high-demensional omics data ([Krawczuk and Łukaszuk, 2016](https://www.sciencedirect.com/science/article/pii/S0933365715001426)). Even common cross-validation has been shown to overfit.
- Most feature selection techniques focus on finding the minimal set of strongest features. Omitting redundant variables that are however still relevant to understanding the biochemical systems.
- There's no freely available and easy-to-use Python tool that implements a minimally biased repeated double cross validation.
- A robust selection requires many (100 - 5.000) models to be trained. Running such a large number of models in reasonable time, requires non-trivial parallelization.

Features
--------
- [x] Repeated double cross-validation
- [x] Multivariate feature selection (Random Forest, XGB or PLS-DA)
- [x] Minimal optimal and all relevant feature selection
- [x] Efficient Parallelization (with Dask)
- [ ] Plotting
- [ ] Predict with trained models

Usage
------------------

The package so far supports Numpy arrays as inputs and dask-based parallelization.

The core functionality of omigami is represented by the `FeatureSelector` class.
This class takes few parameters as input and can be used to select the most important
features from the input dataset. The workflow of the feature selection is the following:

- `n_outer` CV splits are created
- A recursive feature selection is performed and evaluated on every CV split

The recursive feature elimination is also cross validated, so for each outer split:

- `n_inner` CV splits are created (default is `n_outer` - 1)
- for each CV split a model is trained. Performance and feature importances are extracted
- the least important features are discarded

This process is repeated until there are no more features in the model

Eventually, all the scores of the recursive model training are avergaed to get the number
of features for which the model performs the best.

The feature selector returns 3 possible feature sets:

- The minimum number of features for which the model performs well
- The maximum number of features for which the model performs well
- Their geometric mean

The whole process is repeated `n_repetitions` times to enhance selection robustness.

A minimal example
+++++++++++++++++
This is a minimal example to show the usage of the main omigami class.

Let's suppose that our data is composed of N samples for which M features have been
measured. Among these features, we want to select the best predictors for a target class.
The data should be stored in tabular data in two arrays: an array of predictors, `X`, with N rows and M columns,
containing the feature values, and a target array, `y`, containing the classes to predict.

For example, using `pandas` we would do something like

.. code-block:: python

        import pandas as pd
        data = pd.read_csv('test.csv')
        X = data.drop(columns=["target"]).values
        y = data["target"].values

Once the data is ready, we can instantiate the feature selector:

.. code-block:: python


        from omigami.omigami import FeatureSelector
        feature_selector = FeatureSelector(
            repetitions=10,
            n_outer=5,
            estimator="RFC",   # random forest classifier
            metric="MISS",   # missclassifications
        )

The `estimator` parameter denotes the model to be used for the feature elimination. So
far, the only native options supported are "RFC" and "XGBC" (gradient boost classifier),
but the class would also accept any scikit-learn model instance.
`metric` is the score to address the fitness of the model. In this
example we are using the number of missclassified samples. Other possibilities are
given by scikit-learn scores, such as "accuracy".

Fitting the selector is as easy as

.. code-block:: python

        feature_selector.fit(X, y)

It might take a while for it to complete, depending on your machine and on the model
selected.

Once the fit method is completed, selected features can be retrieved as

.. code-block:: python

        selected_features = feature_selector.selected_features

The features are reported as column indexes. To get the names just pass the selection
to the data frame:

.. code-block:: python

        selected_feature_names = data.columns[list(selected_features["min"])]

Parallelization
+++++++++++++++
The fit mthod can be time consuming, for this reason Omigami gives the option
to execute the various CV loops in parallel using a dask cluster.
The dask cluster can be remote, or running in local to exploit the processors of
the user's computer.
For the latter case - which is probably the most common case - it's sufficient to run the following
at the beginning of the script:

.. code-block:: python

        from dask.distributed import Client
        client = Client()

this will allow the user to inspect the status of the calculation at `http://localhost:8787/status`.

Citation
--------
Data Revenue, based on *Variable selection and validation in multivariate modelling (2019) [DOI:10.1093/bioinformatics/bty710](https://doi.org/10.1093/bioinformatics/bty710)*


License
--------
MIT license - free software.
