=======
omigami
=======


.. image:: https://img.shields.io/pypi/v/omigami.svg
        :target: https://pypi.python.org/pypi/omigami

..
    .. image:: https://img.shields.io/travis/datarevenue-berlin/omigami.svg
        :target: https://travis-ci.org/datarevenue-berlin/omigami

..
    .. image:: https://readthedocs.org/projects/omigami/badge/?version=latest
        :target: https://omigami.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Nested cross validation for feature selection in Python


* Free software: MIT license

..
     * Documentation: https://omigami.readthedocs.io.


Features
--------

Omigami is an opensource Python 3 software to perform multivariate feature selection
using a nested cross validation scheme. The original idea is formulated and tested
in *Shi L, Westerhuis JA, Rosén J, Landberg R, Brunius C. Variable selection and
validation in multivariate modelling. Bioinformatics. 2019 Mar 15;35(6):972-980.
doi: 10.1093/bioinformatics/bty710. PMID: 30165467; PMCID: PMC6419897.*
The package so far supports Numpy arrays as inputs and dask-based parallelization.


Install
-------

Omigami can be easily installed using PIP. It's recommendable to use a virtual
environment before performing the install. It can be done either using your python
release as

.. code-block:: bash

    python3 -m venv <path_to_venv>
    source <path_to_venv>/bin/activate

or using Anaconda

.. code-block:: bash

        conda create --name omigami_venv
        conda activate omigami_venv

When the virtual environment is installed, you can use PIP to install omigami:

.. code-block:: bash

        pip install omigami



How to use Omigami
------------------

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
