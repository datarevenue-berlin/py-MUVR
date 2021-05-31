#### Minimally biased features selection for omics studies

[![PyPI version shields.io](https://img.shields.io/pypi/v/omigami.svg)](https://pypi.python.org/pypi/omigami) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- ▶️ [Example Jupyter Notebook](https://github.com/datarevenue-berlin/omigami/blob/main/notebooks/MinimalExample.ipynb)
- 📑 [Blog Post](https://datarevenue.com/en-blog/minimally-biased-feature-selection-untargeted-metabolomics)
- 🎓 [Paper](https://doi.org/10.1093/bioinformatics/bty710)

<!-- image:: https://img.shields.io/travis/datarevenue-berlin/omigami.svg :target: https://travis-ci.org/datarevenue-berlin/omigami -->

<!-- image:: https://readthedocs.org/projects/omigami/badge/?version=latest :target: https://omigami.readthedocs.io/en/latest/?badge=latest :alt: Documentation Status -->


Multivariate recursive feature elimination within a repeated **double cross-validation** protects you against overfitting – and drastically reduces the number of false positive features in your results.

## Installation

```sh
pip install py_muvr
```

## Acknowledgement

This package is based on an algorithm first introduced by Carl Brunius in [Variable selection and validation in multivariate modelling (2019)](https://academic.oup.com/bioinformatics/article/35/6/972/5085367).

**Citation**: Data Revenue, based on *Variable selection and validation in multivariate modelling (2019) [DOI:10.1093/bioinformatics/bty710](https://doi.org/10.1093/bioinformatics/bty710)*

## Motivation

- **Omics studies produce too many false positives**: It's hard to protect against selection bias on high-dimensional omics data ([Krawczuk and Łukaszuk, 2016](https://www.sciencedirect.com/science/article/pii/S0933365715001426)). Even common cross-validation has been shown to overfit.
- **Redundant features are important for biological interpretation**: Most feature selection techniques focus on finding the minimal set of strongest features. Omitting redundant variables that are however still relevant to understanding the biochemical systems.
- **Easy-to-use tool**: There's no freely available and easy-to-use Python tool that implements a minimally biased repeated double cross validation.
- **Small runtime**: A robust selection requires many (100 - 5.000) models to be trained. Running such a large number of models in reasonable time, requires non-trivial parallelization.

## Features

- [x] Repeated double cross-validation
- [x] Multivariate feature selection (Random Forest, XGB or PLS-DA)
- [x] Minimal optimal and all relevant feature selection
- [x] Efficient Parallelization (with Dask)
- [x] Familiar scikit-learn API
- [x] Plotting
- [ ] Predict with trained models

## Usage

### A minimal example

- **test.csv**: This is your omics dataset.
- **target**: Replace this with the name of the column that denotes your class variable
  - e.g. this column will contain (1/0, pathological/control, treatment/non-treatment, etc.)

```python
import pandas as pd
data = pd.read_csv('test.csv')
```
from the data, numpy arrays have to be extracted:

```python
X = data.drop(columns=["target"]).values
y = data["target"].values
```

Once the data is ready, we can get a feature selector, fit it and look at the selected features:

```python
from py_muvr.feature_selector import FeatureSelector

feature_selector = FeatureSelector(
    n_repetitions=10,
    n_outer=5,
    n_inner=4,
    estimator="PLSC",   # partial least squares classifier
    metric="MISS",   # missclassifications
)

feature_selector.fit(X, y)

feature_names = data.drop(columns=["target"]).columns
selected_features = feature_selector.get_selected_features(feature_names=feature_names)
```

It might take a while for it to complete, depending on your machine and on the model selected.

### Selecting min, max and mid feature sets from `selected_features`

The feature selector returns 3 possible feature sets that can be inspected as:

```python
min_feats = selected_features["min"]
mid_feats = selected_features["mid"]
max_feats = selected_features["max"]
```

- **`min_feats`**: The minimum number of features for which the model performs optimally.
  - The minimal set of most informative features. If you choose less features, then the model will perform worse.
- **`max_feats`**: The maximum number of features for which the model performs optimally.
  - The all-relevant feature set. This includes also all weak and redundant, but still relevant features – without including noisy and uninformative features. Using more features would also decrease the performance of the model.
- **`mid_feats`**: The geometric mean of both feature sets.

### Parallelization

The feature selection can be time consuming. To speed it up, Py-MUVR gives the option of executing the various CV loops
in parallel using an [Executor object](https://docs.python.org/3/library/concurrent.futures.html) which should be passed
as keyword parameter to the fit method.

So far, [dask](https://distributed.readthedocs.io/en/1.10.2/executor.html),
[loky](https://loky.readthedocs.io/en/stable/>) (joblib)
and [concurrent](https://docs.python.org/3/library/concurrent.futures.html) executors have been tested.

For example, using the native Python3 `concurrent` library, you would do:

```python
from concurrent.futures import ProcessPoolExecutor
executor = ProcessPoolExecutor()

feature_selector.fit(X, y, executor=executor)
```
Note that you need to pass the `executor` to the `fit()` method.

Another example with Dask would be

```python
from dask.distributed import Client
client = Client()
executor = client.get_executor()

feature_selector.fit(X, y, executor=executor)
```

*Also*: Dask gives you a neat dashboard to see the status of all the jobs at `http://localhost:8787/status`.

## How it works

![Schematic of repeated double cross-validation](https://global-uploads.webflow.com/5d3ec351b1eba4332d213004/5fd0e88651b733b656c3603b_ccukNlmNckEJ3p-Z9fHm2jPdgI9ILBDbOcdOBaBz3_WXA7VltferIk3vU1PPHztX5Gjcr0DMbh2xtvEK1lSYdou2xGAtni-Mq50W_cEpXssg2akHefa-H41jKDApZxctJlnVvk-b.png)

1. The dataset is split into `n_outer` cross-validation splits.
2. Each train split is further split into `n_inner` cross-validation splits.
3. On each cross-validation split multivariate models are trained and evaluated.
4. The least important fraction of features (`features_dropout_rate`) is removed, until there are no more features in the model
5. The whole process is repeated `n_repetitions` times to improve the robustness of the selection.
6. Feature ranks are averaged over all `n_outer` splits and all `n_repetitions`.

## Permutation Test

To test the significance of the selected features, Py-MUVR implements as class to perform a permutation test for the
feature selection

```python
from py_muvr.permutation_test import PermutationTest

permutation_test = PermutationTest(feature_selector, n_permutations=10)
permutation_test.fit(X, y)
p_value = permutation_test.compute_p_values("min")
print("p-value of the 'min' feature set: %s" % p_value)

```

## Visualization

Py-MUVR provides some basic plotting utils to inspect the results of the feature selection. In particular, it provides
two main methods:

- `plot_feature_rank`
- `plot_validation_curves`
- `plot_permutation_scores`

```python
from py_muvr.plot_utils import plot_feature_rank

feature_selection_results = feature_selector.get_feature_selection_results(feature_names)
fig = plot_feature_rank(
    feature_selection_results,
    model="min",  # one of "min", "mid" or "max"
    feature_names=feature_names  # optional
)
```

```python
from py_muvr.plot_utils import plot_validation_curves

fig = plot_validation_curves(feature_selection_results)
```

and

```python
from py_muvr.plot_utils import plot_permutation_scores

fig = plot_permutation_scores(permutation_test, "min")
```

## Parameters

### `FeatureSelector` parameters

- **n_repetitions**: Number of repetitions of the entire double cross-validation loop (default: `8`)
- **n_outer**: Number of cross-validation splits in the outer loop
- **n_inner**: Number of cross-validation splits in the inner loop (default: n_outer-1)
- **estimator**: Multivariate model that you want to use for the feature selection. Supports
  - `"RFC"`: Random Forest Classifier
  - `"XGBC"`: XGBoost Classifier
  - `"PLSC"`: Partial Least Square Classifier
  - `"PLSR"`: Partial Least Square Regressor
  - scikit-learn model and pipeline instances
- **metric**: Metric to be used to assess fitness of estimators. Supports
  - `"MISS"`: Number of missclassifications.
  - several classification and regression scores from [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html) (refer to documentation)
  - custom functions
- **features_dropout_rate**: Fraction of features that will be dropped in each elimination step (float)
- **robust_minimum** (float): Maximum normalized-score value to be considered when computing the selected features
- **random_state** (int): Pass an int for a reproducible output (default: `None`)

## Contribute to Py-MUVR

1. Fork it (https://github.com/datarevenue-berlin/omigami/fork)
2. Create your feature branch (git checkout -b feature/fooBar)
3. Commit your changes (git commit -am 'Add some fooBar')
4. Push to the branch (git push origin feature/fooBar)
5. Create a new Pull Request

## License
MIT license - free software.
