import os
import pathlib
import re
import string
import numpy as np
import pytest
import pandas as pd
from sklearn.datasets import make_classification

PYTHON_SNIPPET_REGEX = re.compile(r"```python([^`]+)```")


@pytest.fixture
def data():
    n_feats = 10
    n_samples = 10
    feature_names = [string.ascii_lowercase[i] for i in range(n_feats)] + ["target"]
    X, y = make_classification(n_feats, n_samples, random_state=0)
    data = pd.DataFrame(
        data=np.hstack((X, y.reshape((n_samples, 1)))), columns=feature_names
    )
    return data


@pytest.fixture
def readme_string():
    readme_dir = pathlib.Path(__file__).parent.parent.absolute()
    with open(os.path.join(readme_dir, "README.md")) as f:
        readme = f.read()
    return readme


@pytest.mark.skip(os.getenv("CI", "False").title())
@pytest.mark.slow
def test_readme_code_examples(data, readme_string):
    """this tests checks that every (almost) python snippet in the README
    would run sequentially without raising errors"""
    python_snippets = PYTHON_SNIPPET_REGEX.findall(readme_string)
    for snippet in python_snippets:
        if "read_csv" in snippet:  # the file is an unexisting example
            continue
        exec(snippet.strip())
