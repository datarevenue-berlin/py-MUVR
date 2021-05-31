from py_muvr.data_structures import SelectedFeatures


def test_n_features(dataset):
    assert dataset.n_features == 12


def test_input_data_slice(dataset):
    assert dataset[:5, 3:7].X.shape == (5, 4)
    assert dataset[[1, 2, 5], [3, 4, 7]].X.shape == (3, 3)
    assert dataset[1:3, :].X.shape == (2, 12)
    assert dataset[:, :].X.shape == dataset.X.shape


def test_selected_features():
    sf = SelectedFeatures(min="min", mid="mid", max="max")

    assert sf["mid"]
