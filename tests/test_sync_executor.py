import pytest
from concurrent.futures import Future
from py_muvr.sync_executor import SyncExecutor


def square(x):
    return x ** 2


def bad_function(x):
    assert False


def test_sync_executor():
    executor = SyncExecutor()
    future = executor.submit(square, 2)
    assert isinstance(future, Future)
    assert future.result() == 4


def test_sync_executor_error():
    executor = SyncExecutor()
    future = executor.submit(bad_function, 2)
    assert isinstance(future, Future)
    with pytest.raises(AssertionError):
        future.result()
    isinstance(future.exception(), AssertionError)
