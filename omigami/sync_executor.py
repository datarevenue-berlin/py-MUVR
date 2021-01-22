from concurrent.futures import Future, Executor


class SyncExecutor(Executor):
    """An executor which immediately invokes all submitted callables."""

    def submit(self, fn, *args, **kwargs):  # pylint: disable=arguments-differ
        """Immediately invokes `fn(*args, **kwargs)` and returns a future
        with the result (or exception)."""
        future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:  # pylint: disable=broad-except
            future.set_exception(e)
        return future
