import time
from datetime import datetime, timedelta


class PerformanceTimer:
    """Context manager class for measuring the CPU execution time of a code block.
    Class is using time.perf_counter() function to measure time under the hood.

    Example:
    >>> with PerformanceTimer() as timer:
    ...     time.sleep(0.5)
    >>> print(timer)
    """

    def __init__(self) -> None:
        self.start_time: float = None
        self.end_time: float = None
        self._time: float = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        self.end_time = time.perf_counter()
        self._time = self.end_time - self.start_time

    def __str__(self) -> str:
        return str(self.timedelta)

    @property
    def time(self) -> float:
        """
        Returns:
            float: time in seconds
        """
        return self._time

    @property
    def timedelta(self) -> timedelta:
        """
        Returns:
            timedelta:  datetime timedelta object
        """
        start_datatime = datetime.fromtimestamp(self.start_time)
        end_datatime = datetime.fromtimestamp(self.end_time)
        return end_datatime - start_datatime
