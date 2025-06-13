"""Timer context manager"""

import time


class Timer:
    def __init__(self, metrics_dict, key_name):
        self.metrics = metrics_dict
        self.key = key_name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.metrics[self.key] = elapsed_time
