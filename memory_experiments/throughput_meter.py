import time

class ThroughputMeter:
    def __init__(self):
        self.start_time = None
        self.token_count = 0

    def start(self):
        self.start_time = time.perf_counter()
        self.token_count = 0

    def update(self, num_tokens):
        self.token_count += num_tokens

    def compute(self):
        elapsed = time.perf_counter() - self.start_time if self.start_time is not None else 0
        return self.token_count / elapsed if elapsed > 0 else 0.0
