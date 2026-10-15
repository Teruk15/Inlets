class RateMonitor:
    """
    """

    def __init__(self, expected_fs, report_every=4800):  # ~every 1s @ 4800Hz
        self.expected_fs = expected_fs
        self.report_every = report_every
        self.first_ts = None
        self.last_ts = None
        self.n_samples = 0

    def update(self, timestamps):
        if len(timestamps) == 0:
            return
        if self.first_ts is None:
            self.first_ts = timestamps[0]

        self.last_ts = timestamps[-1]
        self.n_samples += len(timestamps)

        if self.n_samples % self.report_every < len(timestamps):
            self.report()


    def report(self):
        """
        """
        span = self.last_ts - self.first_ts
        if span <= 0 or self.n_samples < 2:
            return
        effective_fs = (self.n_samples - 1) / span
        off_hz = effective_fs - self.expected_fs
        off_ppm = (off_hz / self.expected_fs) * 1e6
        print(f"[rate check] effective fs = {effective_fs:.4f} Hz "
              f"({off_hz:+.4f} Hz, {off_ppm:+.1f} ppm off from {self.expected_fs} Hz)")