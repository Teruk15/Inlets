from scipy.signal import iirnotch, butter, lfilter, lfilter_zi, resample_poly
from pylsl import StreamInlet, resolve_byprop
import numpy as np

class RateMonitor:
    """Tracks effective native sample rate from LSL timestamps and
    reports drift from the expected fs, mirroring the MATLAB ts check."""

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
        span = self.last_ts - self.first_ts
        if span <= 0 or self.n_samples < 2:
            return
        effective_fs = (self.n_samples - 1) / span
        off_hz = effective_fs - self.expected_fs
        off_ppm = (off_hz / self.expected_fs) * 1e6
        print(f"[rate check] effective fs = {effective_fs:.4f} Hz "
              f"({off_hz:+.4f} Hz, {off_ppm:+.1f} ppm off from {self.expected_fs} Hz)")

def process_chunk(chunk, zi_lp, zi_notch, zi_hp, b_notch, a_notch, b_hp, a_hp, b_lp, a_lp, ds_factor):
    """
    chunk: (n_samples, n_channels) at NATIVE fs — marker column already stripped
    Returns: filtered/CAR'd + decimated chunk, and updated filter states
    """

    # Anti-aliasing lowpass, at native fs (must run before decimation)
    lowpassed, zi_lp = lfilter(b_lp, a_lp, chunk, axis=0, zi=zi_lp)

    # Decimate via stride — safe because the lowpass above already band-limited
    decimated = lowpassed[::ds_factor, :]

    # Notch filter (designed at decimated rate)
    notched, zi_notch = lfilter(b_notch, a_notch, decimated, axis=0, zi=zi_notch)

    # Highpass filter (designed at decimated rate)
    highpassed, zi_hp = lfilter(b_hp, a_hp, notched, axis=0, zi=zi_hp)

    # CAR — stateless, spatial only
    car_ref = highpassed - highpassed.mean(axis=1, keepdims=True)

    return car_ref, zi_lp, zi_notch, zi_hp


# def decimate_markers(marker_chunk, ds_factor):
#     """
#     Downsample the marker channel by taking the max over each group of
#     ds_factor raw samples, so a single-sample trigger pulse is never lost
#     to a stride that happens to skip it (unlike marker[::ds_factor]).

#     NOTE: run rising-edge detection on the RAW marker channel before this,
#     if possible — decimating first still shortens the pulse's visibility
#     window. This function is kept as a fallback / for the decimated buffer
#     path only.
#     """
#     n = marker_chunk.shape[0]
#     n_trim = n - (n % ds_factor)
#     trimmed = marker_chunk[:n_trim]
#     return trimmed.reshape(-1, ds_factor).max(axis=1)


def find_rising_edges(marker_chunk, last_val):
    """
    marker_chunk: 1-D array of raw (native-rate) marker values for this chunk.
    last_val: last raw marker value carried forward from the previous chunk,
              so an edge that falls exactly on a chunk boundary is not missed.

    Returns: (edge_indices, new_last_val)
    edge_indices are indices INTO marker_chunk where a 0->1 (or below-thresh
    to at/above-thresh) transition occurs.
    """
    extended = np.concatenate(([last_val], marker_chunk))
    diffs = np.diff((extended > 0).astype(int))
    edge_indices = np.flatnonzero(diffs == 1)
    new_last_val = marker_chunk[-1] if marker_chunk.size > 0 else last_val
    return edge_indices, new_last_val
 

def is_enough(buffer, marker_idx, fs_ds):
    pre_time = 0.2
    post_time = 0.5
    pre_len = round(fs_ds * pre_time)
    post_len = round(fs_ds * post_time)

    buf_len = buffer.shape[0]

    if (marker_idx - pre_len) < 0 or (marker_idx + post_len) > buf_len:
        return False, None
    else:
        return True, buffer[marker_idx - pre_len:marker_idx + post_len, :].copy()


def main():
    # Define necessary variables
    fs = 4800
    ds_factor = 4
    n_channel = 96  # EXCLUDE marker channel
    erp_time = 0.7  # 700 ms

    fs_ds = fs // ds_factor

    # Anti-aliasing lowpass, designed at ORIGINAL fs, runs first
    lp_cutoff = 500.0  # comfortably below 600 Hz Nyquist at the decimated rate
    b_lp, a_lp = butter(3, lp_cutoff / (fs / 2), btype='lowpass')

    # Notch + Highpass, designed at the DECIMATED rate
    notch_freq = 60.0
    Q = 30.0
    b_notch, a_notch = iirnotch(notch_freq, Q, fs_ds)

    hp_cutoff = 0.5
    b_hp, a_hp = butter(3, hp_cutoff / (fs_ds / 2), btype='highpass')

    # Per-channel filter state (persists across chunks)
    zi_lp = np.tile(lfilter_zi(b_lp, a_lp), (n_channel, 1)).T
    zi_notch = np.tile(lfilter_zi(b_notch, a_notch), (n_channel, 1)).T
    zi_hp = np.tile(lfilter_zi(b_hp, a_hp), (n_channel, 1)).T

    # Find g.Hiamp and establish connection
    print("looking for an EEG stream...")
    streams = resolve_byprop("type", "EEG", timeout=3.0)

    for s in streams:
        if s.source_id() == 'HA-2016.03.01':
            print('g.Hiamp Found!')
            print('Start collecting data...')
            inlet = StreamInlet(s)
            break
    else:
        raise RuntimeError('g.Hiamp was not found!')

    # Begin Processing
    chunk_size_ds = round(fs_ds * erp_time)
    pull_size_native = chunk_size_ds * ds_factor

    buffer = np.zeros((2 * chunk_size_ds, n_channel))
    half = chunk_size_ds
    end = buffer.shape[0]
    pending = False
    marker_idx = 0
    last_marker_val = 0.0

    erps = []  # collected epochs, since erp was being overwritten in-place before

    rate_monitor = RateMonitor(expected_fs=fs) 

    while True:
        chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=pull_size_native)

        if len(chunk) == 0:
            print("empty pull...")
            continue

        rate_monitor.update(timestamps)

        chunk = np.array(chunk, dtype=np.float64)
        print(f"got chunk: {chunk.shape}")

        if chunk.shape[1] - 1 != n_channel:
            print(f"WARNING: expected {n_channel} EEG channels, got {chunk.shape[1]-1}")

        eeg_chunk = chunk[:, :-1]
        marker_chunk_raw = chunk[:, -1]

        eeg_chunk, zi_lp, zi_notch, zi_hp = process_chunk(
            eeg_chunk, zi_lp, zi_notch, zi_hp,
            b_notch, a_notch, b_hp, a_hp, b_lp, a_lp, ds_factor
        )
        print(f"downsampled chunk: {eeg_chunk.shape}")

        # Rising-edge detection on the RAW marker channel, before decimation,
        # so a single-sample pulse can't be lost to the decimation stride.
        edge_indices_native, last_marker_val = find_rising_edges(marker_chunk_raw, last_marker_val)
        edge_indices_ds = edge_indices_native // ds_factor

        # eeg_chunk is now at the decimated rate and should be chunk_size_ds long
        n_new = eeg_chunk.shape[0]
        if n_new != chunk_size_ds:
            # Defensive check: LSL pull sizes can vary slightly chunk to chunk.
            # Truncate/pad as needed rather than letting the buffer assignment
            # below throw on a shape mismatch. Simplest robust approach:
            # resize buffer's second half to match what actually arrived.
            n_new = min(n_new, half)
            eeg_chunk = eeg_chunk[:n_new, :]

        buffer[half:half + n_new, :] = eeg_chunk

        if not pending and edge_indices_ds.size > 0:
            # print(f"trigger detected at native idx {edge_indices_native}")
            marker_idx = int(edge_indices_ds[0]) + half
            pending = True

        if pending:
            enough, epoch = is_enough(buffer, marker_idx, fs_ds)

            if enough:
                erps.append(epoch)
                # print(f"epoch #{len(erps)} collected")
                pending = False
            else:
                marker_idx -= half

        buffer[0:half, :] = buffer[half:end, :]

    # return erps  # (unreachable while True: loop — add a break condition
    #               #  or run this as a generator/callback in real use)


if __name__ == "__main__":
    main()