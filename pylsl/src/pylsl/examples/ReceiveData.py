from scipy.signal import iirnotch, butter, lfilter, lfilter_zi, resample_poly
from pylsl import StreamInlet, resolve_byprop
import matplotlib.pyplot as plt
import numpy as np

class ERSPCalculator:
    """
    Port of MATLAB calculateERP: baseline-corrected high-gamma power
    for one ERP epoch via FFT-based Morlet wavelet convolution.
 
    Single stim_type only (no per-condition tracking, no SEM).
    Call update(epoch) once per completed trial epoch.
 
    epoch: (n_channel, n_samples) at the DECIMATED rate (sf_eff) — already
           excludes the marker channel and is NOT re-decimated here.
    """
 
    def __init__(self, n_channel, sf_eff,
                 low_freq=70.0, high_freq=150.0, step_freq=5.0,
                 baseline_begin=-0.2, baseline_end=-0.1,
                 pre_time=0.2, crop_time=0.1):
        
        self.n_channel = n_channel
        self.sf_eff = sf_eff
        self.pre_time = pre_time
        self.baseline_begin = baseline_begin
        self.baseline_end = baseline_end
        self.crop_time = crop_time
 
        self.count = 0
        self.cum_erp = None
 
        self.freqs = np.arange(low_freq, high_freq + 1e-9, step_freq)
        self.cycles = np.linspace(10, 20, self.freqs.size)
 
        self._wavefft = None # (n_freq, n_conv_pow2)
        self._n_wavelet = None
        self._n_convolution = None
        self._n_conv_pow2 = None
        self._n_wavelet_half = None
 
    
    def build_wavelets(self, n_data):
        s = self.cycles / (2 * np.pi * self.freqs)
        max_s = s.max()
        time = np.arange(-5 * max_s, 5 * max_s + 1 / self.sf_eff, 1 / self.sf_eff)
 
        sinosoid_exp = (2j * np.pi * self.freqs[:, None]) * time[None, :]
        gaussian_exp = -(time[None, :] ** 2) / (2 * (s[:, None] ** 2))
        wavelets = np.exp(sinosoid_exp) * np.exp(gaussian_exp)   # (n_freq, n_wavelet)
 
        n_wavelet = time.size
        n_convolution = n_wavelet + n_data - 1
        n_conv_pow2 = 1 << (n_convolution - 1).bit_length()
 
        wavefft = np.fft.fft(wavelets, n=n_conv_pow2, axis=1)
        max_mag = np.abs(wavefft).max(axis=1, keepdims=True)
        wavefft = wavefft / max_mag
 
        self._wavefft = wavefft # (n_freq, n_conv_pow2)
        self._n_wavelet = n_wavelet
        self._n_convolution = n_convolution
        self._n_conv_pow2 = n_conv_pow2
        self._n_wavelet_half = (n_wavelet - 1) // 2
 
    
    def update(self, epoch):
        """
        epoch: (n_channel, n_samples) single-trial data.
        Returns erp_out_crop: (n_channel, n_samples_cropped) — running
        grand-average, baseline-corrected high-gamma power.
        """
        n_channel, n_data = epoch.shape
        assert n_channel == self.n_channel

        # Same as Init()
        if self._wavefft is None:
            self.build_wavelets(n_data)
        if self.cum_erp is None:
            self.cum_erp = np.zeros((self.n_channel, n_data))
 
        # 1) Wavelet TF decomposition
        eeg_fft = np.fft.fft(epoch, n=self._n_conv_pow2, axis=1)         # (ch, n_conv_pow2)
        conv = np.fft.ifft(
            self._wavefft[:, None, :] * eeg_fft[None, :, :], axis=2
        )                                                                # (freq, ch, n_conv_pow2)
 
        conv = conv[:, :, :self._n_convolution]
        h = self._n_wavelet_half
        conv = conv[:, :, h:h + n_data]                                  # (freq, ch, n_data)
 
        power_trial = np.abs(conv) ** 2                                  # (freq, ch, n_data)
 
        # 2) Baseline correction per frequency (VSSUM, then re-center)
        bl_begin = int(np.floor((self.pre_time + self.baseline_begin) * self.sf_eff))
        bl_end = int(np.floor((self.pre_time + self.baseline_end) * self.sf_eff))
 
        pow_base = power_trial[:, :, bl_begin:bl_end + 1].mean(axis=2, keepdims=True)
        vssum = (power_trial - pow_base) / (power_trial + pow_base)
 
        baseline = vssum[:, :, bl_begin:bl_end + 1].mean(axis=2, keepdims=True)
        corrected = vssum - baseline                                     # (freq, ch, n_data)
 
        hg_power = corrected.mean(axis=0)                                # (ch, n_data)
 
        # 3) Running grand average across trials
        self.cum_erp += hg_power
        self.count += 1
        erp_out = self.cum_erp / self.count                              # (ch, n_data)
 
        n_cut = round(self.crop_time * self.sf_eff)
        erp_out_crop = erp_out[:, n_cut:n_data - n_cut]
 
        return erp_out_crop

class ERSPGridPlotter:
    """
    One subplot per channel, laid out like an electrode grid.
    Reuses persistent Line2D objects (set_ydata) instead of re-plotting,
    so each frame only touches data, not figure structure.
    Full autoscale + canvas redraw is throttled (autoscale_every) since
    that's the expensive part, not the line update itself.
    """
 
    def __init__(self, n_channel, n_rows, n_cols, sf_eff,
                 pre_time, crop_time, n_samples_crop,
                 ylim=None, autoscale_every=5, title="Live ERSP"):
        assert n_rows * n_cols >= n_channel, "grid too small for n_channel"
        self.n_channel = n_channel
        self.sf_eff = sf_eff
        self.ylim = ylim
        self.autoscale_every = autoscale_every
        self._n_updates = 0
 
        # Time axis is fixed by config (pre_time/crop_time/sf_eff never change
        # between calls), so build it once here instead of every update().
        n_cut = round(crop_time * sf_eff)
        self.t = (np.arange(n_samples_crop) + n_cut) / sf_eff - pre_time
 
        plt.ion()
        self.fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.0),
            sharex=True, sharey=(ylim is not None)
        )
        self.axes = np.atleast_2d(axes).ravel()
        self.fig.suptitle(title)
 
        self.lines = []
        for ch in range(n_channel):
            ax = self.axes[ch]
            (line,) = ax.plot(self.t, np.zeros_like(self.t), linewidth=0.8)
            ax.axvline(0.0, linewidth=0.5, alpha=0.4)  # stim onset marker
            ax.set_xlim(self.t[0], self.t[-1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(ch), fontsize=6, pad=1)
            if ylim is not None:
                ax.set_ylim(*ylim)
            self.lines.append(line)
 
        for ax in self.axes[n_channel:]:
            ax.axis("off")
 
        self.fig.tight_layout()
        plt.show(block=False)
 
    def update(self, erp_out_crop):
        """
        erp_out_crop: (n_channel, n_samples_cropped) from ERSPCalculator.update()
        """
        n_ch, n_samp = erp_out_crop.shape
        assert n_samp == self.t.size, "n_samples_crop mismatch with constructor"
 
        for ch in range(n_ch):
            self.lines[ch].set_ydata(erp_out_crop[ch])
 
        self._n_updates += 1
        do_autoscale = (self.ylim is None) and (self._n_updates % self.autoscale_every == 0)
        if do_autoscale:
            for ax in self.axes[:n_ch]:
                ax.relim()
                ax.autoscale_view()
 
        # draw_idle + flush_events: lets the GUI backend coalesce redraws
        # rather than forcing a full synchronous draw every call
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
            
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
    pre_time = 0.3
    post_time = 0.6
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
    erp_time = 0.9  # 900 ms

    pre_time = 0.3   # must match is_enough()'s pre_time
    crop_time = 0.1  # ERSPCalculator default

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
    
    ersp_calc = ERSPCalculator(
        n_channel=n_channel, sf_eff=fs_ds, pre_time=pre_time, crop_time=crop_time
    )
    # ERSPCalculator crops crop_time off each end of the (pre+post) epoch;
    # compute that final sample count once, up front, so the plotter can
    # build its time axis at construction instead of on first update().
    n_epoch_samples = round(fs_ds * erp_time)
    n_cut = round(crop_time * fs_ds)
    n_samples_crop = n_epoch_samples - 2 * n_cut
 
    plotter = ERSPGridPlotter(
        n_channel=n_channel, n_rows=8, n_cols=12, sf_eff=fs_ds,
        pre_time=pre_time, crop_time=crop_time, n_samples_crop=n_samples_crop,
        ylim=(-0.5, 0.5),      # fix ylim to skip autoscale entirely (cheapest);
                               # set to None to autoscale (throttled) instead
        autoscale_every=5,
        title="Live high-gamma ERSP (all channels)"
    )

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
                # erps.append(epoch)
                # print(f"epoch #{len(erps)} collected")
                pending = False
                
                erp_out_crop = ersp_calc.update(epoch.T)
                plotter.update(erp_out_crop)
                print(f"epoch #{len(erps)} -> ERSP updated")
            else:
                marker_idx -= half

        buffer[0:half, :] = buffer[half:end, :]

    # return erps  # (unreachable while True: loop — add a break condition
    #               #  or run this as a generator/callback in real use)


if __name__ == "__main__":
    main()
