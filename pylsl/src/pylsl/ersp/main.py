from pylsl import StreamInlet, resolve_byprop
from preprocess import ChunkFilter
from ersp import ERSPCalculator
from plot import ERSPGridPlotter
from rate import RateMonitor
import numpy as np


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
 

def is_enough(buffer, marker_idx, fs_ds, pre_time, post_time):
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
    n_channel = 21  # EXCLUDE marker channel
    erp_time = 0.9  # 900 ms

    pre_time = 0.3  
    post_time = 0.6 
    crop_time = 0.1  # ERSPCalculator default

    fs_ds = fs // ds_factor

    lp_cutoff = 500.0 # Anti-aliasing lowpass, designed at ORIGINAL fs, runs first

    notch_freq = 60.0 # Notch filter, designed at DECIMATED fs
    Q = 30.0

    hp_cutoff = 0.1 # Highpass filter, designed at DECIMATED fs


    # Find g.Hiamp and establish connection
    print("looking for an EEG stream...")
    streams = resolve_byprop("type", "eeg", timeout=3.0)

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

    # Initialize the Chunk Filter
    chunk_filter = ChunkFilter(
        n_channel=n_channel, fs=fs, fs_ds=fs_ds, ds_factor=ds_factor,
        lp_cutoff=lp_cutoff, notch_freq=notch_freq, Q=Q, hp_cutoff=hp_cutoff
    )

    # Initialize the Rate Monitor
    rate_monitor = RateMonitor(expected_fs=fs, report_every=fs)

    # Initialize the ERSP calculator  
    ersp_calc = ERSPCalculator(
        n_channel=n_channel, sf_eff=fs_ds, pre_time=pre_time, crop_time=crop_time
    )

    # Initialize the ERSP plotter
    n_crop_samples = round(crop_time * fs_ds)
    n_data_samples = round(erp_time  * fs_ds) - 2 * n_crop_samples
    plotter = ERSPGridPlotter(
        n_channel=n_channel, n_rows=8, n_cols=12, sf_eff=fs_ds,
        pre_time=pre_time, crop_time=crop_time, n_samples_crop=n_crop_samples, n_samples_data=n_data_samples,
        ylim=(-0.5, 0.5),      # fix ylim to skip autoscale entirely (cheapest);
                               # set to None to autoscale instead (heavier)
        autoscale_every=5,
        title="Live high-gamma ERSP (all channels)"
    )

    # Main Loop: Pull chunks, filter, downsample, detect triggers, extract epochs, update ERSP
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

        # Divide channel into EEG and Marker channels.
        eeg_chunk = chunk[:, :-1]
        marker_chunk_raw = chunk[:, -1]

        # Process the EEG chunk through the filter pipeline
        eeg_chunk = chunk_filter.process_chunk(eeg_chunk)
        print(f"downsampled chunk: {eeg_chunk.shape}")

        # Rising-edge detection on the RAW marker channel, then downsampling
        edge_indices_native, last_marker_val = find_rising_edges(marker_chunk_raw, last_marker_val)
        edge_indices_ds = edge_indices_native // ds_factor

        # Sanity check if EEG downsampled correctly
        n_new = eeg_chunk.shape[0]
        if n_new != chunk_size_ds:
            n_new = min(n_new, half)
            eeg_chunk = eeg_chunk[:n_new, :]

        # Update the buffer with the new chunk
        buffer[half:half + n_new, :] = eeg_chunk

        if not pending and edge_indices_ds.size > 0:
            print(f"trigger detected at native idx {edge_indices_native}")
            marker_idx = int(edge_indices_ds[0]) + half
            pending = True

        if pending:
            enough, epoch = is_enough(buffer, marker_idx, fs_ds, pre_time, post_time)

            if enough:
                erps.append(epoch)
                print(f"epoch #{len(erps)} collected")
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
