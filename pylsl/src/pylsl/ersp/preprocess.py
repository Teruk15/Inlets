from scipy.signal import iirnotch, butter, lfilter, lfilter_zi
import numpy as np

class ChunkFilter:
    """
    """
    
    def __init__(self, n_channel, fs, fs_ds, ds_factor,
                 lp_cutoff, notch_freq, Q, hp_cutoff):
        self.ds_factor = ds_factor

        self.b_lp,    self.a_lp    = butter(3, lp_cutoff / (fs / 2), btype='lowpass')
        self.b_notch, self.a_notch = iirnotch(notch_freq, Q, fs_ds)
        self.b_hp,    self.a_hp    = butter(3, hp_cutoff / (fs_ds / 2), btype='highpass')

        # Per-channel filter state (persists across chunks)
        self.zi_lp    = np.tile(lfilter_zi(self.b_lp, self.a_lp), (n_channel, 1)).T
        self.zi_notch = np.tile(lfilter_zi(self.b_notch, self.a_notch), (n_channel, 1)).T
        self.zi_hp    = np.tile(lfilter_zi(self.b_hp, self.a_hp), (n_channel, 1)).T


    def process_chunk(self, chunk):
        """
        """

        # Anti-aliasing lowpass, at native fs (must run before decimation)
        lowpassed, self.zi_lp = lfilter(self.b_lp, self.a_lp, chunk, axis=0, zi=self.zi_lp)

        # Decimate via stride — safe because the lowpass above already band-limited
        decimated = lowpassed[::self.ds_factor, :]

        # Notch filter (designed at decimated rate)
        notched, self.zi_notch = lfilter(self.b_notch, self.a_notch, decimated, axis=0, zi=self.zi_notch)

        # Highpass filter (designed at decimated rate)
        highpassed, self.zi_hp = lfilter(self.b_hp, self.a_hp, notched, axis=0, zi=self.zi_hp)

        # CAR — stateless, spatial only
        car_ref = highpassed - highpassed.mean(axis=1, keepdims=True)

        return car_ref
    