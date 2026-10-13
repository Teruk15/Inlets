class ChunkFilter:
    """
    """
    
    def __init__(self, n_channel, fs, fs_ds, ds_factor,
                 lp_cutoff=500.0, notch_freq=60.0, Q=30.0, hp_cutoff=0.1):
        self.ds_factor = ds_factor
    
    def process_chunk(self, chunk):
        """
        chunk: (n_samples, n_channels) at NATIVE fs — marker column already stripped
        Returns: filtered/CAR'd + decimated chunk, and updated filter states
        """

        # Anti-aliasing lowpass, at native fs (must run before decimation)
        lowpassed, zi_lp = lfilter(b_lp, a_lp, chunk, axis=0, zi=zi_lp)

        # Decimate via stride — safe because the lowpass above already band-limited
        decimated = lowpassed[::self.ds_factor, :]

        # Notch filter (designed at decimated rate)
        notched, zi_notch = lfilter(b_notch, a_notch, decimated, axis=0, zi=zi_notch)

        # Highpass filter (designed at decimated rate)
        highpassed, zi_hp = lfilter(b_hp, a_hp, notched, axis=0, zi=zi_hp)

        # CAR — stateless, spatial only
        car_ref = highpassed - highpassed.mean(axis=1, keepdims=True)

        return car_ref, zi_lp, zi_notch, zi_hp