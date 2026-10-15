import numpy as np

class ERSPCalculator:
    """
    """
 
    def __init__(self, n_channel, sf_eff,
                 low_freq=70.0, high_freq=150.0, step_freq=5.0,
                 baseline_begin=-0.2, baseline_end=-0.1,
                 pre_time=0.3, crop_time=0.1):
        
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
        """
        """
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