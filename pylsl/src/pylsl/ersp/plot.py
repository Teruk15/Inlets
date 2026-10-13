import matplotlib.pyplot as plt
import numpy as np

class ERSPGridPlotter:
    """
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