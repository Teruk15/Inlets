from mne_lsl.player import PlayerLSL
import mne
import time

raw = mne.io.read_raw_eeglab(r"mne-lsl\example\0452SF_AN.set", preload=True)

player = PlayerLSL(
    raw,
    chunk_size=8,       # match g.HIamp's ~8-sample burst delivery
    name="g.HIamp",
    source_id="HA-2016.03.01",
    n_repeat=1          # end the run once reach to end of datapoint
)
player.start()

try:
    while player.running:
        time.sleep(1)
except KeyboardInterrupt:
    player.stop()