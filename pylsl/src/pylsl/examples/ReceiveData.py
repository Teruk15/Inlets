"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_byprop


def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_byprop("type", "EEG", timeout=3.0)
    for s in streams:
        print("name:", s.name())
        print("type:", s.type())
        print("source_id:", s.source_id())
        print("channel_count:", s.channel_count())
        print("nominal_srate:", s.nominal_srate())
        print("channel_format:", s.channel_format())
        print("---")

    # # create a new inlet to read from the stream
    # inlet = StreamInlet(streams[0])

    # while True:
    #     # get a new sample (you can also omit the timestamp part if you're not
    #     # interested in it)
    #     sample, timestamp = inlet.pull_sample()
    #     print(timestamp, sample)


if __name__ == "__main__":
    main()
