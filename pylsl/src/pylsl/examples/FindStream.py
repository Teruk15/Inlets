"""Example program to show how to read and find a specific LSL stream."""

from pylsl import StreamInlet, resolve_byprop


def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_byprop("type", "EEG", timeout=3.0)

    # Check all streams
    # for s in streams:
    #     print("name:", s.name())
    #     print("type:", s.type())
    #     print("source_id:", s.source_id())
    #     print("channel_count:", s.channel_count())
    #     print("nominal_srate:", s.nominal_srate())
    #     print("channel_format:", s.channel_format())
    #     print("---")


    for s in streams:
        if s.source_id() == 'HA-2016.03.01':    
            print('g.Hiamp Found!')
            print('Start collecting data...')
            inlet = StreamInlet(s)
            break
    else:
        raise RuntimeError('g.Hiamp was not found!')


    while True:
        # sample: 1-D python list
        sample, timestamp = inlet.pull_sample()
        print(timestamp, sample)


if __name__ == "__main__":
    main()
