classdef LSLInlet < matlab.System
    properties
        StreamType = 'EEG'
        SourceID = 'HA-2016.03.01';
        NumChannels = 97
        Fs = 4800
    end
    properties (Access = private)
        lib
        inlet
    end
    methods (Access = protected)
        function setupImpl(obj)
            obj.lib = lsl_loadlib();
            results = lsl_resolve_byprop(obj.lib, 'type', obj.StreamType, 3.0);
            matched_stream = [];
            for i = 1:length(results)
                info = results{i};
                if strcmp(info.type(), obj.StreamType) && strcmp(info.source_id(), obj.SourceID)
                    matched_stream = info;
                    disp('g.Hiamp found!');
                    disp('Start collecting data...');
                    break;
                end
            end
            if isempty(matched_stream)
                error('LSLInlet:StreamNotFound', ...
                    'No stream found with type "%s" and source_id "%s".', ...
                    obj.StreamType, obj.SourceID);
            end

            actualChannels = matched_stream.channel_count();
            if actualChannels ~= obj.NumChannels
                error('LSLInlet:ChannelMismatch', ...
                    'Stream reports %d channels but NumChannels property is set to %d. Update the block parameter.', ...
                    actualChannels, obj.NumChannels);
            end

            obj.inlet = lsl_inlet(matched_stream);
        end

        function [ts, data] = stepImpl(obj)
            timeout = 0.1;
            [vec, ts_raw] = obj.inlet.pull_sample(timeout);

            if isempty(vec)
                ts = 0;
                data = zeros(1, obj.NumChannels);
            else
                ts = ts_raw;
                data = vec(:)';
            end
        end

        function [s1, s2] = getOutputSizeImpl(obj)
            s1 = [1, 1];                  % ts port
            s2 = [1, obj.NumChannels];    % data port
        end

        function [t1, t2] = getOutputDataTypeImpl(~)
            t1 = 'double';
            t2 = 'double';
        end

        function [c1, c2] = isOutputComplexImpl(~)
            c1 = false;
            c2 = false;
        end

        function [f1, f2] = isOutputFixedSizeImpl(~)
            f1 = true;
            f2 = true;
        end

        function sts = getSampleTimeImpl(obj)
            sts = createSampleTime(obj, 'Type', 'Discrete', ...
                'SampleTime', 1/obj.Fs, 'OffsetTime', 0);
        end
    end
end