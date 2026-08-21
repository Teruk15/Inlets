classdef LSLInlet < matlab.System
    properties
        StreamType = 'EEG'   % configurable in block dialog
        SourceID = 'HA-2016.03.01';
    end
    properties (Access = private)
        lib
        inlet
        nchannels
    end
    methods (Access = protected)

        function setupImpl(obj)
            % Look for the streams for 3.0s
            obj.lib = lsl_loadlib();
            results = lsl_resolve_byprop(obj.lib, 'type', obj.StreamType, 3.0);

            matched_stream = [];   % Initialize as "not found"
            for i = 1:length(results)
                info = results{i};
                % Search for the stream with specific source id
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

            obj.inlet = lsl_inlet(matched_stream);
            obj.nchannels = matched_stream.channel_count();
        end

        function sample = stepImpl(obj)
            [vec, ts] = obj.inlet.pull_sample(0.0);   % non-blocking pull

            % Output as column: [nCh x 1]
            if isempty(vec)
                sample = zeros(obj.nchannels, 1);
            else
                sample = vec(:);
            end
        end

        function out = getOutputSizeImpl(obj)
            out = [obj.nchannels 1];
        end
        function out = getOutputDataTypeImpl(~)
            out = 'double';
        end
        function out = isOutputComplexImpl(~)
            out = false;
        end
        function out = isOutputFixedSizeImpl(~)
            out = true;
        end
    end
end