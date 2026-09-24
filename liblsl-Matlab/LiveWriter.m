classdef LiveWriter < matlab.System
    properties (Nontunable)
        FileName = 'live_log.bin';
        ReopenEvery = 1024;   % steps between close/reopen "flushes"
    end
    properties (Access = private)
        fid = -1
        count = 0
    end
    methods (Access = protected)
        function setupImpl(obj)
            obj.fid = fopen(obj.FileName, 'w');
            obj.count = 0;
        end
        function stepImpl(obj, t, u, m)
            fwrite(obj.fid, [t, u(:), m], 'double');   % one record per step
            obj.count = obj.count + 1;
            if mod(obj.count, obj.ReopenEvery) == 0
                fclose(obj.fid);                    % forces data to disk
                obj.fid = fopen(obj.FileName, 'a');
            end
        end
        function releaseImpl(obj)
            if obj.fid > 0, fclose(obj.fid); end
        end
    end
end