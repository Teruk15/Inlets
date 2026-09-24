classdef LiveWriter < matlab.System
    properties (Nontunable)
        FileName = 'live_log.bin';
        ReopenEvery = 4800;   % Steps between close/reopen "flushes"
    end
    properties (Access = private)
        fid = -1
        count = 0
        t0 = NaN;
    end
    methods (Access = protected)
        function setupImpl(obj)
            obj.fid = fopen(obj.FileName, 'w');
            obj.count = 0;
        end
        function stepImpl(obj, t, u, m)
            if isnan(obj.t0)
                obj.t0 = double(t(1)); % To begin from t = 0
            end
            fwrite(obj.fid, [double(t(1)) - obj.t0; double(u(:)); double(m(:))], 'double'); 

            obj.count = obj.count + 1;
            if mod(obj.count, obj.ReopenEvery) == 0
                fclose(obj.fid);                    % Forces data to disk
                obj.fid = fopen(obj.FileName, 'a');
            end
        end
        function releaseImpl(obj)
            if obj.fid > 0, fclose(obj.fid); end
        end
    end
end