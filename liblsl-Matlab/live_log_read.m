fid = fopen('live_log.bin','r');
raw = fread(fid, inf, 'double');
fclose(fid);

nCh = 96;

nRow = nCh + 2; % time + data + marker
nCol = floor(numel(raw) / nRow);

raw = reshape(raw(1:nRow*nCol), nRow, nCol);

raw = raw';
