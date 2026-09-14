% --- Load logged data ---
ts_raw = squeeze(out.tsout);

% Drop "no new sample" sentinel rows (ts == 0, from stepImpl's empty-vec branch)
valid = ts_raw ~= 0;
ts_raw = ts_raw(valid);

% --- Basic stats: instantaneous sample-to-sample intervals ---
dt = diff(ts_raw);              % time between consecutive samples
fs_instantaneous = 1 ./ dt;      % instantaneous rate estimate per interval

fprintf('--- Instantaneous interval stats ---\n');
fprintf('Mean dt: %.6f s (%.4f Hz)\n', mean(dt), 1/mean(dt));
fprintf('Std  dt: %.6f s\n', std(dt));
fprintf('Min  dt: %.6f s | Max dt: %.6f s\n', min(dt), max(dt));
fprintf('Jitter (std/mean): %.4f%%\n', 100*std(dt)/mean(dt));

% --- Global regression: overall effective sample rate across whole run ---
sample_idx = (0:length(ts_raw)-1)';
p = polyfit(sample_idx, ts_raw, 1);
effective_fs = 1/p(1);
fprintf('\n--- Global regression ---\n');
fprintf('Effective fs (whole run): %.6f Hz\n', effective_fs);

% --- Drift check: split into chunks, regress each separately ---
% This reveals whether the rate is stable over time, or trending
nChunks = 10;
chunkLen = floor(length(ts_raw)/nChunks);
chunk_fs = zeros(nChunks,1);
for k = 1:nChunks
    idx_range = (k-1)*chunkLen+1 : k*chunkLen;
    p_chunk = polyfit(sample_idx(idx_range), ts_raw(idx_range), 1);
    chunk_fs(k) = 1/p_chunk(1);
end

fprintf('\n--- Per-chunk effective fs (drift check) ---\n');
disp(chunk_fs);
fprintf('Range across chunks: %.6f Hz (max-min)\n', max(chunk_fs)-min(chunk_fs));

% --- Visual check ---
figure;
subplot(2,1,1);
plot(dt, '.');
yline(mean(dt), 'r--');
title('Sample-to-sample interval (dt) over run');
xlabel('Sample index'); ylabel('dt (s)');

subplot(2,1,2);
plot(chunk_fs, '-o');
title('Effective sample rate per chunk (drift check)');
xlabel('Chunk #'); ylabel('fs (Hz)');