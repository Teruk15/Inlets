lib = lsl_loadlib();
results = lsl_resolve_byprop(lib, 'type', 'EEG', 3.0);
inlet = lsl_inlet(results{1});

n_warmup = 20;
n_trials = 2000;
timeout_val = 0.1;

% Warm-up (discarded)
for i = 1:n_warmup
    inlet.pull_sample(timeout_val);
end

elapsed_times = nan(1, n_trials);
timed_out = false(1, n_trials);

for i = 1:n_trials
    tic;
    [vec, ts] = inlet.pull_sample(timeout_val);
    elapsed_times(i) = toc;
    timed_out(i) = isempty(vec);
end

n_timeouts = sum(timed_out);
valid = elapsed_times(~timed_out);

fprintf('\n--- Results over %d trials (timeout=%.3fs) ---\n', n_trials, timeout_val);
fprintf('Timeouts: %d (%.2f%%)\n', n_timeouts, 100*n_timeouts/n_trials);
fprintf('Successful pulls - mean: %.4fs, median: %.4fs\n', mean(valid), median(valid));
fprintf('Successful pulls - 95th pct: %.4fs, 99th pct: %.4fs, max: %.4fs\n', ...
    prctile(valid, 95), prctile(valid, 99), max(valid));

figure;
histogram(elapsed_times, 100);
xlabel('Pull latency (s)'); ylabel('Count');
title(sprintf('pull\\_sample latency distribution (timeout=%.3fs)', timeout_val));