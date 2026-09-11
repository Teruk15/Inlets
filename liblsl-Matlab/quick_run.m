lib = lsl_loadlib();
results = lsl_resolve_byprop(lib, 'type', 'EEG', 3.0);
inlet = lsl_inlet(results{1});

for i = 1:20
    tic;
    [vec, ts] = inlet.pull_sample(0.1);
    elapsed = toc;
    if isempty(vec)
        fprintf('Timed out after %.4fs\n', elapsed);
    else
        fprintf('Got sample after %.4fs, ts=%.4f\n', elapsed, ts);
    end
end