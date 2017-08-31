% EEG signal preprocessing aux function - ensure we analyze only brain waves 7-30Hz
function s_filtered = preprocess_bandpass(s,p)
    %s(~isfinite(s)) = 0;
    s(isnan(s)) = 0;

    % REDUCE ARTIFACTS - filtering in specific range (7-30 Hz)
    [b,a] = butter(5, [7, 30]/(p.fs/2));
    s_filtered = filter(b, a, s);
end