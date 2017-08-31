% extract all trials from the EEG signal
% s - input and output(preprocessed) signal
% h - the signal information structure
% p - config structure
% idx - index used for logging
function [s, trials, csp_matrix] = get_trials(s,h,p)
    fprintf('Removing signal artifacts...\n');

    if isfield(p, 'preprocess_func') % apply some EEG pre-processing if required
		fprintf('EEG pre-processing using: %s\n', func2str(p.preprocess_func))
        s = p.preprocess_func(s, p);
    end

    num_samples = size(s,1);
    num_channels = size(s,2);
    signal_time = num_samples/h.SampleRate; % in seconds

    % show some info about signal loaded
    fprintf('Input signal dimensions (samples x channels): %d x %d, rate = %.2f Hz, time = %.2f sec\n', ...
        num_samples, num_channels, h.SampleRate, signal_time);
		
	%--------- extraction from event table 
    % Event | type | Description
    % 276 0x0114 Idling EEG (eyes open)
    % 277 0x0115 Idling EEG (eyes closed)

    % 768 0x0300 Start of a trial
    % 769 0x0301 Cue onset left (class 1)
    % 770 0x0302 Cue onset right (class 2)
    % 771 0x0303 Cue onset foot (class 3)
    % 772 0x0304 Cue onset tongue (class 4)
    % 783 0x030F Cue unknown

    % 1023 0x03FF Rejected trial
    % 1072 0x0430 Eye movements
    % 32766 0x7FFE Start of a new run

    num_events = length(h.EVENT.TYP);
    num_eyes_open = sum(h.EVENT.TYP == hex2dec('114'));
    num_eyes_closed = sum(h.EVENT.TYP == hex2dec('115'));
    num_trials = sum(h.EVENT.TYP == hex2dec('300'));
    num_left = sum(h.EVENT.TYP == hex2dec('301'));
    num_right = sum(h.EVENT.TYP == hex2dec('302'));
    num_foot = sum(h.EVENT.TYP == hex2dec('303'));
    num_tongue = sum(h.EVENT.TYP == hex2dec('304'));
    num_unknown = sum(h.EVENT.TYP == hex2dec('30F'));
    num_rejected = sum(h.EVENT.TYP == hex2dec('3FF'));
    num_eye_move = sum(h.EVENT.TYP == hex2dec('430'));
    num_runs = sum(h.EVENT.TYP == hex2dec('7FFE'));

    fprintf('Number of events: %d\n', num_events);
    fprintf('\trejected = %d\n', num_rejected);
    fprintf('\teyes_open = %d\n', num_eyes_open);
    fprintf('\teyes_closed = %d\n', num_eyes_closed);
    fprintf('\teye_move = %d\n', num_eye_move);
    fprintf('\truns(subjects) = %d\n', num_runs);

    fprintf('Trials: %d\n', num_trials);
    fprintf('\tleft = %d\n', num_left);
    fprintf('\tright = %d\n', num_right);
    fprintf('\tfoot = %d\n', num_foot);
    fprintf('\ttongue = %d\n', num_tongue);
    fprintf('\tunknown = %d\n', num_unknown);

    % do some checks
    event_sum = num_eyes_open+num_eyes_closed+num_trials+num_left+num_right+num_foot+num_tongue+ ...
        num_unknown+num_rejected+num_eye_move+num_runs;
    fprintf('Event sum: %d (unknown events: %d)\n', event_sum, num_events-event_sum);

    fprintf('Taking only %d EEG channels\n', length(p.usable_channels));
    s = s(:, p.usable_channels);

     % downsampling
    if p.downsampling > 1
        fprintf(1,'\tresampling (factor = %d)\n', p.downsampling);
        s = resample(s, p.downsampling, 1);   % downsampling by a factor of DIV;
    else
        p.downsampling = 1;
    end

    newrate = h.SampleRate/p.downsampling;
    new_time = signal_time/p.downsampling;

    fprintf('Original signal sample rate: %.2f Hz, time = %.2f sec, new downsampled: %.2f Hz, new time = %.2f sec\n', ...
      h.SampleRate, signal_time, newrate, new_time);

    h.SampleRate = newrate;
    h.EVENT.POS = round(h.EVENT.POS/p.downsampling);
    h.EVENT.DUR = round(h.EVENT.DUR/p.downsampling);
    h.TRIG      = round(h.TRIG/p.downsampling);

    csp_matrix = [];
    if p.csp == 1
        if ~isempty(p.csp_matrix)
            fprintf('Taking CSP matrix from params\n')
            csp_matrix = p.csp_matrix;
        else
            fprintf('Calculating new CSP matrix\n');
            csp_matrix = multiclass_csp(s,h,p);
        end
        s = s * csp_matrix; % filter using CSP matrix
    end

    pre = ceil(p.trim_low * h.SampleRate);
    post = ceil(p.trim_high * h.SampleRate);
    gap = 0; % no gaps between each trial (in samples)

    fprintf('\tExtract trigger and classlabels.\n');
    fprintf('Signal dimensions: %d x %d\n', size(s, 1), size(s, 2));

    [trials, sz] = trigg(s, h.TRIG, pre, post, gap);
    fprintf('  Segments cut - %d channels, %d samples/trial, trials: %d\n', sz(1), sz(2), sz(3));

    trials = reshape(trials, sz);
end