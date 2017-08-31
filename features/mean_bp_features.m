function [train_data, test_data, train_lbl, test_lbl] = mean_bp_features(p, train_trials, test_trials, train_labels, test_labels)

    bands = [8 14; 19 24; 24 30];
    w = 0.5; % window width
    mode = 2;
    
%    mode       mode == 1 uses FIR filter and log10
%               mode == 2 uses Butterworth IIR filter and log10
%               mode == 3 uses FIR filter and ln
%               mode == 4 uses Butterworth IIR filter and ln
    function out = process(input)
	    % every channel contains signal power mean value
		[channels, samples, trials] = size(input);
	
	    out = zeros(1, channels, trials);
		for i=1:trials
			trial_samples = input(:,:,i);
			for j=1:channels
				row = trial_samples(j,:);
				bp = bandpower(row', p.fs, bands, w, mode);
				out(1, j, i) = mean(mean(bp));
			end
		end
	end
	
	train_data = process(train_trials);
	test_data = process(test_trials);
	
	% same
	train_lbl = train_labels;
	test_lbl = test_labels;
end


