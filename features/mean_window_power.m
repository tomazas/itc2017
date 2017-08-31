function [train_data, test_data, train_lbl, test_lbl] = mean_window_power(p, train_trials, test_trials, train_labels, test_labels)
   
    disp('Computing mean window power features');
    
	function [features] = compute(trial_data)
	    [channels, samples, trials] = size(trial_data);
		
		% divide the sample range into different count of windows
		windows = 1:channels; % window counts for whole samplerange
		nw = length(windows);
		features = zeros(nw, channels, trials);
		
		for i=1:trials
			trial_samples = trial_data(:,:,i);
			for j=1:channels
				row = trial_samples(j,:).^2; % strip_artifacts(trial_samples(j,:), fs);
				
				for k=1:nw
					wcount = windows(k);
					wsize = max(1, floor(samples/wcount)); % window size from count
					
					sums = zeros(1,wcount);
					for step=1:wcount
						si = 1+(step-1)*wsize;
						se = min(si+wsize, samples);
						sums(step) = mean(row(si:se));
					end
					
					features(k, j, i) = mean(sums);  % mean power of signal
				end
            end
            fprintf('trial %d/%d complete\n', i, trials);
		end
		features = log(features); % logarithm of mean values
    end

	% build testing/training set
    train_data = compute(train_trials);
    test_data = compute(test_trials);
    
	% nothing changed - just copy labels
    train_lbl = train_labels;
    test_lbl = test_labels;
end


