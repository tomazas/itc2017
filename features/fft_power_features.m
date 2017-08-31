% each channel power sum is computed from FFT
function [train_data, test_data, train_lbl, test_lbl] = fft_power_features(p, train_trials, test_trials, train_labels, test_labels)
   
    disp('Computing FFT power features');
    
	function [features] = compute(trial_data)
	    [channels, samples, trials] = size(trial_data);
		features = zeros(1, channels, trials);
        
        n = 2^nextpow2(samples); %% fft vector length, power of 2 should be faster!
        
		for i=1:trials
			trial_samples = trial_data(:,:,i);
			for j=1:channels
				row = trial_samples(j,:);
                
                Y = fft(row, n); % will be padded with zeroes if not power of 2
                Y(1) = []; % remove the sum of all coefficients
                power = abs(Y(1:floor(n/2))).^2; % only half of the data is usable
				features(1, j, i) = sum(power); % sum of power (signal is already filtered in 7-30Hz band)
			end
        end
        features = log(features); % bring to some meaningful range
	end

	% build testing/training set
    train_data = compute(train_trials);
    test_data = compute(test_trials);
    
	% nothing changed - just copy labels
    train_lbl = train_labels;
    test_lbl = test_labels;
end


