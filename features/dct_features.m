% each channel power is computed from dct coefficient sum
function [train_data, test_data, train_lbl, test_lbl] = dct_features(p, train_trials, test_trials, train_labels, test_labels)
   
    disp('Computing DCT features');
    
	function [features] = compute(trial_data)
	    [channels, samples, trials] = size(trial_data);
		features = zeros(1, channels, trials);
        
		for i=1:trials
			trial_samples = trial_data(:,:,i);
			for j=1:channels
				row = trial_samples(j,:);
				val = log(sum(dct(row).^2));  % non-log gives Inf for objective value!
				
				features(1, j, i) = val;
			end
        end
	end

	% build testing/training set
    train_data = compute(train_trials);
    test_data = compute(test_trials);
    
	% nothing changed - just copy labels
    train_lbl = train_labels;
    test_lbl = test_labels;
end


